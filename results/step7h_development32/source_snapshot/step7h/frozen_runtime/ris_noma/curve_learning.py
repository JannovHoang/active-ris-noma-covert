"""7g prefix/model-size experiment; fixed validation, train probes, sealed selection."""
from pathlib import Path
import json
import platform
from copy import deepcopy

from filelock import FileLock
import numpy as np
import scipy
import torch

from . import curve_data as data
from .curve_fit import fit, read_fit
from .curve_models import load_model
from .curve_physics import STREAMS, audit_task, commit, read_record, record_path, confidence
from .continuous_dataset import read_lines
from .expanded_data import write_table
from .learning_data import read_json, sha256
from .learning_train import configure_runtime
from .multik_benchmark import verify_manifest
from .multik_core import physical_allocation
from .multik_data_core import at_target
from .multik_dataset import immutable, immutable_bytes
from .multik_graph import fit_normalizer
from .multik_models import predict
from .validation import family_constants

FORMAT = "nested_learning_experiment_7g_v1"
KINDS = ("gnn", "mlp")
SIZES = ("reference", "small")
RULES = ("weighted_loss", "physical")


def folder_for(output, size):
    return Path(output)/"stages"/f"train_{size:04d}"


def make_master(corpus, workers):
    corpus = Path(corpus).resolve()
    prior = read_json(corpus/"plan.json")
    data.validate_settings(prior["config"])
    if type(workers) is not int or not 1 <= workers <= 4:
        raise ValueError("Use 1..4 physics workers")
    return {"step":"7g","format":FORMAT,"config":prior["config"],"workers":workers,
            "corpus_plan_sha256":sha256(corpus/"plan.json"),"code_sha256":prior["code_sha256"],
            "model_parameters":prior["model_parameters"],"physics_streams":STREAMS,
            "versions":{"system":platform.system(),"python":platform.python_version(),
                "numpy":np.__version__,"scipy":scipy.__version__,"torch":str(torch.__version__)},
            "selection_uses":"validation only; weighted-loss snapshots then physical feasibility/rate",
            "test_used_for_selection":False,"reserved_test_generated":False,
            "physical_model_changed":False,"loss_changed":False}


def initialize(output, corpus, workers, resume):
    output, corpus = Path(output).resolve(), Path(corpus).resolve()
    if output == corpus or output in corpus.parents or corpus in output.parents:
        raise ValueError("Training output and corpus must be separate, non-nested folders")
    plan = make_master(corpus, workers)
    if output.exists() and any(output.iterdir()):
        if not resume:
            raise ValueError("Training output is nonempty; use --resume with the same plan")
        if read_json(output/"plan.json") != plan:
            raise ValueError("Changed code/config/corpus/workers/runtime; use a new output")
    output.mkdir(parents=True,exist_ok=True)
    immutable(output/"plan.json",plan)
    for name in plan["code_sha256"]:
        immutable_bytes(output/"source_snapshot"/name,(data.ROOT/name).read_bytes())
    for name in data.COPIES:
        immutable_bytes(output/name,(corpus/name).read_bytes())
    return plan


def make_stage(corpus, output, master, size):
    corpus = Path(corpus)
    prior = read_json(corpus/"plan.json")
    source = data.stage_folder(corpus,size)
    stage_summary = read_json(source/"summary.json")
    count = 2*2*len(master["config"]["training"]["seeds"])
    n_candidates = count*(len(master["config"]["training"]["snapshot_epochs"])+1)
    train_ids = prior["config"]["training_layout_ids"][:size]
    probe_ids = set(prior["config"]["training_layout_ids"][:prior["config"]["physics"]["train_probe_layout_count"]])
    jobs = data.stage_jobs(prior,size)
    plan = {"step":"7g","format":FORMAT,"train_layouts":size,"config":master["config"],
            "master_plan_sha256":sha256(Path(output)/"plan.json"),
            "corpus_plan_sha256":sha256(corpus/"plan.json"),
            "stage_completion_sha256":sha256(source/"completion.json"),
            "training_layout_ids":train_ids,"model_parameters":master["model_parameters"],
            "geometry_spec":prior["geometry_spec"],"physics_streams":STREAMS,
            "workers":master["workers"],
            "validation_jobs":[j for j in jobs if j["split"]=="validation"],
            "train_probe_jobs":[j for j in jobs if j["split"]=="train" and j["layout_id"] in probe_ids],
            "planned_counts":stage_summary["planned_counts"],"labeled_counts":stage_summary["labeled_counts"],
            "coverage":stage_summary["coverage"],
            "stage_validation_views":n_candidates+1,"stage_train_probe_views":count*2+1,
            "max_validation_views":prior["max_validation_candidates"]+1,
            "max_train_probe_views":prior["max_train_probe_views"]+1,
            "normalization":read_json(source/"normalization.json")["parameters"],
            "training_public_counts":prior["config"]["training_public_counts"],
            "reserved_test_generated":False,"test_used_for_selection":False}
    return plan


def cell_plan(folder, stage, size_name):
    c = data.fit_config(stage["config"],size_name)
    return {"step":"7g","config":c,"architecture_name":size_name,
            "normalization":stage["normalization"],"training_layout_ids":stage["training_layout_ids"],
            "training_public_counts":stage["training_public_counts"],
            "stage_completion_sha256":stage["stage_completion_sha256"],
            "corpus_plan_sha256":stage["corpus_plan_sha256"],
            "stage_plan_sha256":sha256(Path(folder)/"plan.json")}


def gather_candidates(folder, stage):
    refs = {}
    manifests = {}
    for size_name in SIZES:
        cell = Path(folder)/"cells"/size_name
        if read_json(cell/"plan.json") != cell_plan(folder,stage,size_name):
            raise ValueError("Cell architecture/config/prefix changed")
        cp = dict(read_json(cell/"plan.json"),plan_sha256=sha256(cell/"plan.json"))
        for kind in KINDS:
            for seed in stage["config"]["training"]["seeds"]:
                record = read_fit(cell,kind,seed,cp)
                path = cell/"fits"/f"{kind}_s{seed}"/"fit_manifest.json"
                manifests[path.relative_to(folder).as_posix()] = sha256(path)
                for key,value in record["candidates"].items():
                    if value["parameters"] != stage["model_parameters"][size_name][kind]:
                        raise ValueError("Checkpoint parameter count differs from declared size")
                    identity = size_name+"::"+key
                    refs[identity] = dict(value,architecture_name=size_name,
                        path=(Path("cells")/size_name/value["path"]).as_posix(),candidate_id=identity)
    if len(refs)+1 != stage["stage_validation_views"]:
        raise ValueError("Wrong declared checkpoint candidate count")
    return refs,manifests


def freeze_predictions(folder,phase,stage,rows,views,anchor):
    folder = Path(folder)
    models = {}
    for ref in views.values():
        if ref["kind"] != "pso_label_reference":
            models.setdefault(ref["candidate_id"],load_model(folder/ref["path"])[0])
    paths = []
    for job in stage[phase+"_jobs"]:
        part = [r for r in rows if r["layout_id"]==job["layout_id"] and r["n_public"]==job["n_public"]]
        if len(part) != len(job["cases"]):
            raise ValueError("Planned case denominator changed")
        path = record_path(folder,phase,"predictions",job)
        cached = read_record(folder,phase,"predictions",job)
        if cached is None:
            records = []
            for view_id,ref in views.items():
                allocations = ([r["allocation"] for r in part] if ref["kind"]=="pso_label_reference"
                    else [y.tolist() for y in predict(models[ref["candidate_id"]],[r["graph"] for r in part],stage["normalization"])])
                for row,allocation in zip(part,allocations):
                    records.append({key:row[key] for key in ("case_id","layout_id","n_public","power_dbm","xi","order")}
                        | {"split":job["split"],"phase":phase,"view_id":view_id,
                           "candidate_id":ref["candidate_id"],"kind":ref["kind"],
                           "architecture_name":ref.get("architecture_name","teacher"),
                           "seed":ref.get("seed"),"selection_rule":ref.get("selection_rule","candidate_pool"),
                           "allocation":allocation,"returned":allocation is not None})
            commit(path,{"job":job,"phase":phase,"rows":records,
                         "anchor_sha256":anchor,"plan_sha256":sha256(folder/"plan.json")})
        elif cached["anchor_sha256"] != anchor:
            raise ValueError("Frozen prediction anchor changed")
        paths.append(path)
    immutable(folder/(phase+"_predictions_frozen.json"),{
        "phase":phase,"plan_sha256":sha256(folder/"plan.json"),"anchor_sha256":anchor,
        "sha256":{p.relative_to(folder).as_posix():sha256(p) for p in paths}})


def phase_rows(folder,phase,stage):
    return [r for j in stage[phase+"_jobs"] for r in read_record(folder,phase,"audit",j)["rows"]]


def audit_phase(folder,phase,stage):
    jobs = [j for j in stage[phase+"_jobs"] if read_record(folder,phase,"audit",j) is None]
    data.execute_jobs(audit_task,[(str(folder),phase,j) for j in jobs],stage["workers"],phase)
    paths = [record_path(folder,phase,"audit",j) for j in stage[phase+"_jobs"]]
    immutable(Path(folder)/(phase+"_audit_frozen.json"),{
        "phase":phase,"plan_sha256":sha256(Path(folder)/"plan.json"),
        "predictions_frozen_sha256":sha256(Path(folder)/(phase+"_predictions_frozen.json")),
        "sha256":{p.relative_to(folder).as_posix():sha256(p) for p in paths}})


def score(rows):
    if not rows:
        raise ValueError("Empty physical validation")
    return {"cases":len(rows),"passed":sum(r["passed"] for r in rows),
            "pass_fraction":float(np.mean([r["passed"] for r in rows])),
            "gated_rate":float(np.mean([r["gated_bob_rate"] for r in rows])),
            "raw_rate":float(np.mean([r["bob_rate"] for r in rows]))}


def select(refs,rows,stage):
    if any(r.get("phase")!="validation" or r.get("split")!="validation" for r in rows):
        raise ValueError("Only validation rows may select 7g checkpoints")
    scores = {key:score([r for r in rows if r["candidate_id"]==key]) for key in refs}
    def physical_key(key):
        s,r = scores[key],refs[key]
        return (-s["passed"],-s["gated_rate"],r["validation_errors"]["weighted_loss"],r["epoch"],key)
    views = {}
    deployment = {}
    for architecture in SIZES:
        for kind in KINDS:
            for seed in stage["config"]["training"]["seeds"]:
                keys=[k for k,r in refs.items() if (r["architecture_name"],r["kind"],r["seed"])==(architecture,kind,seed)]
                chosen={"weighted_loss":min(keys,key=lambda k:(refs[k]["validation_errors"]["weighted_loss"],refs[k]["epoch"],k)),
                        "physical":min(keys,key=physical_key)}
                for rule,key in chosen.items():
                    identity=f"{architecture}::{kind}_s{seed}_{rule}"
                    views[identity]=dict(refs[key],view_id=identity,selection_rule=rule,validation_physics=scores[key])
            keys=[key for key,r in views.items() if (r["architecture_name"],r["kind"],r["selection_rule"])==(architecture,kind,"physical")]
            identity=min(keys,key=lambda k:physical_key(views[k]["candidate_id"]))
            deployment[f"{architecture}::{kind}"]=views[identity]
    return {"views":views,"deployment":deployment,"candidate_validation_scores":scores,
            "test_used_for_selection":False,"selection_policy":"max validation passes, gated rate, lower weighted loss, earlier epoch, ID"}


def teacher_view():
    return {"kind":"pso_label_reference","candidate_id":"pso_label_reference","selection_rule":"offline_teacher",
            "architecture_name":"teacher"}


def train_stage(corpus,output,size,workers=2,resume=False,plots=True):
    output,corpus=Path(output).resolve(),Path(corpus).resolve()
    output.parent.mkdir(parents=True,exist_ok=True)
    with FileLock(str(output)+".lock",timeout=0):
        if (output/"selection_final.json").exists():
            raise ValueError("7g is sealed; keep final selection fixed and use a new experiment for changes")
        data.check_corpus(corpus,size)
        master=initialize(output,corpus,workers,resume)
        previous=[n for n in master["config"]["training_sizes"] if n<size]
        if any(not (folder_for(output,n)/"completion.json").exists() for n in previous):
            raise ValueError("Complete smaller predeclared train prefixes first")
        folder=folder_for(output,size)
        stage=make_stage(corpus,output,master,size)
        folder.mkdir(parents=True,exist_ok=True)
        immutable(folder/"plan.json",stage)
        for name in ("source_physics_plan.json","source_dataset_plan.json"):
            immutable_bytes(folder/name,(corpus/name).read_bytes())
        source=data.stage_folder(corpus,size)
        for name in ("train.jsonl","validation.jsonl","normalization.json"):
            immutable_bytes(folder/"inputs"/name,(source/name).read_bytes())
        if (folder/"completion.json").exists():
            result=check_stage(corpus,output,size)
        else:
            train=read_lines(folder/"inputs/train.jsonl")
            validation=read_lines(folder/"inputs/validation.jsonl")
            labeled_train=[r for r in train if r["has_label"]]
            labeled_val=[r for r in validation if r["has_label"]]
            if any(not any(r["n_public"]==k for r in labeled_train) or not any(r["n_public"]==k for r in labeled_val)
                   for k in stage["training_public_counts"]):
                raise ValueError("Every trained K needs labeled train and validation cases")
            bounds=read_json(corpus/"source_physics_plan.json")["base_config"]["bounds"]
            for architecture in SIZES:
                cell=folder/"cells"/architecture
                cp=cell_plan(folder,stage,architecture)
                immutable(cell/"plan.json",cp)
                active=dict(cp,plan_sha256=sha256(cell/"plan.json"))
                for kind in KINDS:
                    for seed in stage["config"]["training"]["seeds"]:
                        print(f"7g train={size} model={architecture}/{kind} seed={seed}",flush=True)
                        fit(cell,kind,seed,active,bounds,labeled_train,labeled_val)
            refs,manifests=gather_candidates(folder,stage)
            immutable(folder/"fits_frozen.json",{"candidates":refs,"manifests":manifests,"plan_sha256":sha256(folder/"plan.json")})
            configure_runtime(0,1,"cpu")
            val_views=dict(refs,pso_label_reference=teacher_view())
            freeze_predictions(folder,"validation",stage,validation,val_views,sha256(folder/"fits_frozen.json"))
            audit_phase(folder,"validation",stage)
            rows=phase_rows(folder,"validation",stage)
            chosen=select(refs,rows,stage)
            immutable(folder/"selection_frozen.json",dict(chosen,
                plan_sha256=sha256(folder/"plan.json"),fits_frozen_sha256=sha256(folder/"fits_frozen.json"),
                validation_audit_frozen_sha256=sha256(folder/"validation_audit_frozen.json")))
            probe_views=dict(chosen["views"],pso_label_reference=teacher_view())
            freeze_predictions(folder,"train_probe",stage,train,probe_views,sha256(folder/"selection_frozen.json"))
            audit_phase(folder,"train_probe",stage)
            from .curve_analysis import export_stage
            result=export_stage(folder,stage,refs,chosen)
            paths=[p for p in sorted(folder.rglob("*")) if p.is_file() and p.name!="completion.json" and not p.name.endswith(".tmp")]
            immutable(folder/"completion.json",{"sha256":{p.relative_to(folder).as_posix():sha256(p) for p in paths}})
            result=check_stage(corpus,output,size)
        from .curve_analysis import refresh
        refresh(output,plots=plots)
        return result


def check_stage(corpus,output,size):
    corpus,output=Path(corpus),Path(output)
    master=read_json(output/"plan.json")
    if master["format"]!=FORMAT or master["reserved_test_generated"] or master["test_used_for_selection"]:
        raise ValueError("Wrong experiment/final-test policy")
    verify_manifest(data.ROOT,master["code_sha256"])
    verify_manifest(output/"source_snapshot",master["code_sha256"])
    if sha256(corpus/"plan.json")!=master["corpus_plan_sha256"]:
        raise ValueError("Corpus changed")
    folder=folder_for(output,size)
    verify_manifest(folder,read_json(folder/"completion.json")["sha256"])
    stage=read_json(folder/"plan.json")
    if stage!=make_stage(corpus,output,master,size):
        raise ValueError("Stage policy/normalizer/lineage changed")
    train=read_lines(folder/"inputs/train.jsonl")
    validation=read_lines(folder/"inputs/validation.jsonl")
    for name,rows in (("train",train),("validation",validation)):
        if rows!=read_lines(data.stage_folder(corpus,size)/(name+".jsonl")):
            raise ValueError("Training inputs changed from corpus")
        expected_ids=set(master["config"]["training_layout_ids"][:size] if name=="train" else master["config"]["validation_layout_ids"])
        if {r["layout_id"] for r in rows}!=expected_ids or any(r["role"]!=name or r["split"]!=name or r["n_public"] not in stage["training_public_counts"] for r in rows):
            raise ValueError("Layout/K split leakage")
        if any(set(r)!={"case_id","layout_id","n_public","split","regime","role","power_dbm","xi","graph","order","allocation","has_label"} for r in rows):
            raise ValueError("Unexpected audit/provenance fields in model records")
    norm=fit_normalizer([r["graph"] for r in train if r["has_label"]])
    if norm!=stage["normalization"]:
        raise ValueError("Normalizer is not fit on this train prefix only")
    refs,manifests=gather_candidates(folder,stage)
    if read_json(folder/"fits_frozen.json")!={"candidates":refs,"manifests":manifests,"plan_sha256":sha256(folder/"plan.json")}:
        raise ValueError("Frozen candidates changed")
    chosen=read_json(folder/"selection_frozen.json")
    expected=select(refs,phase_rows(folder,"validation",stage),stage)
    if any(chosen[k]!=v for k,v in expected.items()):
        raise ValueError("Checkpoint or deployment selection is not validation-only")
    if (chosen["plan_sha256"]!=sha256(folder/"plan.json") or chosen["fits_frozen_sha256"]!=sha256(folder/"fits_frozen.json")
            or chosen["validation_audit_frozen_sha256"]!=sha256(folder/"validation_audit_frozen.json")):
        raise ValueError("Selection barrier changed")
    for phase in STREAMS:
        frozen=read_json(folder/(phase+"_predictions_frozen.json"))
        audit_frozen=read_json(folder/(phase+"_audit_frozen.json"))
        allowed=validation if phase=="validation" else train
        lookup={r["case_id"]:r for r in allowed}
        views=dict(refs if phase=="validation" else chosen["views"],pso_label_reference=teacher_view())
        for job in stage[phase+"_jobs"]:
            proposals=read_record(folder,phase,"predictions",job)
            audited=read_record(folder,phase,"audit",job)
            path=record_path(folder,phase,"predictions",job)
            apath=record_path(folder,phase,"audit",job)
            if frozen["sha256"].get(path.relative_to(folder).as_posix())!=sha256(path) or audit_frozen["sha256"].get(apath.relative_to(folder).as_posix())!=sha256(apath):
                raise ValueError("Frozen physical records changed")
            if audited["predictions_frozen_sha256"]!=sha256(folder/(phase+"_predictions_frozen.json")):
                raise ValueError("Bank generated before matching predictions froze")
            family,alpha=confidence(stage,phase,job)
            expected_entropy=[read_json(corpus/"plan.json")["root_seed"],job["layout_id"],job["n_public"],STREAMS[phase]]
            if audited["metadata"]["bank_entropy"]!=expected_entropy or audited["family_size"]!=family or audited["alpha_per_group"]!=alpha:
                raise ValueError("Physical bank/family changed")
            seen=set()
            for before,after in zip(proposals["rows"],audited["rows"]):
                if any(after[k]!=v for k,v in before.items()):
                    raise ValueError("Physical audit changed a prediction")
                key=(before["view_id"],before["case_id"])
                if key in seen or before["view_id"] not in views or before["case_id"] not in lookup:
                    raise ValueError("Duplicate/wrong candidate or case")
                seen.add(key)
                reference=views[before["view_id"]]
                if (before["candidate_id"]!=reference["candidate_id"] or before["kind"]!=reference["kind"]
                        or before["architecture_name"]!=reference.get("architecture_name","teacher")
                        or before["seed"]!=reference.get("seed")):
                    raise ValueError("Prediction does not belong to its declared checkpoint")
                row=lookup[before["case_id"]]
                if any(before[k]!=row[k] for k in ("layout_id","n_public","power_dbm","xi","order")):
                    raise ValueError("Continuous input substituted")
                if before["returned"]:
                    physical_allocation(before["allocation"],job["n_public"],read_json(corpus/"source_physics_plan.json")["base_config"]["bounds"])
                    if any(after[k]!=v for k,v in at_target(after,row["xi"]).items()):
                        raise ValueError("Physical decision changed")
                    if after["constants"]!=family_constants(stage["config"]["physics"][phase+"_samples"],family,job["n_public"],alpha):
                        raise ValueError("Confidence constants differ")
                elif before["kind"]!="pso_label_reference" or row["has_label"]:
                    raise ValueError("Invalid no-return")
                if before["kind"]=="pso_label_reference" and before["allocation"]!=row["allocation"]:
                    raise ValueError("Teacher allocation changed")
            expected_keys={(v,r["case_id"]) for v in views for r in allowed
                           if r["layout_id"]==job["layout_id"] and r["n_public"]==job["n_public"]}
            if seen!=expected_keys or len(proposals["rows"])!=len(audited["rows"]):
                raise ValueError("Incomplete planned physical denominator")
    from .curve_analysis import stage_metrics
    metrics=stage_metrics(folder,stage,chosen)
    if read_json(folder/"summary.json")!=metrics:
        raise ValueError("Stage summary differs from stored decisions")
    return {"step":"7g","stage_complete":True,"train_layouts":size,"fits":len(manifests),
            "candidate_checkpoints":len(refs),"planned_counts":stage["planned_counts"],
            "labeled_counts":stage["labeled_counts"],"reserved_test_generated":False,
            "test_used_for_selection":False,"verified_files":len(read_json(folder/"completion.json")["sha256"])}


def completed_stages(output):
    output=Path(output)
    if not (output/"plan.json").exists():
        return []
    master=read_json(output/"plan.json")
    return [n for n in master["config"]["training_sizes"] if (folder_for(output,n)/"completion.json").exists()]


def final_choices(output,sizes):
    output=Path(output)
    options={}
    for kind in KINDS:
        candidates=[]
        for n in sizes:
            folder=folder_for(output,n)
            selected=read_json(folder/"selection_frozen.json")
            for name in SIZES:
                ref=selected["deployment"][f"{name}::{kind}"]
                candidates.append(dict(ref,train_layouts=n,
                    path=(folder.relative_to(output)/ref["path"]).as_posix()))
        winner=min(candidates,key=lambda r:(-r["validation_physics"]["passed"],
            -r["validation_physics"]["gated_rate"],r["validation_errors"]["weighted_loss"],
            r["parameters"],r["train_layouts"],r["seed"],r["candidate_id"]))
        options[kind]=winner
    return options


def seal(corpus,output,reason):
    output=Path(output)
    if not isinstance(reason,str) or not reason.strip():
        raise ValueError("Record why data collection stopped")
    with FileLock(str(output)+".lock",timeout=0):
        if (output/"completion.json").exists():
            return check(corpus,output)
        sizes=completed_stages(output)
        if len(sizes)<2:
            raise ValueError("Compare at least two nested sizes before sealing a learning curve")
        for n in sizes:
            check_stage(corpus,output,n)
        master=read_json(output/"plan.json")
        options=final_choices(output,sizes)
        immutable(output/"selection_final.json",{"step":"7g","deployment":options,"completed_sizes":sizes,
            "reason":reason.strip(),"master_plan_sha256":sha256(output/"plan.json"),
            "corpus_plan_sha256":sha256(Path(corpus)/"plan.json"),
            "stage_completions":{str(n):sha256(folder_for(output,n)/"completion.json") for n in sizes},
            "reserved_test_layout_ids":master["config"]["reserved_test_layout_ids"],
            "reserved_unseen_public_counts":master["config"]["reserved_unseen_public_counts"],
            "test_used_for_selection":False,"reserved_test_generated":False,
            "scope":"development selection; final test and hybrid PSO comparison belong to 7h"})
        from .curve_analysis import refresh
        refresh(output,plots=True)
        paths=[p for p in sorted(output.rglob("*")) if p.is_file() and p.name!="completion.json"
               and "figures" not in p.relative_to(output).parts and not p.name.endswith(".tmp")]
        # Include per-stage completion manifests explicitly.
        paths += [folder_for(output,n)/"completion.json" for n in sizes]
        immutable(output/"completion.json",{"sha256":{p.relative_to(output).as_posix():sha256(p) for p in paths}})
        return check(corpus,output)


def check(corpus,output,size=None):
    output=Path(output)
    stages=[size] if size is not None else completed_stages(output)
    checked=[check_stage(corpus,output,n) for n in stages]
    sealed=(output/"completion.json").exists()
    if sealed:
        verify_manifest(output,read_json(output/"completion.json")["sha256"])
        final=read_json(output/"selection_final.json")
        if final["completed_sizes"]!=completed_stages(output) or final["test_used_for_selection"] or final["reserved_test_generated"]:
            raise ValueError("Final selection/test policy changed")
        if (final["deployment"]!=final_choices(output,final["completed_sizes"])
                or final["master_plan_sha256"]!=sha256(output/"plan.json")
                or final["corpus_plan_sha256"]!=sha256(Path(corpus)/"plan.json")
                or final["reserved_test_layout_ids"]!=read_json(output/"plan.json")["config"]["reserved_test_layout_ids"]
                or any(final["stage_completions"][str(n)]!=sha256(folder_for(output,n)/"completion.json")
                       for n in final["completed_sizes"])):
            raise ValueError("Final validation selection or frozen lineage differs")
        for ref in final["deployment"].values():
            if sha256(output/ref["path"])!=ref["sha256"]:
                raise ValueError("Selected final checkpoint changed")
    return {"step":"7g","complete":sealed,"completed_sizes":stages,"stages":checked,
            "reserved_test_generated":False,"test_used_for_selection":False}
