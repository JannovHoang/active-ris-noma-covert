"""7c: mixed-K supervised fits, independent physical selection, frozen tests."""
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from copy import deepcopy
import csv
import multiprocessing as mp
from pathlib import Path
import platform

from filelock import FileLock
import numpy as np
import scipy
import torch

from .expanded_data import write_table
from .learning_data import read_json, sha256
from .learning_train import configure_runtime
from .multik_core import physical_allocation
from .multik_data_core import at_target
from .multik_dataset import (check as check_dataset, group_name, immutable, immutable_bytes,
                            read_committed, read_jsonl, receipt_path, write_json)
from .multik_graph import fit_normalizer, validate_graph
from .multik_models import predict
from .multik_learning_fit import fit, read_fit
from .multik_learning_models import build_model, load_model
from .multik_learning_physics import STREAMS, audit_task

ROOT=Path(__file__).resolve().parents[1]
KINDS=("gnn","mlp")
RULES=("weighted_loss","physical")
NUMERIC_FILES=("run_step7c.py","ris_noma/multik_learning_models.py","ris_noma/multik_learning_fit.py",
               "ris_noma/multik_learning_physics.py","ris_noma/multik_learning.py")
SOURCE_FILES=("completion.json","plan.json","source_physics_plan.json","feature_schema.json",
              "graphs/train.jsonl","graphs/validation.jsonl","graphs/test.jsonl","graphs/test_unseen_k.jsonl")


def validate_config(settings):
    c=deepcopy(settings)
    expected={"seeds","epochs","snapshot_epochs","batch_size","evaluation_batch_size","learning_rate","weight_decay",
              "gradient_clip","hidden","layers","mlp_inner","loss_weights","resume_every","validation_physics_samples","test_physics_samples","alpha"}
    if set(c)!=expected:
        raise ValueError("Unexpected 7c training fields")
    if not isinstance(c["seeds"],list) or not c["seeds"] or any(type(s) is not int or s<0 for s in c["seeds"]) or c["seeds"]!=sorted(set(c["seeds"])):
        raise ValueError("Use sorted distinct nonnegative training seeds")
    for key in ("epochs","batch_size","evaluation_batch_size","hidden","layers","mlp_inner","resume_every"):
        if type(c[key]) is not int or c[key]<1:
            raise ValueError("Positive integer 7c training dimensions required")
    snapshots=c["snapshot_epochs"]
    if not isinstance(snapshots,list) or not snapshots or any(type(e) is not int or not 1<=e<=c["epochs"] for e in snapshots) or snapshots!=sorted(set(snapshots)) or snapshots[-1]!=c["epochs"]:
        raise ValueError("Snapshot epochs must be distinct/increasing and include the final epoch")
    for key in ("learning_rate","gradient_clip"):
        if type(c[key]) not in (float,int) or not np.isfinite(c[key]) or c[key]<=0:
            raise ValueError("Finite positive optimization coefficients required")
    if type(c["weight_decay"]) not in (float,int) or not np.isfinite(c["weight_decay"]) or c["weight_decay"]<0:
        raise ValueError("Finite nonnegative weight decay required")
    if set(c["loss_weights"])!={"mb","public_total","s"} or any(type(w) not in (float,int) or not np.isfinite(w) or w<=0 for w in c["loss_weights"].values()):
        raise ValueError("Declare positive allocation loss weights")
    for key in ("validation_physics_samples","test_physics_samples"):
        if type(c[key]) is not int or c[key]<1000:
            raise ValueError("Use at least 1000 samples per physical bank")
    if type(c["alpha"]) not in (float,int) or not np.isfinite(c["alpha"]) or not 0<c["alpha"]<1:
        raise ValueError("Require finite 0<alpha<1")
    return c


def load_role(dataset,name,dataset_plan):
    if name not in ("train","validation","test","test_unseen_k"):
        raise ValueError("Unknown dataset split")
    raw=read_jsonl(Path(dataset)/"graphs"/f"{name}.jsonl")
    role="test" if name=="test_unseen_k" else name
    regime="unseen_k" if name=="test_unseen_k" else "seen_k"
    ks=dataset_plan["config"]["unseen_public_counts" if name=="test_unseen_k" else "training_public_counts"]
    keys=set(); rows=[]
    fields=("layout_id","n_public","split","regime","power_dbm","xi","graph","allocation","order")
    for row in raw:
        if (row["split"]!=role or row["regime"]!=regime or row["layout_id"] not in dataset_plan["config"]["splits"][role]
                or row["n_public"] not in ks or not row["label_available"] or row["heldout_used_for_label_filter"] is not False):
            raise ValueError("Graph split/K/label policy differs")
        mark=(row["layout_id"],row["n_public"],row["power_dbm"],row["xi"])
        if mark in keys or validate_graph(row["graph"])!=row["n_public"]:
            raise ValueError("Duplicate/invalid graph case")
        keys.add(mark)
        # Explicitly discard heldout_passed and every audit field. All selected
        # labels enter the split regardless of their recorded label-audit flag.
        rows.append({key:row[key] for key in fields})
    return rows


def inspect(dataset,settings):
    dataset=Path(dataset).resolve()
    status=check_dataset(dataset)
    prior=read_json(dataset/"plan.json")
    source=read_json(dataset/"source_physics_plan.json")
    c=validate_config(settings)
    train=load_role(dataset,"train",prior)
    validation=load_role(dataset,"validation",prior)
    for name,rows in (("train",train),("validation",validation)):
        if not rows or any(not any(r["n_public"]==k for r in rows) for k in prior["config"]["training_public_counts"]):
            raise ValueError(f"Insufficient labeled K coverage in {name}")
        for row in rows:
            physical_allocation(row["allocation"],row["n_public"],source["base_config"]["bounds"])
    normalization=fit_normalizer([r["graph"] for r in train])
    maximum=2*len(c["seeds"])*(len(c["snapshot_epochs"])+1)
    val_jobs=[j for j in prior["jobs"] if j["split"]=="validation"]
    test_jobs=[j for j in prior["jobs"] if j["split"]=="test"]
    tracked=tuple(dict.fromkeys((*prior["code_sha256"],*NUMERIC_FILES)))
    configure_runtime(0,1,"cpu")
    parameters={kind:sum(p.numel() for p in build_model(kind,source["base_config"]["bounds"],c).parameters()) for kind in KINDS}
    counts={name:row["labels"] for name,row in status["counts"].items()}
    plan={"step":"7c","format":"mixed_k_learning_protocol_7c_v1","config":c,
          "dataset_sha256":{name:sha256(dataset/name) for name in SOURCE_FILES},
          "code_sha256":{name:sha256(ROOT/name) for name in tracked},
          "versions":{"system":platform.system(),"python":platform.python_version(),"numpy":np.__version__,"scipy":scipy.__version__,"torch":str(torch.__version__)},
          "normalization":normalization,"labeled_counts":counts,"training_public_counts":prior["config"]["training_public_counts"],
          "unseen_public_counts":prior["config"]["unseen_public_counts"],"power_dbm":prior["config"]["power_dbm"],"xi":prior["config"]["xi"],
          "validation_jobs":val_jobs,"test_jobs":test_jobs,"streams":STREAMS,"model_parameters":parameters,
          "max_validation_candidates":maximum,"max_test_views":2*len(c["seeds"])*len(RULES)+1,
          "planned_validation_rows":maximum*len(validation),
          "planned_test_rows":(2*len(c["seeds"])*len(RULES)+1)*(counts["test"]+counts["test_unseen_k"]),
          "normalization_fitted_on":"train only; one shared normalizer across K",
          "loss":"(mb_weight*e_mb^2 + public_total*mean_public_error^2 + s_weight*e_s^2)/sum_weights; padding ignored",
          "architecture_policy":"unchanged 7a GNN; shared pointwise pooled MLP with matching heads, no messages/edge values, variable K",
          "checkpoint_pool":"fixed snapshot epochs plus best weighted validation loss per fit; all fits/predictions freeze before fresh physical bank",
          "physical_selection":"per kind/seed: max validation passes, gated Bob rate, lower weighted validation loss, earlier epoch, candidate ID",
          "weighted_selection":"per kind/seed: min weighted validation loss, earlier epoch, candidate ID",
          "deployment_selection":"per kind among physical selections: validation passes, gated rate, lower weighted loss, seed",
          "confidence_policy":"alpha/(2*groups) for validation or test; family covers every declared checkpoint/view, power and DEP",
          "test_policy":"all selected views and predictions frozen before fresh test bank; no test seed/checkpoint tuning",
          "physical_gradient":False,"test_used_for_training_or_selection":False,"device":"cpu","threads_per_process":1}
    return plan,source,prior,train,validation


def freeze(output,name,paths):
    immutable(Path(output)/f"{name}.json",{"phase":name,"sha256":{p.relative_to(output).as_posix():sha256(p) for p in paths}})


def candidates(output,plan):
    refs={}; manifests={}
    for kind in KINDS:
        for seed in plan["config"]["seeds"]:
            record=read_fit(output,kind,seed,plan)
            refs.update(record["candidates"])
            path=Path(output)/"fits"/f"{kind}_s{seed}"/"fit_manifest.json"
            manifests[path.relative_to(output).as_posix()]=sha256(path)
    if len(refs)!=plan["max_validation_candidates"]:
        raise ValueError("Checkpoint pool size differs from the declared budget")
    immutable(Path(output)/"fits_frozen.json",{"candidates":refs,"manifests":manifests,"plan_sha256":plan["plan_sha256"]})
    return refs


def commit_predictions(output,phase,jobs,rows,views,normalizer,anchor):
    models={}
    for view in views.values():
        if view["candidate_id"] not in models:
            models[view["candidate_id"]]=load_model(Path(output)/view["path"])[0]
    paths=[]
    for job in jobs:
        path=Path(output)/f"{phase}_predictions"/group_name(job)
        cached=read_committed(path)
        if cached is None:
            subset=[r for r in rows if (r["layout_id"],r["n_public"])==(job["layout_id"],job["n_public"])]
            if not subset:
                raise ValueError("A declared physical job has no graph input")
            predictions=[]; allocations={}
            for view_id,view in views.items():
                key=view["candidate_id"]
                if key not in allocations:
                    allocations[key]=predict(models[key],[r["graph"] for r in subset],normalizer)
                for row,y in zip(subset,allocations[key]):
                    predictions.append({name:row[name] for name in ("layout_id","n_public","split","regime","power_dbm","xi","order")}
                        | {"view_id":view_id,"candidate_id":key,"kind":view["kind"],"seed":view["seed"],
                           "selection_rule":view.get("selection_rule","candidate_pool"),"allocation":y.tolist()})
            if phase=="test":
                for row in subset:
                    predictions.append({name:row[name] for name in ("layout_id","n_public","split","regime","power_dbm","xi","order")}
                        | {"view_id":"pso_label_reference","candidate_id":"pso_label_reference","kind":"pso_label_reference","seed":None,
                           "selection_rule":"frozen_7b_label","allocation":row["allocation"]})
            write_json(path,{**job,"phase":phase,"anchor_sha256":anchor,"rows":predictions})
            write_json(receipt_path(path),{"sha256":sha256(path)})
        elif cached["anchor_sha256"]!=anchor:
            raise ValueError("Committed predictions belong to another checkpoint selection")
        paths.append(path)
    freeze(output,f"{phase}_predictions_frozen",paths)


def audit_phase(output,phase,jobs,workers):
    pending=[job for job in jobs if read_committed(Path(output)/f"{phase}_audit"/group_name(job)) is None]
    done=len(jobs)-len(pending)
    print(f"7c {phase} physics: {done}/{len(jobs)} groups already committed",flush=True)
    if workers==1:
        for job in pending:
            name=audit_task((str(output),phase,job));done+=1
            print(f"7c {phase} physics {done}/{len(jobs)}: {name}",flush=True)
    elif pending:
        executor=ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context("spawn"))
        active={}; queue=iter(pending)
        def submit():
            job=next(queue,None)
            if job is not None:
                active[executor.submit(audit_task,(str(output),phase,job))]=job
        try:
            for _ in range(workers):submit()
            while active:
                finished,_=wait(active,return_when=FIRST_COMPLETED)
                for future in finished:
                    name=future.result();del active[future];done+=1
                    print(f"7c {phase} physics {done}/{len(jobs)}: {name}",flush=True)
                    submit()
        finally:
            for future in active:future.cancel()
            executor.shutdown(wait=True,cancel_futures=True)
    paths=[Path(output)/f"{phase}_audit"/group_name(job) for job in jobs]
    if any(read_committed(path) is None for path in paths):
        raise ValueError("Cannot freeze incomplete physical audit")
    freeze(output,f"{phase}_audit_frozen",paths)


def aggregate(rows):
    return {"cases":len(rows),"passed":sum(r["passed"] for r in rows),
            "pass_fraction":float(np.mean([r["passed"] for r in rows])),
            "bob_rate":float(np.mean([r["bob_rate"] for r in rows])),
            "gated_rate":float(np.mean([r["gated_bob_rate"] for r in rows]))}


def select_checkpoints(refs,rows,plan):
    if not rows or any(r["split"]!="validation" or r["regime"]!="seen_k" for r in rows):
        raise ValueError("Only seen-K validation physics may select checkpoints")
    scores={}
    for key,ref in refs.items():
        group=[r for r in rows if r["candidate_id"]==key]
        if len(group)!=plan["labeled_counts"]["validation"]:
            raise ValueError("A checkpoint has an incomplete physical validation")
        scores[key]=aggregate(group)
    def physical_key(key):
        return (-scores[key]["passed"],-scores[key]["gated_rate"],refs[key]["validation_errors"]["weighted_loss"],refs[key]["epoch"],key)
    views={}
    for kind in KINDS:
        for seed in plan["config"]["seeds"]:
            keys=[key for key,ref in refs.items() if ref["kind"]==kind and ref["seed"]==seed]
            if not keys:raise ValueError("Missing a model/seed in the checkpoint pool")
            chosen={"weighted_loss":min(keys,key=lambda key:(refs[key]["validation_errors"]["weighted_loss"],refs[key]["epoch"],key)),
                    "physical":min(keys,key=physical_key)}
            for rule,key in chosen.items():
                views[f"{kind}_s{seed}_{rule}"]={**refs[key],"candidate_id":key,"selection_rule":rule,"validation_physics":scores[key]}
    deployment={}
    for kind in KINDS:
        ids=[key for key,value in views.items() if value["kind"]==kind and value["selection_rule"]=="physical"]
        selected=min(ids,key=lambda key:(*physical_key(views[key]["candidate_id"])[:3],views[key]["seed"]))
        deployment[kind]={"view_id":selected,**views[selected]}
    return {"views":views,"deployment":deployment,"candidate_validation_scores":scores,
            "test_used_for_selection":False,"selection_policy":plan["physical_selection"]}


def phase_rows(output,phase,jobs):
    return [row for job in jobs for row in read_committed(Path(output)/f"{phase}_audit"/group_name(job))["rows"]]


def export_metrics(output,plan,selection,rows):
    references={(r["layout_id"],r["n_public"],r["power_dbm"],r["xi"]):r["allocation"] for r in rows if r["kind"]=="pso_label_reference"}
    errors=[]
    for row in rows:
        if row["kind"]=="pso_label_reference":continue
        truth=references[(row["layout_id"],row["n_public"],row["power_dbm"],row["xi"])]
        delta=np.asarray(row["allocation"])-np.asarray(truth)
        w=plan["config"]["loss_weights"]
        errors.append({key:row[key] for key in ("view_id","kind","seed","selection_rule","layout_id","n_public","regime","power_dbm","xi")}
            | {"case_mse":float(np.mean(delta**2)),"weighted_loss":float((w["mb"]*delta[0]**2+w["public_total"]*np.mean(delta[1:-1]**2)+w["s"]*delta[-1]**2)/sum(w.values())),
               "mb_mae":float(abs(delta[0])),"public_mae":float(np.mean(abs(delta[1:-1]))),"s_mae":float(abs(delta[-1]))})
    write_table(Path(output)/"test_allocation_errors.csv",errors)
    by_view=[]; by_k=[]; by_dep=[]
    for name in (*selection["views"],"pso_label_reference"):
        subset=[r for r in rows if r["view_id"]==name]
        if not subset:raise ValueError("Missing selected test view")
        meta={key:subset[0][key] for key in ("view_id","kind","seed","selection_rule")}
        error=[r for r in errors if r["view_id"]==name]
        measures={key:float(np.mean([r[key] for r in error])) for key in ("case_mse","weighted_loss","mb_mae","public_mae","s_mae")} if error else {key:None for key in ("case_mse","weighted_loss","mb_mae","public_mae","s_mae")}
        by_view.append({**meta,**aggregate(subset),**measures})
        for k in sorted({r["n_public"] for r in subset}):
            by_k.append({**meta,"n_public":k,"regime":"unseen_k" if k in plan["unseen_public_counts"] else "seen_k",
                         **aggregate([r for r in subset if r["n_public"]==k])})
        for xi in plan["xi"]:
            by_dep.append({**meta,"xi":xi,**aggregate([r for r in subset if r["xi"]==xi])})
    pooled=[]
    for kind in KINDS:
        for rule in RULES:
            seed_rows=[r for r in by_view if r["kind"]==kind and r["selection_rule"]==rule]
            entry={"kind":kind,"selection_rule":rule,"training_seeds":len(seed_rows),"cases_per_seed":seed_rows[0]["cases"]}
            for field in ("pass_fraction","gated_rate","case_mse"):
                values=[r[field] for r in seed_rows]
                entry[field+"_mean"]=float(np.mean(values))
                entry[field+"_std"]=float(np.std(values,ddof=1)) if len(values)>1 else 0.
            pooled.append(entry)
    for name,data in (("test_metrics.csv",by_view),("test_by_k.csv",by_k),("test_by_dep.csv",by_dep),("test_seed_summary.csv",pooled)):
        write_table(Path(output)/name,data)
    return {"by_view":by_view,"by_k":by_k,"by_dep":by_dep,"across_seeds":pooled}


def run(dataset,output,settings,workers=1,resume=False):
    if type(workers) is not int or not 1<=workers<=8:
        raise ValueError("Use 1..8 physics workers; start with 4 on a 32GB laptop")
    dataset,output=Path(dataset).resolve(),Path(output).resolve()
    if dataset==output or dataset in output.parents or output in dataset.parents:
        raise ValueError("Use a separate non-nested 7c result output")
    plan,source,dataset_plan,train,validation=inspect(dataset,settings)
    output.parent.mkdir(parents=True,exist_ok=True)
    with FileLock(str(output.with_name(f".{output.name}.lock")),timeout=0):
        if output.exists() and any(output.iterdir()):
            if not resume or not (output/"plan.json").is_file() or read_json(output/"plan.json")!=plan:
                raise ValueError("Nonempty/changed 7c output; use --resume with identical data/code/config/environment or a new output")
            if (output/"completion.json").exists():return check(output)
        output.mkdir(parents=True,exist_ok=True)
        immutable(output/"plan.json",plan)
        immutable(output/"normalization.json",plan["normalization"])
        for source_name,target_name in (("plan.json","source_dataset_plan.json"),("completion.json","source_dataset_completion.json"),
                                       ("source_physics_plan.json","source_physics_plan.json"),("feature_schema.json","source_feature_schema.json")):
            immutable_bytes(output/target_name,(dataset/source_name).read_bytes())
        active=dict(plan,plan_sha256=sha256(output/"plan.json"))
        for kind in KINDS:
            for seed in plan["config"]["seeds"]:
                fit(output,kind,seed,active,source["base_config"]["bounds"],train,validation)
        refs=candidates(output,active)
        val_views={key:dict(ref,candidate_id=key) for key,ref in refs.items()}
        commit_predictions(output,"validation",plan["validation_jobs"],validation,val_views,plan["normalization"],sha256(output/"fits_frozen.json"))
        audit_phase(output,"validation",plan["validation_jobs"],workers)
        val_rows=phase_rows(output,"validation",plan["validation_jobs"])
        selection=select_checkpoints(refs,val_rows,plan)
        immutable(output/"selection_frozen.json",selection|{"validation_audit_frozen_sha256":sha256(output/"validation_audit_frozen.json"),
                                                           "fits_frozen_sha256":sha256(output/"fits_frozen.json")})
        # Test labels/predictions only enter after all weights and selections freeze.
        test=load_role(dataset,"test",dataset_plan)+load_role(dataset,"test_unseen_k",dataset_plan)
        commit_predictions(output,"test",plan["test_jobs"],test,selection["views"],plan["normalization"],sha256(output/"selection_frozen.json"))
        audit_phase(output,"test",plan["test_jobs"],workers)
        test_rows=phase_rows(output,"test",plan["test_jobs"])
        metrics=export_metrics(output,plan,selection,test_rows)
        if not all(sha256(dataset/name)==digest for name,digest in plan["dataset_sha256"].items()):
            raise ValueError("Source 7b changed during training/evaluation")
        result={"step":"7c","complete":True,"labeled_counts":plan["labeled_counts"],"fits":2*len(plan["config"]["seeds"]),
                "candidate_checkpoints":len(refs),"selected_views":len(selection["views"]),"model_parameters":plan["model_parameters"],
                "validation_rows":len(val_rows),"test_rows":len(test_rows),"metrics":metrics,"deployment":selection["deployment"],
                "test_used_for_training_or_selection":False,"physical_gradient":False,"source_7b_unchanged":True,
                "scope":"direct GNN/pooled-MLP predictions and offline PSO label reference; no warm-start or timing benchmark yet"}
        write_json(output/"summary.json",result)
        files=[p for p in sorted(output.rglob("*")) if p.is_file() and p.name!="completion.json" and not p.name.endswith(".tmp") and "figures" not in p.relative_to(output).parts]
        immutable(output/"completion.json",{"sha256":{p.relative_to(output).as_posix():sha256(p) for p in files}})
        return check(output)


def check(output,dataset=None):
    output=Path(output).resolve()
    manifest=read_json(output/"completion.json")["sha256"]
    required={"plan.json","normalization.json","fits_frozen.json","selection_frozen.json","summary.json",
              "validation_predictions_frozen.json","validation_audit_frozen.json","test_predictions_frozen.json","test_audit_frozen.json"}
    if not required<=set(manifest):raise ValueError("Incomplete 7c result")
    for name,digest in manifest.items():
        path=(output/name).resolve()
        if not path.is_relative_to(output) or not path.is_file() or sha256(path)!=digest:
            raise ValueError(f"Changed/missing 7c result: {name}")
    plan=read_json(output/"plan.json")
    validate_config(plan["config"])
    active=dict(plan,plan_sha256=sha256(output/"plan.json"))
    for source_name,target_name in (("plan.json","source_dataset_plan.json"),("completion.json","source_dataset_completion.json"),
                                   ("source_physics_plan.json","source_physics_plan.json"),("feature_schema.json","source_feature_schema.json")):
        if sha256(output/target_name)!=plan["dataset_sha256"][source_name]:raise ValueError("Copied 7b provenance differs")
    if read_json(output/"normalization.json")!=plan["normalization"]:
        raise ValueError("Normalization changed")
    refs={}
    for kind in KINDS:
        for seed in plan["config"]["seeds"]:refs.update(read_fit(output,kind,seed,active)["candidates"])
    frozen=read_json(output/"fits_frozen.json")
    if frozen["candidates"]!=refs or frozen["plan_sha256"]!=active["plan_sha256"]:
        raise ValueError("Frozen checkpoint pool differs")
    for phase in ("validation","test"):
        jobs=plan[f"{phase}_jobs"]
        for stage in ("predictions","audit"):
            frozen_stage=read_json(output/f"{phase}_{stage}_frozen.json")
            expected={f"{phase}_{stage}/{group_name(job)}" for job in jobs}
            if set(frozen_stage["sha256"])!=expected or any(manifest.get(n)!=h for n,h in frozen_stage["sha256"].items()):
                raise ValueError("Frozen prediction/audit group set differs")
        for job in jobs:
            proposals=read_committed(output/f"{phase}_predictions"/group_name(job))
            audited=read_committed(output/f"{phase}_audit"/group_name(job))
            anchor=sha256(output/("fits_frozen.json" if phase=="validation" else "selection_frozen.json"))
            if proposals["anchor_sha256"]!=anchor or audited["predictions_frozen_sha256"]!=sha256(output/f"{phase}_predictions_frozen.json"):
                raise ValueError("Checkpoint-prediction-audit chain differs")
            if len(proposals["rows"])!=len(audited["rows"]):raise ValueError("Missing physical audit rows")
            pool=plan["max_validation_candidates"] if phase=="validation" else plan["max_test_views"]
            family=pool*len(plan["power_dbm"])*len(plan["xi"])
            samples=plan["config"]["validation_physics_samples" if phase=="validation" else "test_physics_samples"]
            if (audited["family_size"]!=family or audited["bank_samples"]!=samples
                    or audited["alpha_per_group"]!=plan["config"]["alpha"]/(2*len(jobs))):
                raise ValueError("Physical audit family/sample budget differs")
            for before,after in zip(proposals["rows"],audited["rows"]):
                if any(after[key]!=value for key,value in before.items()) or any(after[key]!=value for key,value in job.items()):
                    raise ValueError("Physical audit changed predictions/case identity")
                decision=at_target(after,after["xi"])
                if any(after[key]!=value for key,value in decision.items()):raise ValueError("Saved physical decision differs from its constraints")
    val_rows=phase_rows(output,"validation",plan["validation_jobs"])
    expected=select_checkpoints(refs,val_rows,plan)
    selected=read_json(output/"selection_frozen.json")
    if (any(selected[key]!=value for key,value in expected.items())
            or selected["validation_audit_frozen_sha256"]!=sha256(output/"validation_audit_frozen.json")
            or selected["fits_frozen_sha256"]!=sha256(output/"fits_frozen.json")):
        raise ValueError("Checkpoint choice differs from validation-only rule")
    result=read_json(output/"summary.json")
    if not result["complete"] or result["test_used_for_training_or_selection"] or result["physical_gradient"]:
        raise ValueError("Unexpected 7c learning/test policy")
    test_rows=phase_rows(output,"test",plan["test_jobs"])
    if len(val_rows)!=plan["planned_validation_rows"] or len(test_rows)!=plan["planned_test_rows"]:
        raise ValueError("Validation/test budget differs from plan")
    for metric in result["metrics"]["by_view"]:
        actual=aggregate([row for row in test_rows if row["view_id"]==metric["view_id"]])
        if any(metric[key]!=value for key,value in actual.items()):raise ValueError("Test aggregate differs from its rows")
    if dataset is not None:
        dataset=Path(dataset)
        if any(sha256(dataset/name)!=digest for name,digest in plan["dataset_sha256"].items()):raise ValueError("Source 7b differs")
        train=load_role(dataset,"train",read_json(dataset/"plan.json"))
        if fit_normalizer([r["graph"] for r in train])!=plan["normalization"]:raise ValueError("Normalizer was not fit on these train graphs")
    return result|{"verified_files":len(manifest)}


def status(output):
    output=Path(output)
    plan=read_json(output/"plan.json")
    return {"step":"7c","complete":(output/"completion.json").is_file(),
            "completed_fits":sum((output/"fits"/f"{kind}_s{seed}"/"fit_manifest.json").is_file() for kind in KINDS for seed in plan["config"]["seeds"]),
            "optimizer_resume_epochs":{f"{kind}_s{seed}":read_json(output/"fits"/f"{kind}_s{seed}"/"resume.json")["epoch"]
                for kind in KINDS for seed in plan["config"]["seeds"] if (output/"fits"/f"{kind}_s{seed}"/"resume.json").exists()},
            "committed_physics_groups":{phase:sum(read_committed(output/f"{phase}_audit"/group_name(job)) is not None for job in plan[f"{phase}_jobs"]) for phase in ("validation","test")}}
