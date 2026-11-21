from pathlib import Path
from collections import Counter
import statistics
import time
import numpy as np
from filelock import FileLock

from .common import (ROOT,read,write,digest,freeze,verify,source,settings,initialize,metrics,
                     allocation_errors,committed,code_hashes)
from .fits import prepare_models,read_models
from .physics import freeze_predictions,audit,phase_rows,context,variants
from .analysis import original_failures,select,summarize,plot
from . import baseline
from ris_noma.curve_models import load_model
from ris_noma.multik_models import predict
from ris_noma.stable_assessment import raw_assessment
from ris_noma.multik_data_core import at_target
from ris_noma.multik_graph import fit_normalizer
from ris_noma.learning_train import configure_runtime


def inspect(corpus,training,config):
    return workload(source(corpus,training,config))


def workload(s):
    c=s["settings"]
    cases={p:sum(len(j["cases"]) for j in s["jobs"][p]) for p in ("train","validation")}
    return {"scope":c["scope"],"source_stage_valid":True,"prefixes":c["sizes"],
        "train_labeled":sum(r["has_label"] for r in s["train"]),"validation_labeled":sum(r["has_label"] for r in s["validation"]),
        "new_fits":2*2*len(c["seeds"]),"reused_small32_fits":2*len(c["seeds"]),
        "validation_views":s["family_views"],"physical_case_counts":cases,
        "validation_physics_rows":cases["validation"]*s["family_views"],
        "train_physics_rows_upper_bound":cases["train"]*(2*len(s["source_views"])+2+3*2*len(c["seeds"])+1),
        "physics_samples":c["physics_samples"],"pool_size":s["pool_size"],"alpha_per_group":s["alpha_per_group"],
        "reserved_test_generated":False,"new_pso_evaluations":0,
        "note":"Training is small; physical scoring can dominate. Run pilot before the complete experiment."}


def pilot(out,s,plan):
    out=Path(out)
    if (out/"timing_pilot.json").exists():return read(out/"timing_pilot.json")
    configure_runtime(7,1,"cpu")
    jobs=plan["jobs"]["train"];job=max([j for j in jobs if j["layout_id"]==min(a["layout_id"] for a in jobs)],key=lambda j:j["n_public"])
    part=[r for r in s["train"] if (r["layout_id"],r["n_public"])==(job["layout_id"],job["n_public"])]
    ref=next(v for v in s["source_views"].values() if v["kind"]=="gnn" and v["architecture_name"]=="small")
    model,_=load_model(s["stage"]/ref["path"])
    predictions=predict(model,[r["graph"] for r in part[:2]],s["stage_plan"]["normalization"])
    c,layout,bank,meta=context(out,plan,job,"timing")
    bounds=read(out/"source_physics_plan.json")["base_config"]["bounds"]
    times=[]
    for row,y in zip(part[:2],predictions):
        for delta in (0.,plan["settings"]["deltas"][-1]):
            fixed=baseline.lower_bob(y,delta,bounds);started=time.perf_counter()
            raw_assessment(dict(c,total_dbm=row["power_dbm"]),layout,bank,np.asarray(row["order"],int),fixed,
                           plan["pool_size"],plan["alpha_per_group"])
            times.append(time.perf_counter()-started)
    costs=workload(s)
    total=costs["validation_physics_rows"]+costs["train_physics_rows_upper_bound"]
    groups=sum(map(len,plan["jobs"].values()))
    per=statistics.median(times)
    new_fit_estimate=sum(read(s["stage"]/"cells/small/fits"/f"{kind}_s{seed}"/"fit_manifest.json")["training_seconds"]
                         for kind in ("gnn","mlp") for seed in plan["settings"]["seeds"])
    estimated_physics=(total*per+groups*meta["bank_generation_seconds"])/plan["workers"]
    result={"scope":plan["scope"],"physics_samples":plan["settings"]["physics_samples"],"timing_job":job["id"],
        "median_raw_assessment_seconds":per,"measured_seconds":times,"bank_generation_seconds":meta["bank_generation_seconds"],
        "physics_row_upper_bound":total,"estimated_physics_minutes_before_cache":estimated_physics/60,
        "source_small32_training_minutes_reference":new_fit_estimate/60,"workers":plan["workers"],
        "new_fit_minutes_note":"Reference only: source 32-small fits. Actual 8/16 fits have fewer train samples but fixed validation overhead.",
        "prediction":"Upper-count linear extrapolation; cache can reduce work, contention/temperature can increase wall time.",
        "fresh_timing_stream":True,"timing_used_for_model_selection":False,"reserved_test_generated":False}
    write(out/"timing_pilot.json",result,immutable=True);return result


def check(output,corpus=None,training=None,_source=None):
    out=Path(output).resolve();plan=read(out/"plan.json")
    if plan["code_sha256"]!=code_hashes():raise ValueError("Supplement code changed")
    s=_source or source(corpus or plan["source_corpus"],training or plan["source_training"],plan["settings"])
    if digest(s["stage"]/"completion.json")!=plan["source_stage_completion_sha256"]:
        raise ValueError("Source stage changed")
    if not (out/"completion.json").exists():
        return {"complete":False,"scope":plan["scope"],"source_stage_valid":True,
            "fits_frozen":(out/"fits_frozen.json").exists(),"validation_complete":(out/"validation_audit_frozen.json").exists(),
            "train_complete":(out/"train_audit_frozen.json").exists(),"reserved_test_generated":False}
    completion=read(out/"completion.json");verify(out,completion)
    if completion["plan_sha256"]!=digest(out/"plan.json"):raise ValueError("Changed completion plan")
    refs=read_models(out);all_views=variants(refs,plan["settings"])
    for size in plan["settings"]["sizes"]:
        stored=read(out/"normalizers"/f"train_{size:04d}.json")
        ids=s["stage_plan"]["training_layout_ids"][:size]
        expected=fit_normalizer([r["graph"] for r in s["train"] if r["layout_id"] in ids and r["has_label"]])
        if stored["normalizer"]!=expected or stored["layout_ids"]!=ids:raise ValueError("Normalizer leakage")
    bounds=read(out/"source_physics_plan.json")["base_config"]["bounds"]
    for ref in refs.values():
        if ref["family"]=="baseline":
            expected=baseline.fit([r for r in s["train"] if r["has_label"]],bounds,ref["alpha"]);actual=read(ref["path"])
            for key in ("scalars","public"):
                for part in ("mean","scale","coef"):
                    if not np.allclose(expected[key][part],actual[key][part],rtol=1e-10,atol=1e-10):raise ValueError("Baseline is not train-only")
    total=0
    for phase in ("validation","train"):
        verify(out,read(out/f"{phase}_predictions_frozen.json"));verify(out,read(out/f"{phase}_audit_frozen.json"))
        look={r["case_id"]:r for r in s[phase]}
        expected_views=set(all_views) if phase=="validation" else set(read(out/"selection.json")["train_views"])
        for job in plan["jobs"][phase]:
            record=committed(out/f"{phase}_audit"/(job["id"]+".json"))
            expected_cases={r["case_id"] for r in s[phase] if (r["layout_id"],r["n_public"])==(job["layout_id"],job["n_public"])}
            expected_pairs={(v,c) for v in expected_views for c in expected_cases if phase!="train" or
                all_views[v]["family"]!="curve" or job["layout_id"] in all_views[v]["training_layout_ids"]}
            if {(r["view_id"],r["case_id"]) for r in record["rows"]}!=expected_pairs or len(record["rows"])!=len(expected_pairs):
                raise ValueError("Changed candidate/case Cartesian product")
            if record["pool_size"]!=plan["pool_size"] or record["alpha_per_group"]!=plan["alpha_per_group"]:
                raise ValueError("Train/validation certification differs")
            for r in record["rows"]:
                if r["layout_id"]!=job["layout_id"] or r["phase"]!=phase:raise ValueError("Unexpected layout/split")
                if r["returned"]:
                    expected=at_target(r,r["xi"])
                    if r["passed"]!=expected["passed"] or r["failed_checks"]!=expected["failed_checks"]:
                        raise ValueError("Stored decision differs from its physical checks")
                    if r["constants"]["pool_size"]!=plan["pool_size"] or r["constants"]["alpha_family"]!=plan["alpha_per_group"]:
                        raise ValueError("Changed statistical constants")
                    errors=allocation_errors(r["allocation"],look[r["case_id"]]["allocation"] if look[r["case_id"]]["has_label"] else None,
                                             s["stage_plan"]["config"]["training"]["loss_weights"])
                    if errors!= {k:r[k] for k in errors}:raise ValueError("Stored allocation error differs")
                elif r["family"]!="teacher" or r["passed"]:raise ValueError("Unexpected no-return")
            total+=len(record["rows"])
    recomputed=select(out,plan,save=False)
    if recomputed!=read(out/"selection.json"):raise ValueError("Selection not validation-only")
    return {"complete":True,"scope":plan["scope"],"source_stage_unchanged":True,"new_pso_evaluations":0,
        "new_fits":2*2*len(plan["settings"]["seeds"]),"curve_sizes":plan["settings"]["sizes"],
        "verified_files":len(completion["sha256"]),"physical_rows":total,"reserved_test_generated":False,
        "note":"Supplement complete; original 7g seal and final test are not changed"}


def execute(action,corpus,training,output,config,workers=2,resume=False):
    out=Path(output).resolve();out.parent.mkdir(parents=True,exist_ok=True)
    with FileLock(str(out)+".lock",timeout=0):
        out,s,plan=initialize(corpus,training,out,config,workers,resume)
        if (out/"completion.json").exists():return check(out)
        original_failures(out,s)
        if action=="analyze":return read(out/"source_failure_summary.json")
        if action=="pilot":return pilot(out,s,plan)
        prepare_models(out,s,plan)
        if action=="fit":return {"fits_complete":True,"models":len(read_models(out)),"reserved_test_generated":False}
        freeze_predictions(out,s,plan,"validation");audit(out,plan,"validation")
        selection=select(out,plan)
        freeze_predictions(out,s,plan,"train",selection["train_views"]);audit(out,plan,"train")
        result=summarize(out,plan)
        # Verify source again before declaring completion; source folders are read-only throughout.
        checked_source=source(s["corpus"],s["training"],plan["settings"])
        paths=[p for p in sorted(out.rglob("*")) if p.is_file() and p.name!="completion.json"
               and not p.name.endswith((".tmp",".lock")) and "figures" not in p.relative_to(out).parts]
        freeze(out,"completion.json",paths,plan_sha256=digest(out/"plan.json"),reserved_test_generated=False)
        return check(out,_source=checked_source)
