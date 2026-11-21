from pathlib import Path
from types import SimpleNamespace
import time
import numpy as np
from . import baseline
from .common import read,write,digest,commit,committed,freeze,verify,allocation_errors
from .fits import read_models
from ris_noma.curve_models import load_model
from ris_noma.multik_models import predict
from ris_noma.multik_core import physical_allocation
from ris_noma.multik_data_core import at_target
from ris_noma.stable_assessment import raw_assessment
from ris_noma.continuous_data import geometry_from
from ris_noma.channels import build_bank
from ris_noma.curve_data import execute_jobs


def variants(refs,settings):
    result={}
    for key,ref in refs.items():
        deltas=settings["deltas"] if ref["family"] in ("baseline","frozen32") else [0.]
        for delta in deltas:
            identity=key+f"::d{delta:.6f}"
            result[identity]={**ref,"model_id":key,"delta":delta}
    return result


def freeze_predictions(out,s,plan,phase,views=None):
    out=Path(out);refs=read_models(out)
    if phase not in ("train","validation"):raise ValueError("No final test in this program")
    all_views=variants(refs,plan["settings"])
    if views is not None:all_views={k:all_views[k] for k in views}
    if phase=="train" and not (out/"selection.json").exists():raise ValueError("Select on validation before train audit")
    jobs=plan["jobs"][phase];ids={j["layout_id"] for j in jobs};ks={j["n_public"] for j in jobs}
    rows=[r for r in s[phase] if r["layout_id"] in ids and r["n_public"] in ks]
    lookup={r["case_id"]:r for r in rows};values={};timings={}
    bounds=read(out/"source_physics_plan.json")["base_config"]["bounds"]
    weights=s["stage_plan"]["config"]["training"]["loss_weights"]
    for key in sorted({v["model_id"] for v in all_views.values()}):
        ref=refs[key];start=time.perf_counter()
        if ref["family"]=="teacher":pred=[r["allocation"] if r["has_label"] else None for r in rows]
        elif ref["family"]=="baseline":pred=baseline.predict(read(ref["path"]),rows)
        else:
            model,saved=load_model(ref["path"])
            if saved["normalization"]!=ref["normalization"]:raise ValueError("Model normalizer changed")
            start=time.perf_counter()
            pred=predict(model,[r["graph"] for r in rows],ref["normalization"])
        timings[key]={("forward_pack_seconds" if ref["family"] not in ("teacher","baseline") else "prediction_seconds"):time.perf_counter()-start,"cases":len(rows)}
        values[key]=dict(zip(lookup,pred))
    write(out/f"{phase}_prediction_timing.json",timings)
    paths=[]
    for job in jobs:
        path=out/f"{phase}_predictions"/(job["id"]+".json")
        prior=committed(path)
        if prior is None:
            records=[]
            part=[r for r in rows if r["layout_id"]==job["layout_id"] and r["n_public"]==job["n_public"]]
            if len(part)!=len(job["cases"]):raise ValueError("Case denominator changed")
            for view,v in all_views.items():
                if phase=="train" and v["family"]=="curve" and job["layout_id"] not in v["training_layout_ids"]:continue
                for r in part:
                    y=values[v["model_id"]][r["case_id"]]
                    if y is not None:y=baseline.lower_bob(y,v["delta"],bounds)
                    records.append({k:r[k] for k in ("case_id","layout_id","n_public","power_dbm","xi","order")}
                        | {"phase":phase,"split":phase,"view_id":view,"model_id":v["model_id"],"family":v["family"],
                           "architecture":v["architecture"],"kind":v["kind"],"seed":v["seed"],"size":v["size"],
                           "epoch":v["epoch"],"delta":v["delta"],"allocation":None if y is None else y.tolist(),
                           "returned":y is not None,"has_label":r["has_label"],
                           **allocation_errors(y,r["allocation"] if r["has_label"] and y is not None else None,weights)})
            commit(path,{"job":job,"phase":phase,"rows":records,"plan_sha256":digest(out/"plan.json"),
                         "fits_sha256":digest(out/"fits_frozen.json"),
                         "selection_sha256":digest(out/"selection.json") if phase=="train" else None})
        elif prior["plan_sha256"]!=digest(out/"plan.json") or prior["fits_sha256"]!=digest(out/"fits_frozen.json"):
            raise ValueError("Prediction receipt belongs to another plan")
        paths.append(path);paths.append(Path(str(path)+".receipt.json"))
    freeze(out,f"{phase}_predictions_frozen.json",paths,plan_sha256=digest(out/"plan.json"),
           views=sorted(all_views),selection_sha256=digest(out/"selection.json") if phase=="train" else None)


def context(out,plan,job,phase):
    if phase not in plan["streams"] or phase=="timing" and job["split"]!="train":raise ValueError("Unknown bank role")
    allowed=plan["jobs"]["train" if phase=="timing" else phase]
    if job not in allowed:raise ValueError("Layout outside declared diagnostic train/validation")
    stage=read(Path(out)/"source_stage_plan.json")
    c,layout,meta=geometry_from(stage["geometry_spec"],read(Path(out)/"source_physics_plan.json"),
         read(Path(out)/"source_dataset_plan.json"),job,plan["settings"]["physics_samples"])
    entropy=[c["root_seed"],job["layout_id"],job["n_public"],plan["streams"][phase]]
    start=time.perf_counter();full=build_bank(c,layout,np.random.default_rng(np.random.SeedSequence(entropy)))
    bank=SimpleNamespace(**{n:getattr(full,n) for n in ("h_ar","h_tilde","g_scalar_tilde","signal_gain_tilde","noise_gain_tilde","mean_h_ar_power")})
    meta.update(bank_entropy=entropy,bank_samples=plan["settings"]["physics_samples"],bank_generation_seconds=time.perf_counter()-start)
    return c,layout,bank,meta


def audit_task(args):
    out,phase,job=args;out=Path(out);plan=read(out/"plan.json")
    if phase not in ("train","validation"):raise ValueError("No final test audit")
    anchor=out/f"{phase}_predictions_frozen.json"
    if not anchor.exists() or not (out/"fits_frozen.json").exists():raise ValueError("Freeze fits/predictions before bank generation")
    if phase=="train" and not (out/"selection.json").exists():raise ValueError("Validation selection must precede train audit")
    path=out/f"{phase}_audit"/(job["id"]+".json")
    old=committed(path)
    if old is not None:
        if old["predictions_sha256"]!=digest(anchor):raise ValueError("Audit anchor changed")
        return job["id"]
    prediction=committed(out/f"{phase}_predictions"/(job["id"]+".json"))
    if prediction is None:raise ValueError("Missing prediction receipt")
    c,layout,bank,meta=context(out,plan,job,phase);cache={};rows=[];start=time.perf_counter()
    for row in prediction["rows"]:
        if not row["returned"]:
            gains=bank.signal_gain_tilde.mean(axis=0)
            rows.append({**row,"passed":False,"point_passed":False,"invalid_layout_assumption":bool(gains[0]<=gains[1:-1].max()),
                         "bob_rate":0.,"gated_bob_rate":0.,"failed_checks":["teacher_label_missing"]});continue
        key=(row["power_dbm"],tuple(row["allocation"]))
        if key not in cache:
            cache[key]=raw_assessment(dict(c,total_dbm=row["power_dbm"]),layout,bank,np.asarray(row["order"],int),
                                     row["allocation"],plan["pool_size"],plan["alpha_per_group"])
        raw=cache[key];decision=at_target(raw,row["xi"])
        point=raw["dep_gamma"]>=row["xi"] and all(raw["point_checks_without_dep"].values())
        qos=np.asarray(c["requirements"]["qos_public"])[row["order"]]
        rows.append({**row,**raw,**decision,"point_passed":bool(point),
            "invalid_layout_assumption":not raw["point_checks_without_dep"]["layout_bob_mean_gain_rule"],
            "dep_point_margin":raw["dep_gamma"]-row["xi"],"dep_empirical_margin":raw["dep_empirical"]-row["xi"],
            "dep_lower_margin":raw["dep_lower"]-row["xi"],
            "public_mean_margins":(np.asarray(raw["public_mean_rate_sic"])-qos).tolist(),
            "public_lower_margins":(np.asarray(raw["public_rate_lower_sic"])-qos).tolist()})
    commit(path,{"job":job,"phase":phase,"rows":rows,"metadata":meta,"scoring_seconds":time.perf_counter()-start,
                 "unique_evaluations":len(cache),"predictions_sha256":digest(anchor),"plan_sha256":digest(out/"plan.json"),
                 "pool_size":plan["pool_size"],"alpha_per_group":plan["alpha_per_group"]})
    return job["id"]


def audit(out,plan,phase):
    out=Path(out);anchor=read(out/f"{phase}_predictions_frozen.json");verify(out,anchor)
    pending=[j for j in plan["jobs"][phase] if committed(out/f"{phase}_audit"/(j["id"]+".json")) is None]
    execute_jobs(audit_task,[(str(out),phase,j) for j in pending],plan["workers"],f"supplement {phase}")
    paths=[]
    for j in plan["jobs"][phase]:
        p=out/f"{phase}_audit"/(j["id"]+".json");record=committed(p)
        if record["predictions_sha256"]!=digest(out/f"{phase}_predictions_frozen.json"):raise ValueError("Audit manifest mismatch")
        paths += [p,Path(str(p)+".receipt.json")]
    freeze(out,f"{phase}_audit_frozen.json",paths,plan_sha256=digest(out/"plan.json"))


def phase_rows(out,plan,phase):
    return [r for j in plan["jobs"][phase] for r in committed(Path(out)/f"{phase}_audit"/(j["id"]+".json"))["rows"]]
