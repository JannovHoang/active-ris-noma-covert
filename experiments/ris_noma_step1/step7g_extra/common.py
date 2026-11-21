from pathlib import Path
import csv
import hashlib
import json
import platform
import statistics
import time

import numpy as np
import scipy
import torch
from ris_noma import curve_learning, curve_data
from ris_noma.learning_train import configure_runtime

ROOT = Path(__file__).resolve().parent.parent
FORMAT = "supplement_7g32_v1"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean(value):
    if isinstance(value, np.ndarray): return clean(value.tolist())
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, dict): return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [clean(v) for v in value]
    return value


def write(path, value, immutable=False):
    path = Path(path)
    if immutable and path.exists():
        if read(path) != clean(value): raise ValueError(f"Frozen record changed: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def lines(path):
    with Path(path).open(encoding="utf-8-sig") as f:
        return [json.loads(s) for s in f if s.strip()]


def table(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        w.writerows({k: json.dumps(clean(v), ensure_ascii=False) if isinstance(v,(list,dict)) else v
                    for k,v in r.items()} for r in rows)


def commit(path, value):
    write(path, value)
    write(str(path)+".receipt.json", {"sha256": digest(path)})


def committed(path):
    path=Path(path); receipt=Path(str(path)+".receipt.json")
    if not path.exists() and not receipt.exists(): return None
    if not receipt.exists(): return None  # interrupted before receipt; safe to recompute
    if not path.exists() or read(receipt)["sha256"] != digest(path):
        raise ValueError(f"Committed file was changed: {path}")
    return read(path)


def freeze(folder, name, paths, **metadata):
    folder=Path(folder)
    record={**metadata,"sha256":{str(Path(p).relative_to(folder)):digest(p) for p in paths}}
    write(folder/name, record, immutable=True)
    return record


def verify(folder, record):
    folder=Path(folder).resolve()
    for name, sha in record["sha256"].items():
        p=(folder/name).resolve()
        if not p.is_relative_to(folder) or not p.is_file() or digest(p)!=sha:
            raise ValueError(f"Manifest mismatch: {name}")


def code_hashes():
    paths=[ROOT/"run_step7g_extra.py", *sorted((ROOT/"step7g_extra").rglob("*.py"))]
    return {str(p.relative_to(ROOT)):digest(p) for p in paths if "__pycache__" not in p.parts}


def settings(config):
    c=read(config) if not isinstance(config,dict) else dict(config)
    expected={"scope","sizes","seeds","ridge_alphas","deltas","physics_samples","alpha",
              "bootstrap_resamples","bootstrap_seed","evaluation_train_layouts","evaluation_validation_layouts",
              "evaluation_public_counts","epochs_override","snapshots_override"}
    if set(c)!=expected or c["scope"] not in ("development_diagnostic","smoke"):
        raise ValueError("Unknown supplement configuration")
    if c["sizes"]!=[8,16,32] or not c["seeds"] or len(set(c["seeds"]))!=len(c["seeds"]):
        raise ValueError("Use the nested 8/16/32 train prefixes")
    for key in ("ridge_alphas","deltas"):
        if not c[key] or len(set(c[key]))!=len(c[key]) or any(not np.isfinite(v) for v in c[key]):
            raise ValueError("Use finite distinct baseline/delta candidates")
    if any(a<=0 for a in c["ridge_alphas"]) or c["deltas"][0]!=0 or any(d<0 or d>.02 for d in c["deltas"]):
        raise ValueError("Invalid ridge or small-delta values")
    if type(c["physics_samples"]) is not int or c["physics_samples"]<1000 or not 0<c["alpha"]<1:
        raise ValueError("Invalid physics sample count/alpha")
    if c["scope"]=="development_diagnostic" and (c["seeds"]!=[7,17,27] or c["physics_samples"]!=100000
        or any(c[k] is not None for k in ("evaluation_train_layouts","evaluation_validation_layouts",
             "evaluation_public_counts","epochs_override","snapshots_override"))):
        raise ValueError("Main study must retain all layouts/K, original epochs and 100,000 samples")
    return c


def source(corpus, training, config, thorough=True):
    corpus,training=Path(corpus).resolve(),Path(training).resolve()
    c=settings(config)
    if thorough:
        curve_data.check_corpus(corpus,32)
        curve_learning.check_stage(corpus,training,32)
    stage=training/"stages/train_0032"
    p=read(stage/"plan.json"); master=read(corpus/"plan.json")
    train=lines(stage/"inputs/train.jsonl"); val=lines(stage/"inputs/validation.jsonl")
    if len(p["training_layout_ids"])!=32 or master["config"]["training_public_counts"]!=[2,3,4,5]:
        raise ValueError("This supplement requires the main Windows 7g/32 source")
    train_ids=set(p["training_layout_ids"]); val_ids={r["layout_id"] for r in val}
    reserved=set(master["config"]["reserved_test_layout_ids"])
    if train_ids&val_ids or (train_ids|val_ids)&reserved or any(r["split"] not in ("train","validation") for r in train+val):
        raise ValueError("Train/validation/reserved-test overlap")
    if any(r["n_public"] not in [2,3,4,5] for r in train+val): raise ValueError("No unseen K is generated here")
    original=read(stage/"selection_frozen.json")
    views={k:v for k,v in original["views"].items() if v["selection_rule"]=="physical" and v["seed"] in c["seeds"]}
    if len(views)!=4*len(c["seeds"]): raise ValueError("Missing frozen source views")
    jobs=curve_data.stage_jobs(master,32)
    evaluated={}
    for split,rows,limit in (("train",train,c["evaluation_train_layouts"]),("validation",val,c["evaluation_validation_layouts"])):
        allowed=sorted({r["layout_id"] for r in rows})
        if limit is not None: allowed=allowed[:limit]
        evaluated[split]=[j for j in jobs if j["split"]==split and j["layout_id"] in allowed
            and (c["evaluation_public_counts"] is None or j["n_public"] in c["evaluation_public_counts"])]
    snapshots=p["config"]["training"]["snapshot_epochs"] if c["snapshots_override"] is None else c["snapshots_override"]
    curve_views=len(c["sizes"])*2*len(c["seeds"])*(len(snapshots)+1)
    # Reused 32 has original 6 candidates even in the small smoke configuration.
    if c["scope"]=="smoke": curve_views=2*2*len(c["seeds"])*(len(snapshots)+1)+2*len(c["seeds"])*6
    family_views=len(views)*len(c["deltas"])+len(c["ridge_alphas"])*len(c["deltas"])+curve_views+1
    job_count=sum(map(len,evaluated.values())); max_cases=max(len(j["cases"]) for js in evaluated.values() for j in js)
    result={"corpus":corpus,"training":training,"stage":stage,"settings":c,"stage_plan":p,
        "master":master,"train":train,"validation":val,"source_views":views,"jobs":evaluated,
        "family_views":family_views,"pool_size":family_views*max_cases,"alpha_per_group":c["alpha"]/(2*job_count)}
    return result


def initialize(corpus,training,output,config,workers,resume):
    configure_runtime(0,1,"cpu")
    s=source(corpus,training,config)
    out=Path(output).resolve()
    for base in (s["corpus"],s["training"]):
        if out==base or out.is_relative_to(base) or base.is_relative_to(out):
            raise ValueError("Supplement output must be separate and non-nested with both sources")
    if type(workers) is not int or not 1<=workers<=4: raise ValueError("Use 1..4 physics workers")
    plan={"format":FORMAT,"scope":s["settings"]["scope"],"settings":s["settings"],"workers":workers,
        "source_corpus":str(s["corpus"]),"source_training":str(s["training"]),
        "source_corpus_plan_sha256":digest(s["corpus"]/"plan.json"),
        "source_stage_completion_sha256":digest(s["stage"]/"completion.json"),
        "source_selection_sha256":digest(s["stage"]/"selection_frozen.json"),
        "source_input_sha256":{role:digest(s["stage"]/f"inputs/{role}.jsonl") for role in ("train","validation")},
        "code_sha256":code_hashes(),"family_views":s["family_views"],"pool_size":s["pool_size"],
        "alpha_per_group":s["alpha_per_group"],"jobs":s["jobs"],
        "streams":{"validation":9811,"train":9812,"timing":9813},
        "versions":{"system":platform.system(),"python":platform.python_version(),"numpy":np.__version__,
                    "scipy":scipy.__version__,"torch":str(torch.__version__)},
        "reserved_test_generated":False,"pso_evaluations":0,
        "selection_policy":"validation passes, gated rate, lower weighted loss, earlier epoch, stable ID",
        "certification":"same fixed pool-size/alpha for train and validation; fresh i.i.d. fading streams",
        "note":"Post-hoc development diagnostic; preserves original stage 32; not final test or formal no-overfitting proof"}
    if out.exists() and any(out.iterdir()) and not resume:
        raise ValueError("Supplement output is nonempty; use --resume with the same plan or a new output")
    out.mkdir(parents=True,exist_ok=True);write(out/"plan.json",plan,immutable=True)
    write(out/"source_stage_plan.json",s["stage_plan"],immutable=True)
    write(out/"source_physics_plan.json",read(s["stage"]/"source_physics_plan.json"),immutable=True)
    write(out/"source_dataset_plan.json",read(s["stage"]/"source_dataset_plan.json"),immutable=True)
    return out,s,plan


def allocation_errors(y,target,weights):
    if target is None: return {"case_mse":None,"weighted_loss":None,"mb_error":None,"mb_squared_error":None,
                              "public_squared_error":None,"s_squared_error":None}
    error=np.asarray(y)-np.asarray(target); sq=error**2
    return {"case_mse":float(sq.mean()),"weighted_loss":float((weights["mb"]*sq[0]+weights["public_total"]*sq[1:-1].mean()+weights["s"]*sq[-1])/sum(weights.values())),
        "mb_error":float(error[0]),"mb_squared_error":float(sq[0]),"public_squared_error":float(sq[1:-1].mean()),"s_squared_error":float(sq[-1])}


def metrics(rows):
    if not rows: raise ValueError("Empty evaluation denominator")
    labeled=[r for r in rows if r.get("case_mse") is not None]
    eligible=[r for r in rows if not r.get("invalid_layout_assumption",False)]
    return {"cases":len(rows),"passed":sum(r["passed"] for r in rows),
        "pass_fraction":statistics.mean(r["passed"] for r in rows),
        "gated_rate":statistics.mean(r["gated_bob_rate"] for r in rows),
        "raw_rate":statistics.mean(r["bob_rate"] for r in rows),
        "labeled_cases":len(labeled),"case_mse":statistics.mean(r["case_mse"] for r in labeled) if labeled else None,
        "weighted_loss":statistics.mean(r["weighted_loss"] for r in labeled) if labeled else None,
        "eligible_cases":len(eligible),"eligible_pass_fraction":statistics.mean(r["passed"] for r in eligible) if eligible else None,
        "point_pass_fraction":statistics.mean(r.get("point_passed",r["passed"]) for r in rows)}
