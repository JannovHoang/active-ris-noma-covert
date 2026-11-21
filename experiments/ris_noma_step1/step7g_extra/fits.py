"""Reuse all small 32-layout checkpoints; train only the 8/16 prefixes."""
from pathlib import Path
import time
from . import baseline
from .common import read, write, digest, freeze, verify
from ris_noma.curve_fit import fit, read_fit
from ris_noma.multik_graph import fit_normalizer


def prepare_models(out,s,plan):
    out=Path(out);refs={};paths=[]
    weights=s["stage_plan"]["config"]["training"]["loss_weights"]
    bounds=read(out/"source_physics_plan.json")["base_config"]["bounds"]
    train=[r for r in s["train"] if r["has_label"]];val=[r for r in s["validation"] if r["has_label"]]
    for view,v in s["source_views"].items():
        name="frozen32::"+view
        path=s["stage"]/v["path"]
        refs[name]={"family":"frozen32","size":32,"architecture":v["architecture_name"],"kind":v["kind"],
            "seed":v["seed"],"epoch":v["epoch"],"path":str(path),"sha256":v["sha256"],
            "normalization":s["stage_plan"]["normalization"],"training_layout_ids":s["stage_plan"]["training_layout_ids"],
            "parameters":v["parameters"],"reused":True,"training_seconds":None}
    for size in plan["settings"]["sizes"]:
        ids=s["stage_plan"]["training_layout_ids"][:size]
        rows=[r for r in train if r["layout_id"] in ids]
        if {r["layout_id"] for r in rows}!=set(ids):raise ValueError("A train prefix has unlabeled-only layouts")
        normalizer=fit_normalizer([r["graph"] for r in rows])
        if size==32:
            if normalizer!=s["stage_plan"]["normalization"]:raise ValueError("Reused normalizer differs")
            cell=s["stage"]/"cells/small"
            cp=read(cell/"plan.json");active=dict(cp,plan_sha256=digest(cell/"plan.json"))
        else:
            cell=out/"curve_fits"/f"train_{size:04d}"
            config={**s["stage_plan"]["config"]["training"],**s["stage_plan"]["config"]["architectures"]["small"]}
            config.update(validation_physics_samples=plan["settings"]["physics_samples"],
                          test_physics_samples=plan["settings"]["physics_samples"],alpha=plan["settings"]["alpha"])
            if plan["settings"]["epochs_override"] is not None:
                config["epochs"]=plan["settings"]["epochs_override"]
                config["snapshot_epochs"]=plan["settings"]["snapshots_override"]
            cp={"step":"7g_supplement","config":config,"architecture_name":"small","normalization":normalizer,
                "training_layout_ids":ids,"training_public_counts":[2,3,4,5],
                "stage_completion_sha256":plan["source_stage_completion_sha256"],
                "corpus_plan_sha256":plan["source_corpus_plan_sha256"],"supplement_plan_sha256":digest(out/"plan.json")}
            write(cell/"plan.json",cp,immutable=True);active=dict(cp,plan_sha256=digest(cell/"plan.json"));paths.append(cell/"plan.json")
        for kind in ("gnn","mlp"):
            for seed in plan["settings"]["seeds"]:
                if size==32:
                    record=read_fit(cell,kind,seed,active)
                else:
                    print(f"Supplement train {size}: {kind} seed={seed}",flush=True)
                    record=fit(cell,kind,seed,active,bounds,rows,val)
                    paths.extend(cell/"fits"/f"{kind}_s{seed}"/n for n in record["sha256"])
                    paths.append(cell/"fits"/f"{kind}_s{seed}"/"fit_manifest.json")
                for key,v in record["candidates"].items():
                    name=f"curve::{size}::{key}"
                    refs[name]={"family":"curve","size":size,"architecture":"small","kind":kind,"seed":seed,
                        "epoch":v["epoch"],"path":str(cell/v["path"]),"sha256":v["sha256"],"normalization":normalizer,
                        "training_layout_ids":ids,"parameters":v["parameters"],"reused":size==32,
                        "training_seconds":record["training_seconds"],"training_time_origin":
                          read(s["training"]/"plan.json")["versions"]["system"] if size==32 else plan["versions"]["system"]}
        write(out/"normalizers"/f"train_{size:04d}.json",{"fit_role":"train","layout_ids":ids,
            "labeled_cases":len(rows),"normalizer":normalizer},immutable=True)
        paths.append(out/"normalizers"/f"train_{size:04d}.json")
    for alpha in plan["settings"]["ridge_alphas"]:
        name=f"baseline::a{alpha:g}";path=out/"baselines"/f"ridge_{alpha:g}.json"
        started=time.perf_counter();record=baseline.fit(train,bounds,alpha)
        record["fit_seconds"]=time.perf_counter()-started
        if path.exists():record=read(path)
        write(path,record,immutable=True);paths.append(path)
        refs[name]={"family":"baseline","size":32,"architecture":"scalar_ridge","kind":"baseline","seed":None,
            "epoch":0,"path":str(path),"sha256":digest(path),"training_layout_ids":s["stage_plan"]["training_layout_ids"],
            "alpha":alpha,"parameters":len(record["scalars"]["coef"])*2+len(record["public"]["coef"]),
            "training_seconds":record["fit_seconds"],"reused":False}
    refs["teacher"]={"family":"teacher","kind":"teacher","architecture":"PSO_label","size":32,"seed":None,
                     "epoch":0,"parameters":0,"training_layout_ids":s["stage_plan"]["training_layout_ids"]}
    write(out/"models.json",refs,immutable=True);paths.append(out/"models.json")
    freeze(out,"fits_frozen.json",paths,plan_sha256=digest(out/"plan.json"))
    return refs


def read_models(out):
    out=Path(out);record=read(out/"fits_frozen.json");verify(out,record)
    if record["plan_sha256"]!=digest(out/"plan.json"):raise ValueError("Changed fit plan")
    refs=read(out/"models.json")
    for r in refs.values():
        if r["family"]!="teacher" and digest(r["path"])!=r["sha256"]:raise ValueError("Changed model or baseline")
    return refs
