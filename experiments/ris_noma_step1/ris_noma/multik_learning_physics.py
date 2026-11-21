"""Independent 7c checkpoint validation/final audit; no training imports."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .channels import build_bank
from .learning_data import read_json, sha256
from .multik_data_core import geometry, raw_assessment, at_target
from .multik_dataset import group_name, read_committed, receipt_path, write_json

STREAMS = {"validation": 92, "test": 93}


def context(plan, dataset_plan, source, job, phase):
    if phase not in STREAMS or plan["streams"] != STREAMS:
        raise ValueError("Unknown 7c physical bank role")
    samples = plan["config"]["validation_physics_samples" if phase=="validation" else "test_physics_samples"]
    c,layout = geometry(dataset_plan,source,job,samples)
    rng = np.random.default_rng(np.random.SeedSequence([c["root_seed"],job["layout_id"],job["n_public"],STREAMS[phase]]))
    full = build_bank(c,layout,rng)
    names=("h_ar","h_tilde","g_scalar_tilde","signal_gain_tilde","noise_gain_tilde","mean_h_ar_power")
    bank=SimpleNamespace(**{name:getattr(full,name) for name in names})
    return c,layout,bank


def audit_task(arguments):
    folder,phase,job=arguments
    output=Path(folder)
    path=output/f"{phase}_audit"/group_name(job)
    existing=read_committed(path)
    if existing is not None:
        return path.name
    plan=read_json(output/"plan.json")
    if phase=="validation" and job["split"]!="validation" or phase=="test" and job["split"]!="test":
        raise ValueError("A physical phase received the wrong split")
    if phase=="test" and not (output/"selection_frozen.json").is_file():
        raise ValueError("Freeze all checkpoint selections before the final audit")
    dataset_plan=read_json(output/"source_dataset_plan.json")
    source=read_json(output/"source_physics_plan.json")
    proposals=read_committed(output/f"{phase}_predictions"/group_name(job))
    if proposals is None:
        raise ValueError("Predictions are not committed")
    frozen_hash=sha256(output/f"{phase}_predictions_frozen.json")
    physical,layout,bank=context(plan,dataset_plan,source,job,phase)
    rows,cache=[],{}
    group_count=len(plan[f"{phase}_jobs"])
    pool=plan["max_validation_candidates"] if phase=="validation" else plan["max_test_views"]
    family=pool*len(plan["power_dbm"])*len(plan["xi"])
    alpha=plan["config"]["alpha"]/(2*group_count)
    for proposal in proposals["rows"]:
        key=(proposal["power_dbm"],tuple(proposal["allocation"]))
        if key not in cache:
            cache[key]=raw_assessment(dict(physical,total_dbm=proposal["power_dbm"]),layout,bank,
                          np.asarray(proposal["order"],int),proposal["allocation"],family,alpha)
        raw=cache[key]
        rows.append({**proposal,**raw,**at_target(raw,proposal["xi"])})
    result={**job,"phase":phase,"rows":rows,"unique_physical_evaluations":len(cache),
            "predictions_frozen_sha256":frozen_hash,"bank_samples":physical["n_fading"],
            "family_size":family,"alpha_per_group":alpha,"test_used_for_selection":False}
    write_json(path,result)
    write_json(receipt_path(path),{"sha256":sha256(path)})
    return path.name
