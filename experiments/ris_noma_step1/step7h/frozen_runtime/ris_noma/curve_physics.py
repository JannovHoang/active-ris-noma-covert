"""Frozen validation/train-probe replay. No final-test generator exists in 7g."""
from pathlib import Path
from types import SimpleNamespace
import numpy as np

from .channels import build_bank
from .continuous_data import geometry_from
from .learning_data import read_json, sha256
from .multik_data_core import at_target
from .multik_dataset import write_json, read_committed, receipt_path
from .stable_assessment import raw_assessment

STREAMS = {"validation": 711, "train_probe": 712}


def record_path(folder, phase, what, job):
    return Path(folder)/f"{phase}_{what}"/(job["id"]+".json")


def commit(path, value):
    write_json(path, value)
    write_json(receipt_path(path), {"sha256": sha256(path)})


def read_record(folder, phase, what, job):
    doc = read_committed(record_path(folder, phase, what, job))
    if doc is not None and doc.get("plan_sha256") != sha256(Path(folder)/"plan.json"):
        raise ValueError("7g physical record belongs to another stage")
    return doc


def confidence(plan, phase, job):
    if phase not in STREAMS:
        raise ValueError("7g has no test bank role")
    maximum = plan["max_validation_views"] if phase == "validation" else plan["max_train_probe_views"]
    return maximum*len(job["cases"]), plan["config"]["physics"]["alpha"]/(2*len(plan[phase+"_jobs"]))


def context(plan, folder, job, phase):
    if phase not in STREAMS or plan["physics_streams"] != STREAMS:
        raise ValueError("7g supports validation/train probe only, never final test")
    if job not in plan[phase+"_jobs"]:
        raise ValueError("Geometry is outside the declared physical split")
    samples = plan["config"]["physics"][phase+"_samples"]
    folder = Path(folder)
    c, layout, meta = geometry_from(plan["geometry_spec"],
        read_json(folder/"source_physics_plan.json"), read_json(folder/"source_dataset_plan.json"), job, samples)
    entropy = [c["root_seed"], job["layout_id"], job["n_public"], STREAMS[phase]]
    full = build_bank(c, layout, np.random.default_rng(np.random.SeedSequence(entropy)))
    names = ("h_ar", "h_tilde", "g_scalar_tilde", "signal_gain_tilde", "noise_gain_tilde", "mean_h_ar_power")
    bank = SimpleNamespace(**{name: getattr(full,name) for name in names})
    meta.update(bank_role=phase, bank_entropy=entropy, bank_samples=samples)
    return c, layout, bank, meta


def audit_task(arguments):
    folder, phase, job = arguments
    folder = Path(folder)
    if phase not in STREAMS:
        raise ValueError("No final-test audit in 7g")
    # Barriers must precede channel generation, including on worker processes.
    if not (folder/"fits_frozen.json").is_file():
        raise ValueError("Freeze all stage fits before physical validation")
    if phase == "train_probe" and not (folder/"selection_frozen.json").is_file():
        raise ValueError("Select checkpoints using validation before train probes")
    anchor = folder/(phase+"_predictions_frozen.json")
    if not anchor.is_file():
        raise ValueError("Freeze every prediction before its physical bank")
    existing = read_record(folder, phase, "audit", job)
    if existing is not None:
        return job["id"]
    plan = read_json(folder/"plan.json")
    proposals = read_record(folder, phase, "predictions", job)
    expected = plan["stage_validation_views"] if phase == "validation" else plan["stage_train_probe_views"]
    if proposals is None or len(proposals["rows"]) != expected*len(job["cases"]):
        raise ValueError("Missing candidate/case Cartesian product")
    c, layout, bank, meta = context(plan,folder,job,phase)
    family, alpha = confidence(plan,phase,job)
    cache = {}
    rows = []
    for proposal in proposals["rows"]:
        if not proposal["returned"]:
            if proposal["kind"] != "pso_label_reference" or proposal["allocation"] is not None:
                raise ValueError("Only a missing teacher label may be no-return")
            rows.append(dict(proposal, passed=False, bob_rate=0., gated_bob_rate=0.,
                             failed_checks=["teacher_label_missing"]))
            continue
        key = (proposal["power_dbm"], tuple(proposal["allocation"]))
        if key not in cache:
            cache[key] = raw_assessment(dict(c,total_dbm=proposal["power_dbm"]),layout,bank,
                np.asarray(proposal["order"],int),proposal["allocation"],family,alpha)
        raw = cache[key]
        rows.append(dict(proposal,**raw,**at_target(raw,proposal["xi"])))
    commit(record_path(folder,phase,"audit",job), {
        "job":job,"phase":phase,"rows":rows,"metadata":meta,"family_size":family,
        "alpha_per_group":alpha,"unique_evaluations":len(cache),
        "predictions_frozen_sha256":sha256(anchor),"plan_sha256":sha256(folder/"plan.json"),
        "test_used_for_selection":False})
    return job["id"]
