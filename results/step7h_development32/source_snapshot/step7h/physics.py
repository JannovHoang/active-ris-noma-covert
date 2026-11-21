"""The existing physical equations, fresh role-specific banks, no audit-driven repair."""
from pathlib import Path
from types import SimpleNamespace
from copy import deepcopy
import time
import numpy as np

from .common import read, digest, committed
from ris_noma.channels import build_bank, public_order
from ris_noma.continuous_data import geometry_from
from ris_noma.multik_core import physical_allocation

STREAMS = {"development_input": 10701, "development_audit": 10702, "development_pso": 10703,
           "cv_input": 10711, "cv_audit": 10712, "final_input": 10721, "final_audit": 10722,
           "final_pso": 10723, "timing_input": 10731, "timing_audit": 10732, "timing_pso": 10733}


def entropy(root_seed, job, role, *extra):
    if role not in STREAMS: raise ValueError("Unknown 7h random-bank role")
    values = [root_seed, job["layout_id"], job["n_public"], STREAMS[role], *extra]
    if any(type(v) is not int or v < 0 for v in values): raise ValueError("Invalid RNG stream fields")
    return values


def authorize_context(output, plan, job, role):
    output = Path(output)
    if job not in plan["jobs"]: raise ValueError("Geometry is outside the declared stage jobs")
    stage = plan["stage"]
    permitted = {"development": ("development_input", "development_audit"),
                 "final": ("final_input", "final_audit"), "timing": ("timing_input", "timing_audit"),
                 "cv": ("cv_input", "cv_audit")}
    if stage not in permitted or role not in permitted[stage]: raise ValueError("Bank role does not match stage")
    if stage == "final":
        if not (output / "protocol.json").exists() or digest(output / "protocol.json") != plan["protocol_sha256"]:
            raise ValueError("Seal the development protocol before generating final-test geometry or channels")
        opened = read(output / "test_opened.json")
        if opened["protocol_sha256"] != plan["protocol_sha256"]: raise ValueError("Final-test commitment changed")
    if role.endswith("audit"):
        anchor = output / "allocations_frozen.json"
        if not anchor.exists(): raise ValueError("Freeze all proposals before generating audit fading")
        record = read(anchor)
        if record["plan_sha256"] != digest(output / "plan.json"): raise ValueError("Audit allocation anchor belongs to another plan")
        name = f"search/{job['id']}.json" if stage != "cv" else f"predictions/{job['id']}.json"
        if record["sha256"].get(name) != digest(output / name) or committed(output / name) is None:
            raise ValueError("The committed proposals changed before audit")


def context(output, plan, job, role):
    output = Path(output)
    authorize_context(output, plan, job, role)
    samples = plan["settings"]["audit_samples" if role.endswith("audit") else "search_samples"]
    spec = read(output / "inputs/stage_plan.json")["geometry_spec"]
    c, layout, meta = geometry_from(spec, read(output / "inputs/physics.json"), read(output / "inputs/dataset.json"), job, samples)
    fields = entropy(c["root_seed"], job, role)
    started = time.perf_counter()
    full = build_bank(c, layout, np.random.default_rng(np.random.SeedSequence(fields)))
    bank = SimpleNamespace(**{name: getattr(full, name) for name in
        ("h_ar", "h_tilde", "g_scalar_tilde", "signal_gain_tilde", "noise_gain_tilde", "mean_h_ar_power")})
    meta.update(bank_role=role, bank_entropy=fields, bank_samples=samples,
                bank_generation_seconds=time.perf_counter()-started,
                alice=c["positions"]["alice"], ris=c["positions"]["ris"], receiver_xy=layout.receiver_xy.tolist())
    return c, layout, bank, meta


def corrected(values, delta, bounds):
    if not np.isfinite(delta) or delta < 0: raise ValueError("Delta must be finite and nonnegative")
    y = physical_allocation(values, len(values)-2, bounds)
    old = y[0]
    y[0] = max(bounds["mb"][0], old-delta)
    y[1:-1] *= (1-y[0])/(1-old)
    return physical_allocation(y, len(values)-2, bounds)


def pso_rng(plan, c, job, seed, case_index):
    role = {"development": "development_pso", "final": "final_pso", "timing": "timing_pso"}[plan["stage"]]
    # Method and budget deliberately absent: paired initial particles/random coefficients.
    return np.random.default_rng(np.random.SeedSequence(entropy(c["root_seed"], job, role, seed, case_index)))
