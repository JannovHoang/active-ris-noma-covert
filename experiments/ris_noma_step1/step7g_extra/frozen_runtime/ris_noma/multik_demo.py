"""7a foundation demo: compatibility, small PSO data, one mixed-K GNN.

This is a functional end-to-end check, not the final multi-K benchmark. Labels
are selected before their audit; an audit failure never removes a train label.
The GNN test audit opens only after the shared checkpoint/predictions freeze.
"""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import json
import platform
import shutil
import time

from filelock import FileLock
import numpy as np
import scipy
import torch

from .channels import build_bank, public_order
from .expanded_data import write_table
from .expanded_scenarios import ROOT, scenario_rng
from .geometry import sample_layout
from .graph_data import statistical_graph as legacy_graph
from .learning_data import read_json, sha256
from .learning_train import configure_runtime
from .multik_core import MultiKObjective, assess_allocation, decode_allocation, extend_config, original_allocation
from .multik_graph import SCHEMA_VERSION, fit_normalizer, pack_graphs, statistical_graph
from .multik_models import (VariableAllocationGNN, balanced_allocation_loss, import_legacy_checkpoint,
                            predict, tensor_batch)
from .objective import CachedObjective
from .pso import PSOSettings, optimize
from .random_streams import make_rng
from .selection_bank import check_experiment as check_6g

STREAMS = {"search": 72, "pso": 73, "selection": 74, "label_audit": 75, "gnn_audit": 76}
NUMERIC_FILES = ("run_step7a.py", "ris_noma/multik_core.py", "ris_noma/multik_graph.py",
                 "ris_noma/multik_models.py", "ris_noma/multik_demo.py")


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def immutable(path, value):
    if Path(path).exists():
        if read_json(path) != value:
            raise ValueError(f"Frozen record differs: {Path(path).name}")
    else:
        write_json(path, value)


def committed(path, create):
    path = Path(path)
    receipt = path.with_name(path.name + ".receipt.json")
    if receipt.exists():
        if not path.is_file() or read_json(receipt)["sha256"] != sha256(path):
            raise ValueError(f"Changed completed record: {path.name}")
        return read_json(path)
    result = create()
    write_json(path, result)
    write_json(receipt, {"sha256": sha256(path)})
    return result


def freeze(output, phase, paths):
    content = {"phase": phase, "sha256": {p.relative_to(output).as_posix(): sha256(p) for p in paths}}
    immutable(output / f"{phase}_frozen.json", content)
    return sha256(output / f"{phase}_frozen.json")


def validate_settings(c):
    expected = {"training_public_counts", "unseen_public_counts", "splits", "power_dbm", "xi", "search_margin",
                "search_samples", "selection_samples", "audit_samples", "pso_seeds", "pso_evaluations", "alpha", "training"}
    if set(c) != expected:
        raise ValueError("Unexpected 7a configuration fields")
    for key in ("training_public_counts", "unseen_public_counts"):
        if (not isinstance(c[key], list) or not c[key] or c[key] != sorted(set(c[key]))
                or any(type(k) is not int or k < 1 for k in c[key])):
            raise ValueError("Declare nonempty distinct positive public counts")
    if set(c["training_public_counts"]) & set(c["unseen_public_counts"]):
        raise ValueError("Unseen K must not appear in training/validation labels")
    if set(c["splits"]) != {"train", "validation", "test"} or any(not v for v in c["splits"].values()):
        raise ValueError("Need nonempty layout groups for train, validation and test")
    ids = [i for a in c["splits"].values() for i in a]
    if len(set(ids)) != len(ids) or any(type(i) is not int or i < 800 for i in ids):
        raise ValueError("Use distinct new layout IDs >=800, grouped across all K")
    if (not c["power_dbm"] or any(not np.isfinite(p) for p in c["power_dbm"])
            or len(set(c["power_dbm"])) != len(c["power_dbm"]) or not c["xi"]
            or c["xi"] != sorted(set(c["xi"])) or not 0 <= c["search_margin"] < 1
            or any(not 0 < x < 1 or x + c["search_margin"] >= 1 for x in c["xi"])):
        raise ValueError("Invalid powers, DEP requirements or search margin")
    if not c["pso_seeds"] or len(set(c["pso_seeds"])) != len(c["pso_seeds"]) or any(type(s) is not int or s < 0 for s in c["pso_seeds"]):
        raise ValueError("Use distinct nonnegative PSO seeds")
    for key in ("search_samples", "selection_samples", "audit_samples"):
        if type(c[key]) is not int or c[key] < 1000:
            raise ValueError("Use at least 1000 samples per bank")
    if type(c["pso_evaluations"]) is not int or c["pso_evaluations"] < 30 or c["pso_evaluations"] % 30 or not 0 < c["alpha"] < 1:
        raise ValueError("PSO budget must be whole swarms of 30; require 0<alpha<1")
    t = c["training"]
    if set(t) != {"seed", "epochs", "batch_size", "learning_rate", "weight_decay", "gradient_clip", "hidden", "layers", "mb_weight", "public_weight", "s_weight"}:
        raise ValueError("Unexpected training fields")
    for name in ("epochs", "batch_size", "hidden", "layers"):
        if type(t[name]) is not int or t[name] < 1:
            raise ValueError("Positive integer training dimensions required")
    if type(t["seed"]) is not int or t["seed"] < 0:
        raise ValueError("Nonnegative training seed required")
    for name in ("learning_rate", "gradient_clip", "mb_weight", "public_weight", "s_weight"):
        if not np.isfinite(t[name]) or t[name] <= 0:
            raise ValueError("Positive finite training coefficients required")
    if not np.isfinite(t["weight_decay"]) or t["weight_decay"] < 0:
        raise ValueError("Nonnegative weight decay required")
    return deepcopy(c)


def inspect(previous, config):
    previous = Path(previous).resolve()
    check_6g(previous)
    c = validate_settings(config)
    prior = read_json(previous / "plan.json")
    seen = set(prior["validation_layout_ids"] + prior["test_layout_ids"])
    if seen & {i for a in c["splits"].values() for i in a}:
        raise ValueError("Use new layout IDs outside the prior 6g study")
    source = read_json(previous / "source_physics_plan.json")
    if source["base_config"]["n_public"] != 2:
        raise ValueError("7a imports the completed K=2 source as its regression baseline")
    settings = prior["search_config"]
    pso = PSOSettings(swarm_size=30, sweeps=c["pso_evaluations"] // 30,
        inertia_start=settings["inertia_start"], inertia_end=settings["inertia_end"],
        cognitive=settings["cognitive"], social=settings["social"], velocity_max=settings["velocity_max"],
        initial_velocity_scale=settings["initial_velocity_scale"])
    code = tuple(dict.fromkeys((*prior["code_sha256"], *NUMERIC_FILES)))
    plan = {"step": "7a", "purpose": "variable-K foundation; small end-to-end functional demo",
            "schema": SCHEMA_VERSION, "config": c, "streams": STREAMS, "pso_settings": asdict(pso),
            "source_sha256": {n: sha256(previous / n) for n in ("completion.json", "plan.json", "source_physics_plan.json", "model/gnn.pt")},
            "code_sha256": {n: sha256(ROOT / n) for n in code},
            "versions": {"system": platform.system(), "python": platform.python_version(), "numpy": np.__version__,
                         "scipy": scipy.__version__, "torch": str(torch.__version__)},
            "legacy_model": prior["selected_model"], "legacy_compatibility_layout": prior["test_layouts"][0],
            "legacy_compatibility_plan": {"streams": prior["streams"], "power_dbm": prior["power_dbm"]},
            "los_extension": "preserve supplied Bob/public/Willie vectors; additional public vectors fixed U[0,2pi) using seed [root,7101,public_ID]; not geometry-based steering",
            "training_selection": "best allocation loss on validation graphs; one seed; no physical gradients; foundation only",
            "labels": "search point feasible -> independent selection screen -> highest search-bank rate; independent label audit never filters labels",
            "test_policy": "shared checkpoint and all predictions frozen before fresh GNN audit; no test-based tuning",
            "architecture_has_fixed_K_limit": False,
            "final_paper_benchmark": False}
    return plan, source


def random_stream(base, lid, k, role, *details):
    return np.random.default_rng(np.random.SeedSequence([base["root_seed"], lid, k, STREAMS[role], *details]))


def context(plan, source, lid, k, role, samples):
    base = source["base_config"]
    c = extend_config(base, k)
    c.update(layout_id=lid, n_fading=samples, total_dbm=plan["config"]["power_dbm"][0])
    geometry = scenario_rng(base, lid, "receiver_geometry")
    for name, width in source["experiment"]["position_half_width_m"].items():
        c["positions"][name] = (np.asarray(base["positions"][name]) + geometry.uniform(-width, width, 2)).tolist()
    layout = sample_layout(c, make_rng(c["root_seed"], lid, "geometry"))
    bank = build_bank(c, layout, random_stream(c, lid, k, role))
    return c, layout, bank


def compact_score(c, layout, bank, order, allocation, xi, family, alpha):
    summary, screen = assess_allocation(c, layout, bank, order, allocation, xi, family, alpha)
    return {"passed": screen["accepted"], "bob_rate": summary["bob_mean_rate_bit_s_hz"],
            "gated_bob_rate": summary["bob_mean_rate_bit_s_hz"] if screen["accepted"] else 0.,
            "dep_gamma": summary["dep_min"], "dep_empirical": screen["dep_empirical_min"],
            "dep_lower": screen["dep_uniform_lower"], "bob_outage": summary["bob_outage_estimate"],
            "bob_outage_upper": screen["bob_outage_upper"],
            "public_rate_lower_sic": screen["public_rate_lower_sic_order"],
            "qos_lower_margin": float(np.min(np.asarray(screen["public_rate_lower_sic_order"]) - np.asarray(c["requirements"]["qos_public"])[order])),
            "failed_checks": [name for name, ok in screen["checks"].items() if not ok]}


def compatibility(plan, source, legacy_path):
    from .dep_label_experiment import channel_context
    c, layout, bank = channel_context(plan["legacy_compatibility_plan"], source,
                                      plan["legacy_compatibility_layout"], "test_search", 10000)
    order = public_order(bank)
    new, checkpoint, old = import_legacy_checkpoint(legacy_path)
    graph = statistical_graph(c, layout, bank, order, .85)
    previous = legacy_graph(c, layout, bank, order, .85)
    for field in ("node_raw", "global_raw", "edge_index", "edge_attr"):
        if not np.array_equal(np.asarray(graph[field]), previous[field]):
            raise ArithmeticError("K=2 graph differs from the existing representation")
    data = tensor_batch(pack_graphs([graph], checkpoint["normalization"]))
    with torch.inference_mode():
        a = old(**{key: data[key] for key in ("node", "global_features", "edge_index", "edge_attr")}).numpy()
        b = new(**data).numpy()
    error = float(np.max(np.abs(a - b)))
    if error > 2e-7:
        raise ArithmeticError("Imported weights changed the K=2 prediction")
    fast, legacy = MultiKObjective(c, layout, bank, order), CachedObjective(c, layout, bank, order)
    maximum = 0.
    for z in np.random.default_rng(710).random((20, 3)):
        x, y = fast.evaluate(z, .85), legacy.evaluate(z, .85)
        maximum = max(maximum, abs(x.score - y.score), abs(x.violation - y.violation), abs(x.metrics["dep_min"] - y.metrics["dep_min"]))
        if x.feasible != y.feasible or maximum > 1e-10:
            raise ArithmeticError("Generic objective differs from the K=2 baseline")
    return {"legacy_checkpoint_sha256": sha256(legacy_path), "same_state_dict_keys": True,
            "k2_graph_exact_match": True, "legacy_prediction_max_abs_difference": error,
            "cached_objective_max_abs_difference": maximum, "objective_points_checked": 20,
            "parameters": sum(p.numel() for p in new.parameters()),
            "interpretation": "K=2 regression only; importing weights is not multi-K training"}


def search_group(plan, source, lid, k, split):
    c = plan["config"]
    physical, layout, bank = context(plan, source, lid, k, "search", c["search_samples"])
    order = public_order(bank)
    eligible = MultiKObjective(physical, layout, bank, order).layout_eligible
    rows, evaluations = [], 0
    for pi, power in enumerate(c["power_dbm"]):
        radio = dict(physical, total_dbm=power)
        objective = MultiKObjective(radio, layout, bank, order)
        for xi_index, xi in enumerate(c["xi"]):
            case = {"layout_id": lid, "n_public": k, "split": split, "power_dbm": power, "xi": xi,
                    "graph": statistical_graph(radio, layout, bank, order, xi), "proposals": []}
            if eligible:
                for seed in c["pso_seeds"]:
                    result = optimize(lambda z: objective.evaluate(z, xi + c["search_margin"]), k + 1,
                                      PSOSettings(**plan["pso_settings"]), random_stream(radio, lid, k, "pso", seed, pi, xi_index))
                    mb, public, s = decode_allocation(result.position, k, radio["bounds"])
                    y, z = original_allocation(mb, public, s, order, radio["bounds"])
                    case["proposals"].append({"seed": seed, "allocation": y.tolist(), "z_sic": z.tolist(),
                                               "search_rate": result.outcome.score, "search_feasible": result.outcome.feasible})
                    evaluations += result.evaluations
            rows.append(case)
    return {"layout_id": lid, "n_public": k, "split": split, "order": order.tolist(),
            "eligible": eligible, "rows": rows, "pso_evaluations": evaluations,
            "positions": {"receiver_xy": layout.receiver_xy.tolist(), "alice": physical["positions"]["alice"], "ris": physical["positions"]["ris"]}}


def select_group(plan, source, search, frozen_hash):
    c, lid, k = plan["config"], search["layout_id"], search["n_public"]
    physical, layout, bank = context(plan, source, lid, k, "selection", c["selection_samples"])
    order = np.asarray(search["order"], int)
    groups = len(c["training_public_counts"]) * sum(map(len, c["splits"].values())) * len(c["power_dbm"])
    alpha, family = c["alpha"] / groups, len(c["xi"]) * len(c["pso_seeds"])
    rows = []
    for case in search["rows"]:
        proposals = []
        radio = dict(physical, total_dbm=case["power_dbm"])
        for proposal in case["proposals"]:
            score = compact_score(radio, layout, bank, order, proposal["allocation"], case["xi"], family, alpha)
            proposals.append({**proposal, "selection": score})
        passing = [p for p in proposals if p["search_feasible"] and p["selection"]["passed"]]
        winner = min(passing, key=lambda p: (-p["search_rate"], p["seed"])) if passing else None
        rows.append({key: value for key, value in case.items() if key != "proposals"} | {
            "proposals": proposals, "label_available": winner is not None,
            "allocation": None if winner is None else winner["allocation"],
            "selected_seed": None if winner is None else winner["seed"],
            "selection": None if winner is None else winner["selection"]})
    return {"layout_id": lid, "n_public": k, "order": search["order"], "rows": rows, "search_frozen_sha256": frozen_hash}


def audit_group(plan, source, selected, frozen_hash):
    c, lid, k = plan["config"], selected["layout_id"], selected["n_public"]
    physical, layout, bank = context(plan, source, lid, k, "label_audit", c["audit_samples"])
    order = np.asarray(selected["order"], int)
    groups = len(c["training_public_counts"]) * sum(map(len, c["splits"].values())) * len(c["power_dbm"])
    rows = []
    for row in selected["rows"]:
        result = compact_score(dict(physical, total_dbm=row["power_dbm"]), layout, bank, order,
                               row["allocation"], row["xi"], len(c["xi"]), c["alpha"] / groups) if row["label_available"] else None
        rows.append({**row, "order": selected["order"], "label_audit": result, "heldout_used_for_label_filter": False})
    return {"layout_id": lid, "n_public": k, "rows": rows, "selection_frozen_sha256": frozen_hash}


def train_shared(output, plan, source, rows):
    path = output / "training_frozen.json"
    if path.exists():
        result = read_json(path)
        if any(sha256(output / name) != h for name, h in result["sha256"].items()):
            raise ValueError("A frozen shared model/history changed")
        return result
    data = {role: [r for r in rows if r["split"] == role and r["label_available"]] for role in ("train", "validation", "test")}
    c = plan["config"]; t = c["training"]
    if any(not data[role] for role in data) or any(k not in {r["n_public"] for r in data[role]} for role in ("train", "validation") for k in c["training_public_counts"]):
        raise ValueError("Insufficient labeled graphs for the declared mixed-K foundation demo; report coverage and use a new config/output")
    normalizer = fit_normalizer([r["graph"] for r in data["train"]])
    immutable(output / "normalization.json", normalizer)
    configure_runtime(t["seed"], 1, "cpu")
    model = VariableAllocationGNN(source["base_config"]["bounds"], t["hidden"], t["layers"])
    optimizer = torch.optim.Adam(model.parameters(), lr=t["learning_rate"], weight_decay=t["weight_decay"])
    random = np.random.default_rng(t["seed"])
    def batch(records):
        d = tensor_batch(pack_graphs([r["graph"] for r in records], normalizer, [r["allocation"] for r in records]))
        return d, d.pop("target")
    def loss_on(records):
        model.eval()
        d, target = batch(records)
        with torch.inference_mode():
            return float(balanced_allocation_loss(model(**d), target, d["node_mask"], t["mb_weight"], t["public_weight"], t["s_weight"]))
    best, best_epoch, best_state = loss_on(data["validation"]), 0, deepcopy(model.state_dict())
    history = [{"epoch": 0, "train_loss": loss_on(data["train"]), "validation_loss": best}]
    for epoch in range(1, t["epochs"] + 1):
        model.train()
        indices = random.permutation(len(data["train"]))
        for start in range(0, len(indices), t["batch_size"]):
            records = [data["train"][i] for i in indices[start:start + t["batch_size"]]]
            d, target = batch(records)
            optimizer.zero_grad(set_to_none=True)
            loss = balanced_allocation_loss(model(**d), target, d["node_mask"], t["mb_weight"], t["public_weight"], t["s_weight"])
            if not torch.isfinite(loss):
                raise ArithmeticError("Nonfinite mixed-K training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), t["gradient_clip"], error_if_nonfinite=True)
            optimizer.step()
        train_loss, val_loss = loss_on(data["train"]), loss_on(data["validation"])
        history.append({"epoch": epoch, "train_loss": train_loss, "validation_loss": val_loss})
        if val_loss < best:
            best, best_epoch, best_state = val_loss, epoch, deepcopy(model.state_dict())
        if epoch == 1 or epoch % 20 == 0:
            print(f"Shared GNN epoch={epoch}: train={train_loss:.6g}, validation={val_loss:.6g}", flush=True)
    model.load_state_dict(best_state)
    checkpoint = {"format": "ris_noma_variable_k_7a_v1", "state_dict": best_state,
                  "training": t, "bounds": source["base_config"]["bounds"], "normalization": normalizer,
                  "schema": SCHEMA_VERSION, "trained_public_counts": c["training_public_counts"],
                  "selected_epoch": best_epoch, "selection": "validation_balanced_allocation_loss",
                  "test_used_for_selection": False, "foundation_demo_only": True, "plan_sha256": sha256(output / "plan.json")}
    (output / "models").mkdir(exist_ok=True)
    torch.save(checkpoint, output / "models/shared_gnn.pt")
    write_table(output / "training_history.csv", history)
    result = {"trained_public_counts": c["training_public_counts"], "one_shared_checkpoint": True,
              "selected_epoch": best_epoch, "validation_loss": best, "training_seed": t["seed"],
              "parameters": sum(p.numel() for p in model.parameters()),
              "labeled_counts": {role: len(a) for role, a in data.items()},
              "test_used_for_selection": False,
              "sha256": {n: sha256(output / n) for n in ("models/shared_gnn.pt", "normalization.json", "training_history.csv")}}
    immutable(path, result)
    return result


def load_shared(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if (checkpoint.get("format") != "ris_noma_variable_k_7a_v1" or checkpoint["test_used_for_selection"]
            or checkpoint["schema"] != SCHEMA_VERSION):
        raise ValueError("Not a supported validation-selected variable-K checkpoint")
    t = checkpoint["training"]
    model = VariableAllocationGNN(checkpoint["bounds"], t["hidden"], t["layers"])
    model.load_state_dict(checkpoint["state_dict"], strict=True); model.eval()
    return model, checkpoint


def create_predictions(output, plan, source, dataset_rows, model, checkpoint):
    c = plan["config"]
    labels = {(r["layout_id"], r["n_public"], r["power_dbm"], r["xi"]): r for r in dataset_rows if r["split"] == "test"}
    rows, shape_checks = [], []
    for lid in c["splits"]["test"]:
        for k in sorted(c["training_public_counts"] + c["unseen_public_counts"]):
            physical, layout, bank = context(plan, source, lid, k, "search", c["search_samples"])
            order = public_order(bank)
            for power in c["power_dbm"]:
                for xi in c["xi"]:
                    graph = statistical_graph(dict(physical, total_dbm=power), layout, bank, order, xi)
                    y = predict(model, [graph], checkpoint["normalization"])[0]
                    reference = labels.get((lid, k, power, xi))
                    rows.append({"layout_id": lid, "n_public": k, "power_dbm": power, "xi": xi,
                                 "method": "shared_gnn_demo", "allocation": y.tolist(), "order": order.tolist(),
                                 "unseen_K": k in c["unseen_public_counts"]})
                    if reference is not None and reference["label_available"]:
                        rows.append({**rows[-1], "method": "pso_label_reference", "allocation": reference["allocation"]})
            shape_checks.append({"n_public": k, "nodes": k + 1, "edges": k * (k + 1), "outputs": k + 2,
                                 "parameters": sum(p.numel() for p in model.parameters()), "checkpoint_sha256": sha256(output / "models/shared_gnn.pt")})
            del bank
    return {"training_frozen_sha256": sha256(output / "training_frozen.json"), "rows": rows, "shape_checks": shape_checks,
            "interpretation": "same model and parameter set at every K; physical quality is a separate audit"}


def audit_predictions(output, plan, source, predictions):
    c, rows = plan["config"], []
    ks = sorted(c["training_public_counts"] + c["unseen_public_counts"])
    groups = len(c["splits"]["test"]) * len(ks) * len(c["power_dbm"])
    for lid in c["splits"]["test"]:
        for k in ks:
            physical, layout, bank = context(plan, source, lid, k, "gnn_audit", c["audit_samples"])
            for r in predictions["rows"]:
                if (r["layout_id"], r["n_public"]) != (lid, k):
                    continue
                score = compact_score(dict(physical, total_dbm=r["power_dbm"]), layout, bank, np.asarray(r["order"], int),
                                      r["allocation"], r["xi"], 2 * len(c["xi"]), c["alpha"] / groups)
                rows.append({**r, **score})
            del bank
    return {"test_predictions_sha256": sha256(output / "test_predictions.json"), "rows": rows,
            "test_used_for_selection": False, "foundation_demo_only": True}


def run(previous, output, config, resume=False):
    previous, output = Path(previous).resolve(), Path(output).resolve()
    if previous == output or previous in output.parents or output in previous.parents:
        raise ValueError("Use a separate non-nested output directory")
    configure_runtime(0, 1, "cpu")
    plan, source = inspect(previous, config)
    output.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output.with_name(f".{output.name}.lock")), timeout=0):
        if output.exists() and any(output.iterdir()):
            if not resume or not (output / "plan.json").exists() or read_json(output / "plan.json") != plan:
                raise ValueError("Use a new output, or --resume with unchanged code, inputs, config and versions")
            if (output / "completion.json").is_file():
                return check(output)
        output.mkdir(exist_ok=True)
        immutable(output / "plan.json", plan)
        immutable(output / "source_physics_plan.json", source)
        (output / "models").mkdir(exist_ok=True)
        legacy = output / "models/legacy_6f_gnn.pt"
        if not legacy.exists():
            shutil.copyfile(previous / "model/gnn.pt", legacy)
        if sha256(legacy) != plan["source_sha256"]["model/gnn.pt"]:
            raise ValueError("Legacy checkpoint changed")
        committed(output / "compatibility.json", lambda: compatibility(plan, source, legacy))
        c = plan["config"]
        jobs = [(lid, k, role) for role, ids in c["splits"].items() for lid in ids for k in c["training_public_counts"]]
        searches, paths = [], []
        for index, (lid, k, role) in enumerate(jobs, 1):
            path = output / "label_search" / f"layout_{lid:06d}_k{k}.json"
            searches.append(committed(path, lambda lid=lid, k=k, role=role: search_group(plan, source, lid, k, role)))
            paths.append(path)
            print(f"PSO data group {index}/{len(jobs)}: layout={lid}, K={k}", flush=True)
        search_hash = freeze(output, "label_search", paths)
        choices, paths = [], []
        for index, search in enumerate(searches, 1):
            path = output / "label_selection" / f"layout_{search['layout_id']:06d}_k{search['n_public']}.json"
            choices.append(committed(path, lambda search=search: select_group(plan, source, search, search_hash)))
            paths.append(path)
            print(f"Independent label selection {index}/{len(jobs)}", flush=True)
        selection_hash = freeze(output, "label_selection", paths)
        audits, paths = [], []
        for index, selected in enumerate(choices, 1):
            path = output / "label_audit" / f"layout_{selected['layout_id']:06d}_k{selected['n_public']}.json"
            audits.append(committed(path, lambda selected=selected: audit_group(plan, source, selected, selection_hash)))
            paths.append(path)
            print(f"Independent label audit {index}/{len(jobs)}", flush=True)
        freeze(output, "label_audit", paths)
        records = [r for group in audits for r in group["rows"]]
        immutable(output / "dataset_records.json", {"schema": SCHEMA_VERSION, "records": records,
                  "heldout_used_for_label_filter": False, "group_split_across_K": True})
        trained = train_shared(output, plan, source, records)
        model, checkpoint = load_shared(output / "models/shared_gnn.pt")
        predictions = committed(output / "test_predictions.json", lambda: create_predictions(output, plan, source, records, model, checkpoint))
        audit = committed(output / "gnn_audit.json", lambda: audit_predictions(output, plan, source, predictions))
        coverage = []
        for role in c["splits"]:
            for k in c["training_public_counts"]:
                group = [r for r in records if r["split"] == role and r["n_public"] == k]
                available = [r for r in group if r["label_available"]]
                coverage.append({"split": role, "n_public": k, "cases": len(group), "labels": len(available),
                                 "label_audit_passes": sum(r["label_audit"]["passed"] for r in available)})
        physical_results = []
        for k in sorted(c["training_public_counts"] + c["unseen_public_counts"]):
            for method in ("shared_gnn_demo", "pso_label_reference"):
                group = [r for r in audit["rows"] if (r["n_public"], r["method"]) == (k, method)]
                if group:
                    physical_results.append({"n_public": k, "method": method, "cases": len(group),
                        "passed": sum(r["passed"] for r in group), "gated_rate": float(np.mean([r["gated_bob_rate"] for r in group])),
                        "unseen_K": k in c["unseen_public_counts"]})
        write_table(output / "label_coverage.csv", coverage)
        write_table(output / "smoke_physics.csv", physical_results)
        write_table(output / "model_shapes.csv", predictions["shape_checks"])
        write_table(output / "smoke_physics_runs.csv", audit["rows"])
        result = {"step": "7a", "complete": True, "foundation_only": True, "final_paper_benchmark": False,
                  "compatibility": read_json(output / "compatibility.json"), "shared_training": trained,
                  "coverage": coverage, "smoke_physics": physical_results,
                  "model_shapes": predictions["shape_checks"], "planned_cases": len(records),
                  "pso_evaluations": sum(r["pso_evaluations"] for r in searches),
                  "source_6g_unchanged": all(sha256(previous / n) == h for n, h in plan["source_sha256"].items()),
                  "warning_scope": "one seed, tiny graph dataset and one new test layout by default; no superiority/generalization guarantee"}
        write_json(output / "summary.json", result)
        files = [p for p in output.rglob("*") if p.is_file() and p.name != "completion.json" and not p.name.endswith(".tmp") and "figures" not in p.parts]
        immutable(output / "completion.json", {"sha256": {p.relative_to(output).as_posix(): sha256(p) for p in sorted(files)}})
        return check(output)


def check(output):
    output = Path(output).resolve()
    manifest = read_json(output / "completion.json")["sha256"]
    required = {"plan.json", "summary.json", "source_physics_plan.json", "compatibility.json", "dataset_records.json",
                "models/shared_gnn.pt", "training_frozen.json", "test_predictions.json", "gnn_audit.json",
                "label_search_frozen.json", "label_selection_frozen.json", "label_audit_frozen.json"}
    if not required.issubset(manifest):
        raise ValueError("Missing required 7a result files")
    for name, digest in manifest.items():
        path = (output / name).resolve()
        if not path.is_relative_to(output) or not path.is_file() or sha256(path) != digest:
            raise ValueError(f"7a result missing or changed: {name}")
    for phase in ("label_search", "label_selection", "label_audit"):
        if any(manifest.get(n) != h for n, h in read_json(output / f"{phase}_frozen.json")["sha256"].items()):
            raise ValueError("Changed frozen data phase")
    result, plan = read_json(output / "summary.json"), read_json(output / "plan.json")
    data = read_json(output / "dataset_records.json")
    if (not result["complete"] or result["step"] != "7a" or not result["foundation_only"]
            or data["heldout_used_for_label_filter"] or read_json(output / "gnn_audit.json")["test_used_for_selection"]):
        raise ValueError("Unexpected test/label filtering policy")
    marks = [(r["layout_id"], r["n_public"], r["power_dbm"], r["xi"]) for r in data["records"]]
    expected = sum(map(len, plan["config"]["splits"].values())) * len(plan["config"]["training_public_counts"]) * len(plan["config"]["power_dbm"]) * len(plan["config"]["xi"])
    if len(marks) != expected or len(set(marks)) != expected:
        raise ValueError("Missing or duplicate graph cases")
    split_by_id = {lid: role for role, ids in plan["config"]["splits"].items() for lid in ids}
    for row in data["records"]:
        if row["split"] != split_by_id[row["layout_id"]] or row["heldout_used_for_label_filter"]:
            raise ValueError("A layout crossed splits across K, or audit filtered labels")
        if row["label_available"] != (row["allocation"] is not None):
            raise ValueError("Label availability flag differs from its allocation")
    for path in (output / "label_selection").glob("layout_*.json"):
        if not path.name.endswith(".receipt.json") and read_json(path)["search_frozen_sha256"] != sha256(output / "label_search_frozen.json"):
            raise ValueError("Label selection belongs to another search freeze")
    for path in (output / "label_audit").glob("layout_*.json"):
        if not path.name.endswith(".receipt.json") and read_json(path)["selection_frozen_sha256"] != sha256(output / "label_selection_frozen.json"):
            raise ValueError("Label audit belongs to another selection freeze")
    trained, predictions, audit = (read_json(output / name) for name in ("training_frozen.json", "test_predictions.json", "gnn_audit.json"))
    if (predictions["training_frozen_sha256"] != sha256(output / "training_frozen.json")
            or audit["test_predictions_sha256"] != sha256(output / "test_predictions.json")
            or any(manifest.get(n) != h for n, h in trained["sha256"].items())):
        raise ValueError("Shared checkpoint/prediction/audit freeze links differ")
    model_hash = sha256(output / "models/shared_gnn.pt")
    if (any(r["checkpoint_sha256"] != model_hash for r in predictions["shape_checks"])
            or len({r["parameters"] for r in predictions["shape_checks"]}) != 1):
        raise ValueError("Different models or parameter counts were used across K")
    return result
