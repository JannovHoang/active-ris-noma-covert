"""7b label generation; unchanged scalar-noise physics and variable-K decoder.

Each layout/K has independent search, selection and audit streams. A fixed pool
is shared across DEP targets at the SAME power; powers never share allocations.
"""
from types import SimpleNamespace

import numpy as np

from .channels import build_bank, public_order
from .expanded_scenarios import scenario_rng
from .geometry import sample_layout
from .multik_core import (MultiKObjective, assess_allocation, decode_allocation,
                         extend_config, original_allocation)
from .multik_graph import statistical_graph
from .pso import PSOSettings, optimize
from .random_streams import make_rng

STREAMS = {"search": 82, "pso": 83, "selection": 84, "heldout": 85}


def stream(plan, source, job, role, *details):
    if role not in STREAMS or plan["streams"] != STREAMS:
        raise ValueError("Unexpected 7b random stream policy")
    fields = [source["base_config"]["root_seed"], job["layout_id"], job["n_public"], STREAMS[role], *details]
    return np.random.default_rng(np.random.SeedSequence(fields))


def geometry(plan, source, job, samples):
    c = extend_config(source["base_config"], job["n_public"])
    c.update(layout_id=job["layout_id"], n_fading=samples, total_dbm=plan["config"]["power_dbm"][0])
    rng = scenario_rng(source["base_config"], job["layout_id"], "receiver_geometry")
    for name, width in source["experiment"]["position_half_width_m"].items():
        c["positions"][name] = (np.asarray(source["base_config"]["positions"][name]) + rng.uniform(-width, width, 2)).tolist()
    layout = sample_layout(c, make_rng(c["root_seed"], job["layout_id"], "geometry"))
    return c, layout


def context(plan, source, job, role):
    sizes = {"search": "search_samples", "selection": "selection_samples", "heldout": "heldout_samples"}
    c, layout = geometry(plan, source, job, plan["config"][sizes[role]])
    full = build_bank(c, layout, stream(plan, source, job, role))
    # The old evaluator needs h_ar, but not h_ru/phase after channel aggregation.
    # Drop only unused storage; all remaining arrays and physics are unchanged.
    names = ("h_ar", "h_tilde", "g_scalar_tilde", "signal_gain_tilde", "noise_gain_tilde", "mean_h_ar_power")
    bank = SimpleNamespace(**{name: getattr(full, name) for name in names})
    return c, layout, bank


def raw_assessment(c, layout, bank, order, allocation, family, alpha):
    """Compute statistics once; acceptance at every declared DEP is exact.

    No DEP selection is hidden here: non-DEP point/bound checks are retained,
    while both Gamma DEP and the uniform empirical DEP lower bound are exposed.
    """
    info, screen = assess_allocation(c, layout, bank, order, allocation, .5, family, alpha)
    point = {name: ok for name, ok in info["constraints"]["checks"].items() if name != "covertness_gamma"}
    bounds = {name: ok for name, ok in screen["checks"].items()
              if name not in ("dep_uniform_lower_ge_xi", "gamma_sample_model_checks")}
    return {"bob_rate": info["bob_mean_rate_bit_s_hz"], "dep_gamma": info["dep_min"],
            "dep_empirical": screen["dep_empirical_min"], "dep_lower": screen["dep_uniform_lower"],
            "public_mean_rate_sic": info["public_mean_rate_bit_s_hz"],
            "public_rate_lower_sic": screen["public_rate_lower_sic_order"],
            "qos_lower_margin": float(np.min(np.asarray(screen["public_rate_lower_sic_order"])
                                  - np.asarray(c["requirements"]["qos_public"])[order])),
            "bob_outage": info["bob_outage_estimate"], "bob_outage_upper": screen["bob_outage_upper"],
            "point_checks_without_dep": point, "bound_checks_without_dep": bounds,
            "constants": screen["constants"]}


def at_target(raw, xi):
    checks = {"covertness_gamma": bool(raw["dep_gamma"] >= xi),
              "dep_uniform_lower_ge_xi": bool(raw["dep_lower"] >= xi),
              **raw["point_checks_without_dep"], **raw["bound_checks_without_dep"]}
    passed = bool(all(checks.values()))
    return {"passed": passed, "failed_checks": [name for name, ok in checks.items() if not ok],
            "gated_bob_rate": raw["bob_rate"] if passed else 0.}


def choose_label(candidates, xi_index, xi):
    eligible = [p for p in candidates if p["search_feasible_by_xi"][xi_index]
                and at_target(p["selection"], xi)["passed"]]
    return min(eligible, key=lambda p: (-p["search_rate"], p["candidate_id"])) if eligible else None


def search_group(plan, source, job):
    settings = plan["config"]
    try:
        c, layout, bank = context(plan, source, job, "search")
    except ValueError as error:
        # Only the explicit geometric rejection is a missing-data outcome.
        if not str(error).startswith("Layout bị loại:"):
            raise
        return {**job, "geometry_valid": False, "geometry_error": str(error),
                "layout_eligible": False, "positions": None, "order": None,
                "powers": [], "pso_evaluations": 0, "pool_target_checks": 0}
    order = public_order(bank)
    eligible = MultiKObjective(c, layout, bank, order).layout_eligible
    positions = {"receiver_xy": layout.receiver_xy.tolist(), "alice": c["positions"]["alice"], "ris": c["positions"]["ris"]}
    powers, total_evaluations, target_checks = [], 0, 0
    for pi, power in enumerate(settings["power_dbm"]):
        radio = dict(c, total_dbm=power)
        objective = MultiKObjective(radio, layout, bank, order)
        graphs = [statistical_graph(radio, layout, bank, order, xi) for xi in settings["xi"]]
        proposals = []
        if eligible:
            for ti, xi in enumerate(settings["xi"]):
                for seed in settings["pso_seeds"]:
                    result = optimize(lambda z: objective.evaluate(z, xi + settings["search_margin"]),
                                      job["n_public"] + 1, PSOSettings(**plan["pso_settings"]),
                                      stream(plan, source, job, "pso", seed, pi, ti))
                    mb, public, s = decode_allocation(result.position, job["n_public"], radio["bounds"])
                    y, canonical = original_allocation(mb, public, s, order, radio["bounds"])
                    outcomes = [objective.evaluate(canonical, threshold + settings["search_margin"])
                                for threshold in settings["xi"]]
                    proposals.append({"candidate_id": f"p{pi}_d{ti}_s{seed}", "origin_xi": xi, "seed": seed,
                                      "allocation": y.tolist(), "z_sic": canonical.tolist(),
                                      "search_rate": outcomes[0].score, "search_dep_gamma": outcomes[0].metrics["dep_min"],
                                      "search_feasible_by_xi": [o.feasible for o in outcomes],
                                      "evaluations": result.evaluations, "feasible_evaluations": result.feasible_evaluations,
                                      "history": result.history})
                    total_evaluations += result.evaluations
                    target_checks += len(outcomes)
        powers.append({"power_dbm": power, "graphs": graphs, "candidates": proposals})
    return {**job, "geometry_valid": True, "geometry_error": None, "layout_eligible": eligible,
            "positions": positions, "order": order.tolist(), "powers": powers,
            "pso_evaluations": total_evaluations, "pool_target_checks": target_checks}


def select_group(plan, source, search, search_frozen_sha256):
    c = plan["config"]
    job = {name: search[name] for name in ("layout_id", "n_public", "split", "regime")}
    groups = len(plan["jobs"])
    family = len(c["power_dbm"]) * len(c["xi"]) * len(c["pso_seeds"])
    powers = []
    if search["geometry_valid"] and search["layout_eligible"]:
        physical, layout, bank = context(plan, source, job, "selection")
    for item in search["powers"]:
        candidates = []
        for proposal in item["candidates"]:
            raw = raw_assessment(dict(physical, total_dbm=item["power_dbm"]), layout, bank,
                                 np.asarray(search["order"], int), proposal["allocation"], family, c["alpha"] / (2 * groups))
            candidates.append({"candidate_id": proposal["candidate_id"], "origin_xi": proposal["origin_xi"],
                               "seed": proposal["seed"], "allocation": proposal["allocation"],
                               "search_rate": proposal["search_rate"],
                               "search_feasible_by_xi": proposal["search_feasible_by_xi"], "selection": raw})
        choices = []
        for ti, xi in enumerate(c["xi"]):
            winner = choose_label(candidates, ti, xi)
            reason = ("bob_mean_gain_rule_failed" if not search["layout_eligible"] else
                      "no_search_feasible_candidate" if not any(p["search_feasible_by_xi"][ti] for p in candidates)
                      else "no_candidate_passed_selection") if winner is None else None
            choices.append({"xi": xi, "label_available": winner is not None,
                            "selected_candidate_id": None if winner is None else winner["candidate_id"],
                            "selected_origin_xi": None if winner is None else winner["origin_xi"],
                            "allocation": None if winner is None else winner["allocation"],
                            "no_label_reason": reason})
        powers.append({"power_dbm": item["power_dbm"], "candidates": candidates, "choices": choices})
    if not search["geometry_valid"]:
        powers = [{"power_dbm": power, "candidates": [],
                   "choices": [{"xi": xi, "label_available": False, "selected_candidate_id": None,
                                "selected_origin_xi": None, "allocation": None, "no_label_reason": "invalid_geometry"}
                               for xi in c["xi"]]} for power in c["power_dbm"]]
    return {**job, "order": search["order"], "geometry_valid": search["geometry_valid"],
            "layout_eligible": search["layout_eligible"], "powers": powers,
            "search_frozen_sha256": search_frozen_sha256}


def audit_group(plan, source, search, selected, selection_frozen_sha256):
    c = plan["config"]
    job = {name: selected[name] for name in ("layout_id", "n_public", "split", "regime")}
    rows = []
    if any(r["label_available"] for p in selected["powers"] for r in p["choices"]):
        physical, layout, bank = context(plan, source, job, "heldout")
    for pi, item in enumerate(selected["powers"]):
        for ti, choice in enumerate(item["choices"]):
            winner = next((p for p in item["candidates"] if p["candidate_id"] == choice["selected_candidate_id"]), None)
            raw = raw_assessment(dict(physical, total_dbm=item["power_dbm"]), layout, bank,
                                 np.asarray(selected["order"], int), choice["allocation"],
                                 len(c["power_dbm"]) * len(c["xi"]), c["alpha"] / (2 * len(plan["jobs"]))) if winner else None
            graph = search["powers"][pi]["graphs"][ti] if search["geometry_valid"] else None
            rows.append({**job, "power_dbm": item["power_dbm"], **choice, "graph": graph,
                         "order": selected["order"], "geometry_valid": selected["geometry_valid"],
                         "layout_eligible": selected["layout_eligible"],
                         "search_rate": None if winner is None else winner["search_rate"],
                         "selection": None if winner is None else winner["selection"],
                         "heldout": None if raw is None else {**raw, **at_target(raw, choice["xi"])},
                         "heldout_used_for_label_filter": False})
    return {**job, "rows": rows, "selection_frozen_sha256": selection_frozen_sha256}
