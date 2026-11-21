"""Reuse a fixed label-generated proposal pool across covertness targets.

All requirements except xi must be identical. Existing validation bounds are
simultaneous over this entire pool. No original per-target PASS boolean or old
heldout result is used to rank candidates for the new target.
"""
import numpy as np
from .validation import family_constants, binomial_upper


def target_checks(c, order, proposal, assessment, xi, bounded=True):
    if not np.isfinite(xi) or not 0 < xi < 1:
        raise ValueError("xi must lie in (0,1)")
    m, b = assessment["metrics"], assessment["screen"]
    qos = np.asarray(c["requirements"]["qos_public"])[np.asarray(order, int)]
    rates = np.asarray(b["public_rate_lower_sic_order"] if bounded else m["public_mean_rate_bit_s_hz"])
    outage = b["bob_outage_upper"] if bounded else m["bob_outage_estimate"]
    dep = b["dep_uniform_lower"] if bounded else b["dep_empirical_min"]
    return {"source_label_run_feasible": bool(proposal["label_feasible"]),
            "label_gamma_dep_ge_target": bool(proposal["label_metrics"]["dep_min"] >= xi),
            "bank_gamma_dep_ge_target": bool(m["dep_min"] >= xi),
            "dep_ge_target": bool(dep >= xi), "public_qos": bool(np.all(rates >= qos)),
            "bob_outage": bool(outage <= c["requirements"]["outage_bob_max"]),
            "sic_ceiling_margins": bool(m["outage_ceiling_margins_positive"]),
            "frozen_layout_rule": bool(m["constraints"]["checks"]["layout_bob_mean_gain_rule"])}


def verify_validation_records(c, order, proposals, assessments, n, pool_size, alpha):
    """Reconcile saved CI arithmetic and allocation mapping before reuse.

    This checks serialized results, not the original raw fading samples.
    Independence still relies on the recorded source experimental protocol.
    """
    ids = [p["proposal_id"] for p in proposals]
    lookup = {a["proposal_id"]: a for a in assessments}
    if len(ids) != pool_size or len(set(ids)) != len(ids) or len(lookup) != len(assessments) or set(ids) != set(lookup):
        raise ValueError("Source proposal/assessment IDs or pool size do not match")
    expected = family_constants(n, pool_size, c["n_public"], alpha)
    for p in proposals:
        a = lookup[p["proposal_id"]]
        m, b = a["metrics"], a["screen"]
        if m["n_fading"] != n:
            raise ValueError("Source validation sample count does not match its plan")
        for key, value in expected.items():
            if not np.isclose(b["constants"][key], value, rtol=1e-10, atol=1e-14):
                raise ValueError(f"Source validation family budget mismatch: {key}")
        allocation = p["label_metrics"]["allocation_sic_order"]
        if not np.allclose([allocation["mb"], *allocation["public"], allocation["s"]],
                           [m["allocation_sic_order"]["mb"], *m["allocation_sic_order"]["public"], m["allocation_sic_order"]["s"]], rtol=0, atol=1e-12):
            raise ValueError("Source label and validation allocations differ")
        mb, mp = allocation["mb"], np.asarray(allocation["public"])
        if not np.isclose(mb + mp.sum(), 1, rtol=0, atol=1e-12) or np.any(mp < 0):
            raise ValueError("Source allocation violates simplex")
        A = mb + np.cumsum(mp[::-1])[::-1] - mp + c["kappa"] ** 2
        caps = np.log1p(mp / A) / np.log(2)
        rates_lower = np.maximum(0., np.asarray(m["public_mean_rate_bit_s_hz"]) - caps * expected["rate_fraction_subtraction"])
        dep_lower = max(0., b["dep_empirical_min"] - expected["dep_subtraction"])
        outage_upper = binomial_upper(m["bob_outage_count"], n, expected["outage_alpha_per_candidate"])
        if not np.allclose(rates_lower, b["public_rate_lower_sic_order"], rtol=1e-10, atol=1e-12):
            raise ValueError("Source public-rate confidence bounds do not reconcile")
        if not np.isclose(dep_lower, b["dep_uniform_lower"], rtol=1e-10, atol=1e-12):
            raise ValueError("Source DEP confidence bound does not reconcile")
        if not np.isclose(outage_upper, b["bob_outage_upper"], rtol=1e-10, atol=1e-12):
            raise ValueError("Source outage confidence bound does not reconcile")
    return lookup


def select_across_xi(c, order, proposals, assessments, xis):
    lookup = {a["proposal_id"]: a for a in assessments}
    ids = [p["proposal_id"] for p in proposals]
    if len(set(ids)) != len(ids) or len(lookup) != len(assessments) or set(ids) != set(lookup):
        raise ValueError("Duplicate or mismatched source IDs")
    selected = []
    for xi in xis:
        candidates = [p for p in proposals if all(target_checks(c, order, p, lookup[p["proposal_id"]], xi).values())]
        winner = min(candidates, key=lambda p: (-p["label_metrics"]["bob_mean_rate_bit_s_hz"], p["gamma_target"] - xi, p["seed"], p["proposal_id"])) if candidates else None
        selected.append({"xi": float(xi), "proposal_id": winner["proposal_id"] if winner else None,
                         "source_xi": winner["xi"] if winner else None,
                         "source_margin": winner["margin"] if winner else None,
                         "gamma_target": winner["gamma_target"] if winner else None,
                         "effective_gamma_margin": winner["gamma_target"] - xi if winner else None,
                         "feasible_pool_count": len(candidates)})
    # Nested feasibility: lower targets cannot have a lower best label score.
    rates = [(s["xi"], next(p["label_metrics"]["bob_mean_rate_bit_s_hz"] for p in proposals if p["proposal_id"] == s["proposal_id"]))
             for s in selected if s["proposal_id"] is not None]
    rates.sort()
    if any(b[1] > a[1] + 1e-12 for a, b in zip(rates, rates[1:])):
        raise ArithmeticError("Cross-target label scores violate nested feasibility")
    return selected
