"""Independent finite-pool validation for the existing K=2 physics model.

All PSO proposals must be fixed before seeing this bank. Confidence bounds use
a union bound over the declared pool. They are conditional on the fixed layout,
LoS and i.i.d. fading model; they do not prove global optimality or real-world
validity of that model. H0/H1 may be paired and proposals may share one bank.
"""
import math
import numpy as np
from scipy.stats import beta

from .dep import minimize_empirical_dep
from .evaluator import evaluate_with_dep


FRESH_STREAM = {"validation": 6, "heldout": 7}


def fresh_rng(root_seed, layout_id, role, seed):
    """New streams; the earlier step-3 evaluation stream (2) is not reused."""
    if role not in FRESH_STREAM:
        raise ValueError("role must be validation or heldout")
    fields = [root_seed, layout_id, FRESH_STREAM[role], seed]
    if any(type(x) is not int or x < 0 for x in fields):
        raise ValueError("RNG entropy fields must be nonnegative integers")
    return np.random.default_rng(np.random.SeedSequence(fields))


def family_constants(n, pool_size, n_public, alpha):
    """Split alpha equally among DEP, public QoS, and Bob outage families.

    DKW: each of 2*J CDFs gets alpha/(6*J).
    Hoeffding: each of J*K bounded rate means gets alpha/(3*J*K).
    Clopper-Pearson: each of J outage proportions gets alpha/(3*J).
    """
    for value in [n, pool_size, n_public]:
        if type(value) is not int or value < 1:
            raise ValueError("n, pool_size and n_public must be positive integers")
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must lie in (0,1)")
    epsilon = math.sqrt(math.log(12 * pool_size / alpha) / (2 * n))
    return {"n": n, "pool_size": pool_size, "alpha_family": float(alpha),
            "alpha_each_metric_family": float(alpha / 3),
            "cdf_epsilon_each": epsilon, "dep_subtraction": 2 * epsilon,
            "rate_fraction_subtraction": math.sqrt(math.log(3 * pool_size * n_public / alpha) / (2 * n)),
            "outage_alpha_per_candidate": float(alpha / (3 * pool_size))}


def binomial_upper(failures, n, alpha):
    """One-sided exact Clopper-Pearson upper confidence limit."""
    if type(n) is not int or n < 1 or type(failures) is not int or not 0 <= failures <= n:
        raise ValueError("Need integer counts: 0 <= failures <= n, n >= 1")
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must lie in (0,1)")
    if failures == n:
        return 1.0
    if failures == 0:
        return float(-np.expm1(math.log(alpha) / n))
    return float(beta.isf(alpha, failures + 1, n - failures))


def statistical_screen(c, summary, samples, order, xi, pool_size, alpha):
    """Simultaneous bounds for a label-generated, validation-independent pool."""
    K = c["n_public"]
    rates = np.asarray(samples["public_rates"], float)
    failures = np.asarray(samples["bob_outage"])
    n = len(failures)
    constants = family_constants(n, pool_size, K, alpha)
    if rates.shape != (n, K) or failures.dtype != bool or not np.all(np.isfinite(rates)):
        raise ValueError("Invalid rate samples or outage indicators")
    allocation = summary["allocation_sic_order"]
    mb, mp = allocation["mb"], np.asarray(allocation["public"], float)
    A = mb + np.cumsum(mp[::-1])[::-1] - mp + c["kappa"] ** 2
    # gamma_j = mp_j*U/(A_j*U+1) <= mp_j/A_j, for every required decoder.
    rate_caps = np.log1p(mp / A) / np.log(2)
    if np.any(rates < -1e-12) or np.any(rates > rate_caps[None, :] + 1e-10):
        raise ArithmeticError("Public rate samples exceed the theoretical SIC ceilings")
    lower_rates = np.maximum(0., rates.mean(axis=0) - rate_caps * constants["rate_fraction_subtraction"])
    qos = np.asarray(c["requirements"]["qos_public"])[np.asarray(order, int)]
    empirical = minimize_empirical_dep(samples["z0_willie"], samples["z1_willie"])
    dep_lower = float(max(0., empirical["dep"] - constants["dep_subtraction"]))
    outage_upper = binomial_upper(int(failures.sum()), n, constants["outage_alpha_per_candidate"])
    checks = {
        "dep_uniform_lower_ge_xi": bool(dep_lower >= xi),
        "public_mean_rate_lower_ge_qos": bool(np.all(lower_rates >= qos)),
        "bob_outage_upper_le_limit": bool(outage_upper <= c["requirements"]["outage_bob_max"]),
        "positive_sic_ceiling_margins": bool(summary["outage_ceiling_margins_positive"]),
        "gamma_sample_model_checks": bool(summary["constraints"]["feasible_under_gamma_sample_model"]),
    }
    return {"accepted": bool(all(checks.values())), "checks": checks,
            "dep_empirical_min": float(empirical["dep"]), "dep_empirical_threshold_shift_w": empirical["shift_w"],
            "dep_uniform_lower": dep_lower, "dep_lower_margin": dep_lower - xi,
            "public_rate_lower_sic_order": lower_rates.tolist(), "public_rate_ceilings_sic_order": rate_caps.tolist(),
            "bob_outage_upper": outage_upper, "constants": constants,
            "scope": "finite proposal family, fixed layout/LoS, i.i.d. fading under the selected simulator; no global-optimality or real-world certificate"}


def assess_proposal(c, layout, bank, order, proposal, pool_size, alpha, eligible):
    a = proposal["label_metrics"]["allocation_sic_order"]
    summary, samples, _ = evaluate_with_dep(c, layout, bank, order, a["mb"], a["public"], a["s"], proposal["xi"], eligible)
    return {"proposal_id": proposal["proposal_id"], "metrics": summary,
            "screen": statistical_screen(c, summary, samples, order, proposal["xi"], pool_size, alpha)}


def select_on_validation(proposals, assessments, xis):
    """No heldout argument: select before the final bank is generated or read."""
    lookup = {r["proposal_id"]: r for r in assessments}
    if len(lookup) != len(assessments):
        raise ValueError("Duplicate validation proposal IDs")
    ids = [p["proposal_id"] for p in proposals]
    if len(set(ids)) != len(ids) or set(ids) != set(lookup):
        raise ValueError("Every proposal must have exactly one matching validation assessment")
    selected = []
    for xi in xis:
        candidates = [p for p in proposals if p["xi"] == xi and p["label_feasible"] and lookup[p["proposal_id"]]["screen"]["accepted"]]
        winner = min(candidates, key=lambda p: (-p["label_metrics"]["bob_mean_rate_bit_s_hz"], p["margin"], p["seed"], p["proposal_id"])) if candidates else None
        selected.append({"xi": float(xi), "proposal_id": winner["proposal_id"] if winner else None,
                         "status": "selected_on_validation" if winner else "no_candidate_passed_validation"})
    return selected
