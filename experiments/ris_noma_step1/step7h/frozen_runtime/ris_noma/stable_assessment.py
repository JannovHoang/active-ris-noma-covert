"""7d evaluator: unchanged physics with a forward-roundoff identity check.

The legacy evaluator's relative-only check of (V1-V0)=delta can reject valid
very weak Willie signals. All metric arithmetic and feasibility thresholds are
preserved; only that internal algebraic consistency check uses a machine-epsilon
bound scaled by the powers being subtracted. Legacy source files are untouched.
"""
import numpy as np
from .config import dbm_to_w
from .metrics import power_budget
from .dep import fit_willie, minimize_gamma_dep
from .evaluator import constraint_report
from .multik_core import physical_allocation, sic_order
from .validation import statistical_screen


def assert_power_identity(v0, v1, delta):
    v0, v1, delta = (np.asarray(value, float) for value in (v0, v1, delta))
    if not all(np.all(np.isfinite(value)) for value in (v0, v1, delta)) or np.any(v0 < 0) or np.any(delta < 0):
        raise ArithmeticError("Invalid Willie power identity inputs")
    error = np.abs((v1 - v0) - delta)
    bound = 32 * np.finfo(float).eps * (np.abs(v0) + np.abs(v1) + np.abs(delta))
    if np.any(v1 < v0 - bound) or np.any(error > bound):
        raise ArithmeticError("Willie powers violate the identity beyond floating-point roundoff")


def evaluate_candidate(c, layout, bank, order, mb, mp, s):
    K, M = c["n_public"], c["n_fading"]
    mp = np.asarray(mp, float)  # mp[j] ứng với SIC rank j+1.
    if mp.shape != (K,) or not np.all(np.isfinite(mp)) or np.any(mp < 0) or not np.isfinite(mb) or mb < 0 or not np.isclose(mb + mp.sum(), 1, rtol=0, atol=1e-12):
        raise ValueError("Candidate vi phạm simplex")
    if not c["bounds"]["mb"][0] <= mb <= c["bounds"]["mb"][1]:
        raise ValueError("mb nằm ngoài bounds đã công bố")
    order = np.asarray(order, int)
    if order.shape != (K,) or not np.array_equal(np.sort(order), np.arange(K)):
        raise ValueError("order phải là hoán vị public user IDs")
    # Thứ tự receiver nội bộ từ đây: [Bob,C1,...,CK,Willie].
    columns = np.r_[0, 1 + order, K + 1]
    budget = power_budget(c, layout.loss_ar, s)
    pa, rho2 = budget["pa_w"], budget["rho2"]
    g = rho2 * bank.signal_gain_tilde[:, columns]
    q = rho2 * bank.noise_gain_tilde[:, columns]
    noise_dbm = c["noise_dbm"]
    receiver_noise = dbm_to_w(np.r_[noise_dbm["bob"], np.asarray(noise_dbm["public"])[order], noise_dbm["willie"]])
    sigma_r2 = float(dbm_to_w(noise_dbm["ris"]))
    U = pa * g / (sigma_r2 * q + receiver_noise[None, :])
    k2 = c["kappa"] ** 2
    A = mb + (np.cumsum(mp[::-1])[::-1] - mp) + k2
    # gamma[:,receiver,j]; chỉ các decoder cần thiết được dùng trong rate/outage.
    gamma = mp[None, None, :] * U[:, :, None] / (A[None, None, :] * U[:, :, None] + 1)
    gamma_b = mb * U[:, 0] / (k2 * U[:, 0] + 1)
    rb = np.log2(1 + gamma_b)
    public_rates = np.empty((M, K))
    for j in range(K):
        decoders = np.r_[0, np.arange(j + 1, K + 1)]
        public_rates[:, j] = np.log2(1 + gamma[:, decoders, j]).min(axis=1)

    req = c["requirements"]
    targets = np.asarray(req["target_public"])[order]
    thresholds = np.expm1(np.log(2) * targets)
    threshold_b = float(np.expm1(np.log(2) * req["target_bob"]))
    d_public = mp - thresholds * A
    d_bob = mb - threshold_b * k2
    failure_b = np.any(gamma[:, 0, :] < thresholds[None, :], axis=1) | (gamma_b < threshold_b)
    if np.any(d_public <= 0) or d_bob <= 0:
        bound_b = np.inf
    else:
        bound_b = float(max(np.max(thresholds / d_public), threshold_b / d_bob))
    failure_b_threshold = U[:, 0] < bound_b
    if not np.array_equal(failure_b, failure_b_threshold):
        raise ArithmeticError("Outage theo chuỗi SIC và ngưỡng tương đương không khớp")
    outage_public = np.array([np.any(gamma[:, i, :i] < thresholds[None, :i], axis=1).mean() for i in range(1, K + 1)])
    qos_sorted = np.asarray(req["qos_public"])[order]
    public_mean = public_rates.mean(axis=0)
    p_out = float(failure_b.mean())

    # (9.6)-(9.10): public powers và rho giữ nguyên dưới H0,H1.
    v0 = (1 + k2) * (1 - mb) * pa * g[:, -1] + sigma_r2 * q[:, -1] + receiver_noise[-1]
    v1 = (1 + k2) * pa * g[:, -1] + sigma_r2 * q[:, -1] + receiver_noise[-1]
    expected_delta = (1 + k2) * mb * pa * g[:, -1]
    assert_power_identity(v0, v1, expected_delta)
    xw = rho2 * np.abs(bank.h_tilde[:, -1]) ** 2
    yw = rho2 * np.abs(bank.g_scalar_tilde[:, -1]) ** 2
    sample_covariance = float(np.cov(xw, yw, ddof=c.get("moment_ddof", 1))[0, 1])
    # Đúng quy ước trung bình của (4.5), không thêm kappa vào budget.
    ris_power_samples = rho2 * (pa / layout.loss_ar * np.sum(np.abs(bank.h_ar) ** 2, axis=1) + sigma_r2 * c["n_ris"])
    means = bank.signal_gain_tilde.mean(axis=0)
    bob_strongest = bool(means[0] > np.max(means[1:-1]))
    margin_ok = bool(np.all(d_public > 0) and d_bob > 0)
    summary = {
        "allocation_sic_order": {"mb": float(mb), "public": mp.tolist(), "s": float(s)},
        "power": budget,
        "bob_mean_rate_bit_s_hz": float(rb.mean()),
        "public_mean_rate_bit_s_hz": public_mean.tolist(),
        "bob_outage_estimate": p_out,
        "bob_outage_count": int(failure_b.sum()),
        "bob_outage_zero_events_upper95_one_sided": float(-np.expm1(np.log(.05) / M)) if not np.any(failure_b) else None,
        "n_fading": M,
        "public_outage_estimate": outage_public.tolist(),
        "bob_fixed_rate_goodput_bit_s_hz": float(req["target_bob"] * (1 - p_out)),
        "outage_margin_bob": float(d_bob),
        "outage_margins_public": d_public.tolist(),
        "bob_outage_threshold_U": None if not np.isfinite(bound_b) else bound_b,
        "qos_pass_per_public": (public_mean >= qos_sorted).tolist(),
        "bob_outage_point_estimate_pass": bool(p_out <= req["outage_bob_max"]),
        "outage_ceiling_margins_positive": margin_ok,
        "bob_strongest_mean_gain_this_bank": bob_strongest,
        "ris_output_budget_ratio_sample_mean": float(ris_power_samples.mean() / budget["pris_w"]),
        "ris_output_budget_ratio_sample_max": float(ris_power_samples.max() / budget["pris_w"]),
        "willie_mean_v0_w": float(v0.mean()), "willie_mean_v1_w": float(v1.mean()),
        "willie_X_Y_sample_covariance": sample_covariance,
        "dep_min": None, "full_feasibility": "not_evaluated_dep_missing",
    }
    samples = {"U": U, "gamma_public": gamma, "gamma_bob": gamma_b,
               "bob_rate": rb, "public_rates": public_rates, "bob_outage": failure_b,
               "v0_willie": v0, "v1_willie": v1, "xw": xw, "yw": yw}
    return summary, samples


def evaluate_with_dep(c, layout, bank, order, mb, mp, s, xi, layout_eligible=None):
    summary, samples = evaluate_candidate(c, layout, bank, order, mb, mp, s)
    h0, h1, z0, z1, moments = fit_willie(
        samples["xw"], samples["yw"], summary["power"]["pa_w"], mb, c["kappa"],
        layout.loss_ar, float(layout.loss_ru[-1]), float(dbm_to_w(c["noise_dbm"]["ris"])),
        ddof=c.get("moment_ddof", 1))
    optimum = minimize_gamma_dep(h0, h1)
    sigma_w2 = float(dbm_to_w(c["noise_dbm"]["willie"]))
    optimum["lambda_star_w"] = sigma_w2 + optimum["shift_star_w"]
    summary["dep_min"] = optimum["dep_min"]
    summary["lambda_star_w"] = optimum["lambda_star_w"]
    summary["dep_model"] = moments["variance_model"]
    summary["constraints"] = constraint_report(summary, xi, layout_eligible)
    summary["full_feasibility"] = "sample_approx_feasible" if summary["constraints"]["feasible_under_gamma_sample_model"] else "sample_approx_infeasible"
    samples = {**samples, "z0_willie": z0, "z1_willie": z1}
    return summary, samples, {"h0": h0, "h1": h1, "moments": moments, "gamma_optimum": optimum}


def assess_allocation(c, layout, bank, order, values, xi, family_size, alpha):
    k = c["n_public"]
    order = sic_order(order, k)
    y = physical_allocation(values, k, c["bounds"])
    summary, samples, _ = evaluate_with_dep(c, layout, bank, order, y[0], y[1:-1][order], y[-1], xi)
    return summary, statistical_screen(c, summary, samples, order, xi, family_size, alpha)


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
