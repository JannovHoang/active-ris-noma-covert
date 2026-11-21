"""Một candidate cố định: công suất, SINR, rate, outage, công suất Willie.

Chưa tính DEP, chưa kết luận candidate khả thi cho bài toán covert đầy đủ.
"""
import numpy as np
from .config import dbm_to_w


def power_budget(c, loss_ar, s):
    if not c["bounds"]["s"][0] <= s <= c["bounds"]["s"][1]:
        raise ValueError("s nằm ngoài bounds")
    n = c["n_ris"]
    p0 = float(dbm_to_w(c["total_dbm"]) - n * sum(c["circuit_w"].values()))
    pa, pris = (1 - s) * p0, s * p0
    rho2 = pris / (n * (pa / loss_ar + dbm_to_w(c["noise_dbm"]["ris"])))
    return {"p0_w": p0, "pa_w": pa, "pris_w": pris, "rho2": float(rho2), "rho": float(np.sqrt(rho2))}


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
    if np.any(v1 < v0) or not np.allclose(v1 - v0, expected_delta, rtol=1e-9, atol=0):
        raise ArithmeticError("Công suất Willie vi phạm (9.10)")
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
