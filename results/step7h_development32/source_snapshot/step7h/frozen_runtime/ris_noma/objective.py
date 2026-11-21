"""Evaluator tiền tính cho PSO K=2; đối chiếu được với evaluator bước 2.

Bank và thứ tự SIC được cố định. Không dùng bank evaluation để tìm nghiệm.
"""
import numpy as np
from .config import dbm_to_w
from .metrics import power_budget
from .dep import fit_gamma, minimize_gamma_dep
from .pso import Outcome


def decode_k2(z, bounds):
    z = np.asarray(z, float)
    if z.shape != (3,) or not np.all(np.isfinite(z)) or np.any((z < 0) | (z > 1)):
        raise ValueError("Particle K=2 cần ba số trong [0,1]")
    mb = bounds["mb"][0] + (bounds["mb"][1] - bounds["mb"][0]) * z[0]
    public = (1 - mb) * np.array([z[1], 1 - z[1]])
    s = bounds["s"][0] + (bounds["s"][1] - bounds["s"][0]) * z[2]
    return float(mb), public, float(s)


class CachedObjective:
    def __init__(self, c, layout, bank, order):
        if c["n_public"] != 2:
            raise ValueError("PSO bước 3 hiện hỗ trợ K=2")
        self.c, self.layout = c, layout
        self.order = np.asarray(order, int)
        if self.order.shape != (2,) or not np.array_equal(np.sort(self.order), [0, 1]):
            raise ValueError("order phải là hoán vị hai public IDs")
        columns = np.r_[0, self.order + 1]
        self.g = bank.signal_gain_tilde[:, columns]
        self.q = bank.noise_gain_tilde[:, columns]
        noise = c["noise_dbm"]
        self.receiver_noise = dbm_to_w(np.r_[noise["bob"], np.asarray(noise["public"])[self.order]])
        self.sigma_r2 = float(dbm_to_w(noise["ris"]))
        self.sigma_w2 = float(dbm_to_w(noise["willie"]))
        self.k2 = c["kappa"] ** 2
        req = c["requirements"]
        self.qos = np.asarray(req["qos_public"])[self.order]
        self.target = np.expm1(np.log(2) * np.asarray(req["target_public"])[self.order])
        self.target_b = float(np.expm1(np.log(2) * req["target_bob"]))
        self.outage_limit = req["outage_bob_max"]
        x = np.abs(bank.h_tilde[:, -1]) ** 2
        y = np.abs(bank.g_scalar_tilde[:, -1]) ** 2
        self.xy_mean = np.array([x.mean(), y.mean()])
        ddof = c.get("moment_ddof", 1)
        self.xy_var = np.array([x.var(ddof=ddof), y.var(ddof=ddof)])
        means = bank.signal_gain_tilde.mean(axis=0)
        self.layout_eligible = bool(means[0] > np.max(means[1:-1]))

    def evaluate(self, z, xi):
        if not np.isfinite(xi) or not 0 < xi < 1:
            raise ValueError("xi phải trong (0,1)")
        mb, mp, s = decode_k2(z, self.c["bounds"])
        budget = power_budget(self.c, self.layout.loss_ar, s)
        pa, rho2 = budget["pa_w"], budget["rho2"]
        # Chia cả tử/mẫu cho rho², cùng Ui của bước 2.
        U = pa * self.g / (self.sigma_r2 * self.q + self.receiver_noise[None, :] / rho2)
        A = np.array([mb + mp[1] + self.k2, mb + self.k2])
        gamma_b = mb * U[:, 0] / (self.k2 * U[:, 0] + 1)
        rb = float(np.log2(1 + gamma_b).mean())
        # SINR tăng theo Ui: lấy min Ui trên đúng tập decoder rồi tính rate.
        weakest = np.column_stack([U.min(axis=1), U[:, [0, 2]].min(axis=1)])
        public_rates = np.log2(1 + mp[None, :] * weakest / (A[None, :] * weakest + 1)).mean(axis=0)
        margins = mp - self.target * A
        margin_b = mb - self.target_b * self.k2
        margin_ok = bool(np.all(margins > 0) and margin_b > 0)
        if margin_ok:
            threshold = float(max(np.max(self.target / margins), self.target_b / margin_b))
            outage = float(np.mean(U[:, 0] < threshold))
        else:
            outage = 1.0
        a1 = (1 + self.k2) * pa / (self.layout.loss_ar * self.layout.loss_ru[-1])
        a0, b = (1 - mb) * a1, self.sigma_r2 / self.layout.loss_ru[-1]
        laws = []
        for ai in [a0, a1]:
            mu = float(rho2 * (ai * self.xy_mean[0] + b * self.xy_mean[1]))
            variance = float(rho2 ** 2 * (ai ** 2 * self.xy_var[0] + b ** 2 * self.xy_var[1]))
            laws.append(fit_gamma(mu, variance))
        dep_result = minimize_gamma_dep(*laws)
        dep = dep_result["dep_min"]
        violations = {
            "covertness": float(max(0, xi - dep) / xi),
            "qos": float(np.maximum(0, self.qos - public_rates).dot(1 / self.qos)),
            "outage": float(max(0, outage - self.outage_limit) / self.outage_limit),
            "layout": 0.0 if self.layout_eligible else 1.0,
        }
        violation = float(sum(violations.values()))
        feasible = bool(dep >= xi and np.all(public_rates >= self.qos)
                        and outage <= self.outage_limit and margin_ok and self.layout_eligible)
        if not feasible and violation == 0:
            # Với target dương và outage_limit<1, margin không dương phải gây outage=1.
            raise ArithmeticError("Constraint flags không thống nhất với vi phạm chuẩn hóa")
        metrics = {"allocation_sic_order": {"mb": mb, "public": mp.tolist(), "s": s},
                   "bob_mean_rate_bit_s_hz": rb, "public_mean_rate_bit_s_hz": public_rates.tolist(),
                   "dep_min": dep, "lambda_star_w": self.sigma_w2 + dep_result["shift_star_w"],
                   "bob_outage_estimate": outage, "outage_margins_public": margins.tolist(),
                   "outage_margin_bob": float(margin_b), "normalized_violations": violations,
                   "feasible_under_gamma_sample_model": feasible}
        return Outcome(rb, violation, feasible, metrics)
