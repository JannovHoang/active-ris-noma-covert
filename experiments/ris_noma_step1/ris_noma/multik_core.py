"""Variable-public-user allocation and cached PSO objective for step 7a.

The existing channel generator, scalar RIS noise, power budget, full evaluator
and statistical screen are reused. No files of steps 1--6g are changed.
"""
from copy import deepcopy

import numpy as np

from .config import dbm_to_w, validate_config
from .dep import fit_gamma, minimize_gamma_dep
from .evaluator import evaluate_with_dep
from .metrics import power_budget
from .pso import Outcome
from .validation import statistical_screen


def public_count(k):
    if type(k) is not int or k < 1:
        raise ValueError("n_public must be a positive integer")
    return k


def sic_order(order, k):
    order = np.asarray(order)
    if (order.shape != (k,) or not np.issubdtype(order.dtype, np.integer)
            or not np.array_equal(np.sort(order), np.arange(k))):
        raise ValueError("SIC order must be a permutation of the public IDs")
    return order.astype(int)


# Giải mã K+1 biến PSO thành mb, K phần công suất public và s.
# mb và s là tỷ lệ không có đơn vị; K là số public user, khác hệ số Rician.
def decode_allocation(z, k, bounds):
    """K+1 PSO coordinates: mb, K-1 sorted simplex cuts, s.

    Uniform independent cuts give uniform Dirichlet(1,...,1) public fractions.
    For K=2 this is exactly the previous [mb, public fraction, s] decoder.
    Sorting is used only by PSO, never by the differentiable GNN head.
    """
    public_count(k)
    z = np.asarray(z, float)
    if z.shape != (k + 1,) or not np.all(np.isfinite(z)) or np.any((z < 0) | (z > 1)):
        raise ValueError("Need K+1 finite PSO coordinates in [0,1]")
    # Ánh xạ tuyến tính tọa độ [0,1] về khoảng mb đã khai báo.
    mb = bounds["mb"][0] + np.ptp(bounds["mb"]) * z[0]
    s = bounds["s"][0] + np.ptp(bounds["s"]) * z[-1]
    # K-1 điểm cắt chia ngân sách public thành K đoạn không âm.
    # Tổng các đoạn bằng 1-mb, nên mb + sum(public) = 1.
    public = (1 - mb) * np.diff(np.r_[0., np.sort(z[1:-1]), 1.])
    return float(mb), public, float(s)


# Kiểm tra vector [mb, public theo ID gốc, s]; chỉ sửa sai số làm tròn nhỏ.
# Hàm này không chứng nhận DEP, QoS hoặc outage.
def physical_allocation(values, k, bounds):
    """Correct float32 simplex rounding only; never repair physical failures."""
    public_count(k)
    y = np.asarray(values, float).copy()
    if y.shape != (k + 2,) or not np.all(np.isfinite(y)):
        raise ValueError("Need K+2 finite values [mb, public original IDs, s]")
    for i, name in ((0, "mb"), (-1, "s")):
        lo, hi = bounds[name]
        if not lo - 1e-6 <= y[i] <= hi + 1e-6:
            raise ValueError("Allocation is outside the declared bounds")
        y[i] = np.clip(y[i], lo, hi)
    p = y[1:-1]
    if np.any(p < -1e-6) or abs(y[:-1].sum() - 1.) > 1e-6 or p.sum() <= 0:
        raise ValueError("Allocation is outside the power simplex")
    p = np.maximum(p, 0.)
    y[1:-1] = (1 - y[0]) * p / p.sum()
    return y


# Chuyển public đang theo hạng SIC về ID người dùng gốc và tìm tọa độ PSO
# tương ứng. Nhờ đó nhãn, đầu ra mạng và nghiệm PSO dùng cùng quy ước lưu trữ.
def original_allocation(mb, public, s, order, bounds):
    k = len(public)
    order = sic_order(order, k)
    p = np.asarray(public, float)
    y = np.r_[mb, np.zeros(k), s]
    y[1:-1][order] = p
    y = physical_allocation(y, k, bounds)
    # Cộng dồn tỷ trọng public để khôi phục các điểm cắt của phép giải mã.
    cuts = np.cumsum(y[1:-1][order] / (1 - y[0]))[:-1]
    z = np.r_[(y[0] - bounds["mb"][0]) / np.ptp(bounds["mb"]), cuts,
              (y[-1] - bounds["s"][0]) / np.ptp(bounds["s"])]
    # Canonical cumulative cuts can acquire tiny endpoint rounding.
    if np.any(z < -1e-12) or np.any(z > 1 + 1e-12):
        raise ArithmeticError("Inverse allocation map left the PSO interval")
    z = np.clip(z, 0., 1.)
    mb2, p2, s2 = decode_allocation(z, k, bounds)
    if not np.allclose(np.r_[mb2, p2, s2], np.r_[y[0], y[1:-1][order], y[-1]], atol=1e-12, rtol=0):
        raise ArithmeticError("Variable-K allocation map did not round-trip")
    return y, z


# Tiện ích tương thích cấu hình nhiều K. Quy tắc LoS tường minh ở đây
# là quy tắc cũ; profile hình học được áp dụng riêng trong geometry_los.py.
def extend_config(base, k, overrides=None):
    """Preserve supplied explicit LoS vectors; add deterministic public vectors.

    New public LoS phases are U[0,2pi), sampled once per public ID using a
    separate seeded stream. They stay fixed across layouts and channel banks,
    matching the supplied explicit-phase convention, not a steering model.
    Heterogeneous user requirements must be provided explicitly when extended.
    """
    public_count(k)
    c = deepcopy(base)
    old_k, n = c["n_public"], c["n_ris"]
    overrides = {} if overrides is None else overrides
    allowed = {"qos_public", "target_public", "noise_public"}
    if set(overrides) - allowed:
        raise ValueError("Unknown per-public override")
    for name, container, field in (("qos_public", c["requirements"], "qos_public"),
                                    ("target_public", c["requirements"], "target_public"),
                                    ("noise_public", c["noise_dbm"], "public")):
        values = list(container[field])
        if name in overrides:
            values = list(overrides[name])
            if len(values) != k:
                raise ValueError("Provide one override value per public user")
        elif k <= len(values):
            values = values[:k]
        elif all(v == values[0] for v in values):
            values = [values[0]] * k
        else:
            raise ValueError(f"Supply {name} explicitly for additional heterogeneous users")
        container[field] = values
    if c["los"]["mode"] == "explicit_phases":
        old = np.asarray(c["los"]["ru_phase_rad"], float)
        public = []
        for i in range(k):
            if i < old_k:
                public.append(old[i + 1])
            else:
                random = np.random.default_rng(np.random.SeedSequence([c["root_seed"], 7101, i]))
                public.append(random.uniform(0., 2 * np.pi, n))
        c["los"]["ru_phase_rad"] = np.vstack([old[0], *public, old[-1]]).tolist()
    c["n_public"] = k
    reference = c["reference_allocation"]
    if k != old_k:
        reference["public"] = [(1 - reference["mb"]) / k] * k
    return validate_config(c)


class MultiKObjective:
    """Cached version of the unchanged full evaluator, now for K>=1."""
    def __init__(self, c, layout, bank, order):
        self.k = public_count(c["n_public"])
        self.order = sic_order(order, self.k)
        self.c, self.layout = c, layout
        # Chỉ lấy Bob và public, sắp public theo SIC; Willie được xử lý riêng cho DEP.
        columns = np.r_[0, self.order + 1]
        self.g, self.q = bank.signal_gain_tilde[:, columns], bank.noise_gain_tilde[:, columns]
        # Đổi nhiễu thu từ dBm sang watt trước khi cộng và chia trong SINR.
        self.receiver_noise = dbm_to_w(np.r_[c["noise_dbm"]["bob"], np.asarray(c["noise_dbm"]["public"])[self.order]])
        self.sigma_r2, self.sigma_w2 = float(dbm_to_w(c["noise_dbm"]["ris"])), float(dbm_to_w(c["noise_dbm"]["willie"]))
        self.k2 = c["kappa"] ** 2
        req = c["requirements"]
        self.qos = np.asarray(req["qos_public"])[self.order]
        # Đổi mục tiêu rate R (bit/s/Hz) thành ngưỡng SINR Gamma = 2**R - 1.
        self.target = np.expm1(np.log(2) * np.asarray(req["target_public"])[self.order])
        self.target_b = float(np.expm1(np.log(2) * req["target_bob"]))
        self.outage_limit = req["outage_bob_max"]
        x, y = np.abs(bank.h_tilde[:, -1]) ** 2, np.abs(bank.g_scalar_tilde[:, -1]) ** 2
        # Cache moment tín hiệu/nhiễu tại Willie để mỗi lần chấm PSO không phải
        # sinh lại kênh. Bank tìm kiếm được giữ cố định trong cả lượt tối ưu.
        self.xy_mean = np.array([x.mean(), y.mean()])
        self.xy_var = np.array([x.var(ddof=c.get("moment_ddof", 1)), y.var(ddof=c.get("moment_ddof", 1))])
        means = bank.signal_gain_tilde.mean(axis=0)
        # Kiểm tra giả định Bob có độ lợi trung bình lớn hơn mọi public user.
        self.layout_eligible = bool(means[0] > np.max(means[1:-1]))

    def evaluate(self, z, xi):
        if not np.isfinite(xi) or not 0 < xi < 1:
            raise ValueError("DEP target must be strictly between zero and one")
        mb, mp, s = decode_allocation(z, self.k, self.c["bounds"])
        budget = power_budget(self.c, self.layout.loss_ar, s)
        pa, rho2 = budget["pa_w"], budget["rho2"]
        # U là chất lượng kênh chuẩn hóa chưa chứa phân bổ NOMA và méo phần cứng.
        # Viết nhiễu thu/rho2 tương đương chia cả tử và mẫu SINR cho rho2.
        U = pa * self.g / (self.sigma_r2 * self.q + self.receiver_noise[None, :] / rho2)
        # A_j gồm mb, các public stream giải mã sau j và kappa**2.
        A = mb + np.cumsum(mp[::-1])[::-1] - mp + self.k2
        # Rate trung bình Bob sau SIC lý tưởng; méo phần cứng và nhiễu vẫn còn.
        rb = float(np.log2(1 + mb * U[:, 0] / (self.k2 * U[:, 0] + 1)).mean())
        # Mỗi public stream phải được giải mã tại user đích và mọi decoder SIC
        # cần thiết. Dùng decoder yếu nhất, đồng thời tính cả Bob ở dòng kế tiếp.
        weakest_public = np.minimum.accumulate(U[:, 1:][:, ::-1], axis=1)[:, ::-1]
        weakest = np.minimum(U[:, :1], weakest_public)
        rates = np.log2(1 + mp[None, :] * weakest / (A[None, :] * weakest + 1)).mean(axis=0)
        # Độ dư d_j và d_b phải dương thì mục tiêu giải mã mới có thể đạt.
        # Khi dương, toàn chuỗi SIC tại Bob quy về một ngưỡng duy nhất trên U_b.
        margins, margin_b = mp - self.target * A, mb - self.target_b * self.k2
        margin_ok = bool(np.all(margins > 0) and margin_b > 0)
        outage = (float(np.mean(U[:, 0] < max(float(np.max(self.target / margins)), self.target_b / margin_b)))
                  if margin_ok else 1.)
        a1 = (1 + self.k2) * pa / (self.layout.loss_ar * self.layout.loss_ru[-1])
        b = self.sigma_r2 / self.layout.loss_ru[-1]
        # Moment matching Gamma dùng phương sai xấp xỉ bỏ covariance X,Y,
        # đúng quy ước bộ chấm tìm kiếm; audit thực nghiệm được thực hiện riêng.
        laws = [fit_gamma(float(rho2 * (ai * self.xy_mean[0] + b * self.xy_mean[1])),
                          float(rho2 ** 2 * (ai ** 2 * self.xy_var[0] + b ** 2 * self.xy_var[1])))
                for ai in ((1 - mb) * a1, a1)]
        dep = minimize_gamma_dep(*laws)
        # Chuẩn hóa mức thiếu hụt để PSO so sánh các nghiệm chưa khả thi.
        # Một rate cao không đủ để phương án được coi là đạt.
        violations = {"covertness": float(max(0., xi - dep["dep_min"]) / xi),
                      "qos": float(np.maximum(0., self.qos - rates).dot(1 / self.qos)),
                      "outage": float(max(0., outage - self.outage_limit) / self.outage_limit),
                      "layout": 0. if self.layout_eligible else 1.}
        feasible = bool(dep["dep_min"] >= xi and np.all(rates >= self.qos)
                        and outage <= self.outage_limit and margin_ok and self.layout_eligible)
        violation = float(sum(violations.values()))
        if feasible != (violation == 0):
            raise ArithmeticError("Variable-K constraint flags differ from normalized violations")
        return Outcome(rb, violation, feasible, {
            "allocation_sic_order": {"mb": mb, "public": mp.tolist(), "s": s},
            "bob_mean_rate_bit_s_hz": rb, "public_mean_rate_bit_s_hz": rates.tolist(),
            "dep_min": dep["dep_min"], "lambda_star_w": self.sigma_w2 + dep["shift_star_w"],
            "bob_outage_estimate": outage, "outage_margins_public": margins.tolist(),
            "outage_margin_bob": float(margin_b), "normalized_violations": violations,
            "feasible_under_gamma_sample_model": feasible})


def assess_allocation(c, layout, bank, order, values, xi, family_size, alpha):
    k = c["n_public"]
    order = sic_order(order, k)
    y = physical_allocation(values, k, c["bounds"])
    summary, samples, _ = evaluate_with_dep(c, layout, bank, order, y[0], y[1:-1][order], y[-1], xi)
    return summary, statistical_screen(c, summary, samples, order, xi, family_size, alpha)
