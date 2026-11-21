"""DEP của detector công suất trung bình, theo mục 9-11 derivation v4.

Baseline Gamma giữ phương sai không covariance (10.5). Monte Carlo dùng nguyên
các cặp mẫu Rician để chẩn đoán xấp xỉ; không đổi mô hình nhiễu scalar RIS.
Chỉ số Python 0/1 tương ứng H0/H1, tức Gamma delta_1/delta_2 trong v4.
"""
from dataclasses import asdict, dataclass
import math
import numpy as np
from scipy.optimize import brentq
from scipy.special import gammainc, gammaincc, gammaln, lambertw
from scipy.stats import gamma as scipy_gamma


@dataclass(frozen=True)
# Phân phối Gamma xấp xỉ công suất Z: shape không có đơn vị, scale_w là watt.
class GammaLaw:
    shape: float
    scale_w: float

    def __post_init__(self):
        if not all(np.isfinite(v) and v > 0 for v in [self.shape, self.scale_w, self.mean_w]):
            raise ValueError("Gamma shape, scale, mean phải hữu hạn và dương")

    @property
    def mean_w(self):
        return self.shape * self.scale_w


# Moment matching: shape=mu**2/variance và scale=variance/mu.
# Không thay phân phối fading gốc; chỉ xấp xỉ biến công suất Willie để tìm nhanh.
def fit_gamma(mean_w, variance_w2):
    """Gamma(shape=mu²/v, scale=v/mu). Không sửa variance=0 bằng số epsilon."""
    if not all(np.isfinite(v) and v > 0 for v in [mean_w, variance_w2]):
        raise ValueError("Moment Gamma phải hữu hạn và dương; variance=0 cần mô hình riêng")
    return GammaLaw(float((mean_w / np.sqrt(variance_w2)) ** 2), float(variance_w2 / mean_w))


def _paired_samples(z0, z1):
    z0, z1 = np.asarray(z0, float), np.asarray(z1, float)
    if z0.ndim != 1 or z0.shape != z1.shape or len(z0) < 2:
        raise ValueError("Cần hai vector mẫu ghép cặp cùng chiều, M>=2")
    if not np.all(np.isfinite(z0)) or not np.all(np.isfinite(z1)) or np.any(z0 < 0) or np.any(z1 < 0):
        raise ValueError("Công suất mẫu phải hữu hạn, không âm")
    return z0, z1


def fit_willie(xw, yw, pa_w, mb, kappa, loss_ar, loss_rw, sigma_r2_w, ddof=1):
    """Xw,Yw đã chứa rho² từ evaluate_candidate của bước 1.

    Trả Gamma baseline và Z0/Z1 KHÔNG chứa hằng sigma_w².
    ddof dùng thống nhất cho mọi variance/covariance mẫu; Octave gốc var(X,1)
    tương ứng ddof=0, còn demo bước 1-3 mặc định ddof=1.
    """
    xw, yw = _paired_samples(xw, yw)
    if type(ddof) is not int or ddof not in [0, 1]:
        raise ValueError("ddof phải là 0 hoặc 1")
    if not 0 <= mb < 1 or not np.isfinite(kappa) or kappa < 0:
        raise ValueError("Cần 0<=mb<1 và kappa>=0")
    if not all(np.isfinite(v) and v > 0 for v in [pa_w, loss_ar, loss_rw, sigma_r2_w]):
        raise ValueError("Công suất và mẫu số suy hao phải dương, hữu hạn")
    a1 = (1 + kappa ** 2) * pa_w / (loss_ar * loss_rw)
    # H0 thiếu phần stream Bob; công suất public và hệ số khuếch đại được giữ
    # như trong H1, theo mô hình hai giả thuyết đã khai báo.
    a0 = (1 - mb) * a1
    b = sigma_r2_w / loss_rw
    means = np.array([xw.mean(), yw.mean()])
    variances = np.array([xw.var(ddof=ddof), yw.var(ddof=ddof)])
    # X,Y dùng chung fading/pha nên có thể tương quan. Covariance được lưu
    # để chẩn đoán, dù phương sai Gamma tìm kiếm phía dưới bỏ số hạng này.
    covariance = float(np.cov(xw, yw, ddof=ddof)[0, 1])
    laws, info, z = [], [], []
    for a in [a0, a1]:
        zi = a * xw + b * yw
        mu = float(a * means[0] + b * means[1])
        # Đây là phương sai xấp xỉ đã chọn; phương sai đầy đủ còn có 2*a*b*Cov(X,Y).
        var_base = float(a ** 2 * variances[0] + b ** 2 * variances[1])
        law = fit_gamma(mu, var_base)
        laws.append(law)
        info.append({"mean_w": mu, "variance_baseline_w2": var_base,
                     "variance_paired_samples_w2": float(zi.var(ddof=ddof)),
                     "omitted_covariance_term_w2": float(2 * a * b * covariance),
                     **asdict(law)})
        z.append(zi)
    if np.any(z[1] < z[0]):
        raise ArithmeticError("Công suất Rician phải thỏa Z1>=Z0")
    diagnostics = {"variance_model": "baseline_without_covariance_eq_10_5",
                   "sample_variance_ddof": ddof, "n_fading": len(xw),
                   "coefficients": {"a_h0": a0, "a_h1": a1, "b": b},
                   "X_mean": float(means[0]), "Y_mean": float(means[1]),
                   "X_variance": float(variances[0]), "Y_variance": float(variances[1]),
                   "XY_covariance": covariance, "H0": info[0], "H1": info[1]}
    return laws[0], laws[1], z[0], z[1], diagnostics


# DEP dùng trong dự án là P_FA + P_MD, nằm trong [0,1] sau tối ưu ngưỡng.
# Xác suất lỗi với hai tiên nghiệm bằng nhau mới là DEP/2.
def gamma_dep(shift_w, h0, h1):
    """x=lambda-sigma_w². Trả PFA+PMD; không chia 2 ở đây."""
    x = np.asarray(shift_w, float)
    if np.any(np.isnan(x)):
        raise ValueError("Ngưỡng không được là NaN")
    positive = np.maximum(x, 0)
    # Báo động giả: dưới H0 nhưng công suất vượt ngưỡng, nên dùng đuôi trên CDF.
    pfa = gammaincc(h0.shape, positive / h0.scale_w)
    # Bỏ sót: dưới H1 nhưng công suất không vượt ngưỡng, nên dùng CDF H1.
    pmd = gammainc(h1.shape, positive / h1.scale_w)
    return np.where(x <= 0, 1.0, pfa + pmd)


def _log_equation(h0, h1):
    pref = max(h0.mean_w, h1.mean_w)
    t0, t1 = h0.scale_w / pref, h1.scale_w / pref
    delta = h1.shape - h0.shape
    c = 1 / t0 - 1 / t1
    # Viết lại để giảm cancellation của các log scale gần nhau.
    relative_change = (t1 - t0) / t0
    log_ratio = math.log1p(relative_change) if abs(relative_change) < .5 else math.log(t1) - math.log(t0)
    log_c = float(gammaln(h1.shape) - gammaln(h0.shape) + delta * math.log(t0) + h1.shape * log_ratio)
    return pref, delta, c, log_c


# Đạo hàm DEP theo ngưỡng bằng f1-f0. Mọi giao điểm PDF dương đều
# là ứng viên cực trị; giao điểm đầu tiên chưa chắc cho DEP nhỏ nhất.
def stationary_thresholds(h0, h1):
    """Tìm mọi nghiệm dương f0=f1 bằng chia miền đơn điệu của log-PDF.

    u=x/pref, y=log(u): phi(y)=delta*y+c*exp(y)-logC.
    phi' có nhiều nhất một nghiệm, nên chia miền tại đó thu được mọi nghiệm
    biểu diễn được bằng float64. Không dùng một bounded minimizer đơn lẻ.
    """
    if h0 == h1:
        return []
    pref, delta, c, log_c = _log_equation(h0, h1)
    log_pref = math.log(pref)
    # Miền x dương biểu diễn được trong float64, chuyển sang y=log(x/pref).
    low = math.log(np.nextafter(0.0, 1.0)) - log_pref
    high = math.log(np.finfo(float).max) - log_pref - 2
    if delta == 0:
        u = log_c / c if c else -1.0
        y_roots = [math.log(u)] if u > 0 else []
    elif c == 0:
        y_roots = [log_c / delta]
    else:
        norm = max(abs(delta), abs(c), abs(log_c))
        ds, cs, ls = delta / norm, c / norm, log_c / norm

        def phi(y):
            exponent = math.log(abs(cs)) + y
            term = math.copysign(math.exp(min(exponent, 700)), cs)
            return ds * y + term - ls

        points = [low, high]
        if np.signbit(delta) != np.signbit(c):
            turn = math.log(-delta / c)
            if low < turn < high:
                points.insert(1, turn)
        y_roots = []
        for left, right in zip(points[:-1], points[1:]):
            f_left, f_right = phi(left), phi(right)
            if f_left == 0:
                y_roots.append(left)
            if f_right == 0:
                y_roots.append(right)
            if np.signbit(f_left) != np.signbit(f_right):
                # Giải nghiệm trong miền log để xử lý công suất rất nhỏ và tránh tràn số.
                y_roots.append(brentq(phi, left, right, xtol=1e-12, rtol=1e-14))
    roots = []
    for y in sorted(y_roots):
        log_x = log_pref + y
        if not math.log(np.nextafter(0.0, 1.0)) <= log_x < math.log(np.finfo(float).max):
            raise ArithmeticError("Ngưỡng dừng nằm ngoài miền float64; cần đổi thang hoặc precision")
        x = math.exp(log_x)
        if not roots or not math.isclose(x, roots[-1], rel_tol=1e-10, abs_tol=0):
            roots.append(x)
    if not roots:
        raise ArithmeticError("Hai Gamma khác nhau nhưng không bracket được giao PDF; cần kiểm tra số học")
    return roots


# So DEP tại toàn bộ điểm dừng và giá trị biên 1; không dùng một ngưỡng
# chọn sẵn hoặc chỉ lấy giao điểm PDF đầu tiên.
def minimize_gamma_dep(h0, h1):
    roots = stationary_thresholds(h0, h1)
    candidates = np.asarray([0.0, *roots])
    values = gamma_dep(candidates, h0, h1)
    winner = int(np.argmin(values))
    x = float(candidates[winner])
    return {"dep_min": float(values[winner]), "shift_star_w": x,
            "pfa": float(gammaincc(h0.shape, x / h0.scale_w)),
            "pmd": float(gammainc(h1.shape, x / h1.scale_w)),
            "equal_prior_error_probability": float(values[winner] / 2),
            "stationary_shifts_w": roots, "stationary_dep": gamma_dep(np.array(roots), h0, h1).tolist(),
            "boundary_dep": 1.0, "solver": "all_stationary_log_pdf_brentq_with_boundary"}


def lambertw_threshold_check(h0, h1):
    """Kiểm chứng phụ (11.10); báo rõ nếu đối số W không biểu diễn được."""
    if h0 == h1:
        return {"status": "identical_gamma", "stationary_shifts_w": []}
    pref, delta, c, log_c = _log_equation(h0, h1)
    if delta == 0 or c == 0:
        return {"status": "degenerate_analytic", "stationary_shifts_w": stationary_thresholds(h0, h1)}
    log_abs_argument = math.log(abs(c / delta)) + log_c / delta
    if not -700 < log_abs_argument < 700:
        return {"status": "argument_out_of_safe_range_use_log_pdf", "stationary_shifts_w": []}
    arg = math.copysign(math.exp(log_abs_argument), c / delta)
    if arg < -1 / math.e:
        return {"status": "argument_below_real_branch_check_roundoff", "stationary_shifts_w": []}
    branches = [0, -1] if arg < 0 else [0]
    roots = []
    for branch in branches:
        w = lambertw(arg, k=branch)
        u = delta / c * w
        if np.isfinite(u.real) and abs(u.imag) <= 1e-10 * max(1, abs(u.real)) and u.real > 0:
            roots.append(float(pref * u.real))
    return {"status": "computed", "stationary_shifts_w": sorted(set(roots))}


def gamma_grid(h0, h1, points=300, tail_tolerance=1e-10):
    if type(points) is not int or points < 2 or not 0 < tail_tolerance < .1:
        raise ValueError("Grid cần >=2 điểm và tail tolerance trong (0,0.1)")
    pref = max(h0.mean_w, h1.mean_w)
    upper_u = max(scipy_gamma.isf(tail_tolerance, h.shape, scale=h.scale_w / pref) for h in [h0, h1])
    if not np.isfinite(upper_u) or upper_u <= 0:
        raise ArithmeticError("Không xác định được cận grid Gamma")
    shifts = pref * np.linspace(0, upper_u, points)
    values = gamma_dep(shifts, h0, h1)
    best = int(np.argmin(values))
    return shifts, values, {"points": points, "tail_tolerance_each_law": tail_tolerance,
                            "dep_min": float(values[best]), "shift_star_w": float(shifts[best]),
                            "upper_shift_w": float(shifts[-1])}


# CDF thực nghiệm dùng đúng quy ước detector: H1 khi công suất > ngưỡng.
# Các mẫu H0/H1 có thể ghép cặp, không cần giả sử hai biến độc lập.
def empirical_dep(shift_w, z0, z1):
    """Detector: chọn H1 nếu Z>x; H0 nếu Z<=x. Xử lý ties nhất quán.

    PMD dùng <= để phù hợp partition của detector. Với phân phối liên tục,
    tương đương dấu < trong v4; tại sample knots, convention phải nhất quán.
    """
    z0, z1 = _paired_samples(z0, z1)
    shifts = np.asarray(shift_w, float)
    if np.any(np.isnan(shifts)):
        raise ValueError("Ngưỡng không được NaN")
    pfa = 1 - np.searchsorted(np.sort(z0), shifts, side="right") / len(z0)
    pmd = np.searchsorted(np.sort(z1), shifts, side="right") / len(z1)
    return pfa + pmd


def empirical_at_threshold(shift_w, z0, z1):
    z0, z1 = _paired_samples(z0, z1)
    pfa = z0 > shift_w
    pmd = z1 <= shift_w
    total = pfa.astype(float) + pmd.astype(float)
    return {"shift_w": float(shift_w), "pfa": float(pfa.mean()), "pmd": float(pmd.mean()),
            "dep": float(total.mean()), "paired_standard_error": float(total.std(ddof=1) / np.sqrt(len(total))),
            "equal_prior_error_probability": float(total.mean() / 2), "n_fading": len(total)}


# Tìm DEP nhỏ nhất của detector ngưỡng trên bank hiện tại. Cận DKW ở
# validation.py mới xử lý sai số lấy mẫu khi so với DEP quần thể.
def minimize_empirical_dep(z0, z1):
    """Quét toàn bộ điểm nhảy CDF; là minimum thực nghiệm trên chính bank."""
    z0, z1 = _paired_samples(z0, z1)
    # CDF thực nghiệm chỉ nhảy tại các mẫu; duyệt hết điểm nhảy tránh
    # bỏ sót ngưỡng tốt nhất do lưới ngưỡng quá thưa.
    knots = np.unique(np.r_[0.0, z0, z1])
    values = empirical_dep(knots, z0, z1)
    best = int(np.argmin(values))
    result = empirical_at_threshold(float(knots[best]), z0, z1)
    result["selected_threshold_pointwise_se_descriptive"] = result.pop("paired_standard_error")
    result.update({"method": "all_empirical_cdf_knots", "knots": len(knots),
                   "interpretation": "minimum_on_this_bank_not_population_certificate"})
    return result


def gamma_cdf_error(samples, law):
    ordered = np.sort(np.asarray(samples, float))
    cdf = gammainc(law.shape, ordered / law.scale_w)
    n = len(ordered)
    return float(max(np.max(np.arange(1, n + 1) / n - cdf), np.max(cdf - np.arange(n) / n)))
