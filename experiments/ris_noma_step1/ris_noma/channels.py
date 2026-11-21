"""Rician chuẩn hóa; giữ đúng mô hình scalar-noise Gu của v4."""
from dataclasses import dataclass
import numpy as np


# NLoS Gaussian phức chuẩn hóa: chia sqrt(2) để tổng công suất trung bình
# của phần thực và ảo bằng 1.
def complex_normal(rng, shape):
    """CN(0,1): phần thực và ảo đều có variance 1/2."""
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / np.sqrt(2.0)


# k_factor là tỷ số công suất LoS/NLoS, không phải số public user K.
# Các hệ số căn bảo toàn công suất kênh trung bình khi đổi k_factor.
def rician(rng, shape, k_factor, los):
    """(2.12); |los|=1, vì vậy E|h|²=1."""
    if not np.isfinite(k_factor) or k_factor < 0 or not np.allclose(np.abs(los), 1):
        raise ValueError("Rician cần KR >= 0 và |hLoS| = 1")
    return np.sqrt(k_factor / (k_factor + 1)) * los + np.sqrt(1 / (k_factor + 1)) * complex_normal(rng, shape)


@dataclass(frozen=True)
class ChannelBank:
    h_ar: np.ndarray  # (M,N), dùng chung cho MỖI receiver.
    h_ru: np.ndarray  # (M,K+2,N), [Bob, public gốc, Willie].
    phase: np.ndarray  # (M,N): exp(j theta), chưa có rho.
    h_tilde: np.ndarray  # (M,K+2)
    g_scalar_tilde: np.ndarray  # (M,K+2), tổng phức Gu/rho.
    signal_gain_tilde: np.ndarray  # (M,K+2): |H_tilde|²/(Lar Lru).
    noise_gain_tilde: np.ndarray  # (M,K+2): |G_tilde|²/Lru.
    mean_h_ar_power: float


# Một bank có M hiện thực fading và N phần tử RIS; cùng h_ar được dùng
# cho mọi receiver để giữ đúng tương quan vật lý của đường Alice-RIS.
def build_bank(c, layout, rng):
    M, N, K = c["n_fading"], c["n_ris"], c["n_public"]
    if c["los"]["mode"] == "unit_phasor_demo":
        los_ar, los_ru = np.ones(N, complex), np.ones((K + 2, N), complex)
    else:
        los_ar = np.exp(1j * np.asarray(c["los"]["ar_phase_rad"]))
        los_ru = np.exp(1j * np.asarray(c["los"]["ru_phase_rad"]))
    h_ar = rician(rng, (M, N), c["rician_k"], los_ar)
    # Trục receiver lưu Bob, K public theo ID gốc, rồi Willie.
    h_ru = rician(rng, (M, K + 2, N), c["rician_k"], los_ru)
    # (3.4), (5.3)-(5.7). Không dùng liên hợp: v4 định nghĩa tích har*hru.
    # Mỗi phần tử bù pha hai chặng tới Bob để các thành phần cộng đồng pha.
    # Đây là quy tắc pha định trước; PSO không tối ưu riêng N góc pha.
    phase = np.exp(-1j * (np.angle(h_ar) + np.angle(h_ru[:, 0, :])))
    h_tilde = np.sum(h_ar[:, None, :] * h_ru * phase[:, None, :], axis=2)
    # Giữ mô hình nhiễu scalar RIS đã chọn: cộng biên độ phức trước khi lấy
    # bình phương mô-đun; không thay bằng tổng công suất nhiễu độc lập.
    g_tilde = np.sum(h_ru * phase[:, None, :], axis=2)
    # Độ lợi đã gồm suy hao đường truyền nhưng chưa nhân rho2 khuếch đại RIS.
    signal = np.abs(h_tilde) ** 2 / (layout.loss_ar * layout.loss_ru[None, :])
    noise = np.abs(g_tilde) ** 2 / layout.loss_ru[None, :]
    arrays = [h_ar, h_ru, phase, h_tilde, g_tilde, signal, noise]
    for a in arrays:
        a.setflags(write=False)  # Không để evaluator vô tình thay đổi bank dùng chung.
    return ChannelBank(*arrays, float(np.mean(np.abs(h_ar) ** 2)))


# Xếp public yếu tới mạnh theo độ lợi trung bình; ID và hạng SIC là hai
# khái niệm khác nhau, cần ánh xạ khi lưu phân bổ.
def public_order(bank):
    """Rank dài hạn theo gain trung bình từ bank label, cố định cho cả evaluation."""
    return np.argsort(bank.signal_gain_tilde[:, 1:-1].mean(axis=0), kind="stable")
