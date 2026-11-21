"""Một layout cố định cho M fading realizations."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Layout:
    public_xy: np.ndarray  # Thứ tự vật lý user 0,1,... trước khi sắp SIC.
    receiver_xy: np.ndarray  # [Bob, public 0,..., public K-1, Willie].
    distance_ar_m: float
    distance_ru_m: np.ndarray
    loss_ar: float
    loss_ru: np.ndarray


def path_loss(distance_m, beta0, exponent, d0):
    # (3.2): mẫu số suy hao công suất; KHÔNG phải hệ số biên độ.
    return (np.asarray(distance_m, float) / d0) ** exponent / beta0


def sample_layout(c, rng):
    p = c["positions"]
    center = np.asarray(p["public_center"], float)
    public = center + rng.uniform(-c["public_half_width_m"], c["public_half_width_m"], size=(c["n_public"], 2))
    receivers = np.vstack([p["bob"], public, p["willie"]])
    d_ar = float(np.linalg.norm(np.asarray(p["alice"]) - p["ris"]))
    d_ru = np.linalg.norm(receivers - np.asarray(p["ris"]), axis=1)
    if d_ar < c["min_link_distance_m"] or np.any(d_ru < c["min_link_distance_m"]):
        raise ValueError("Layout bị loại: có liên kết ngắn hơn min_link_distance_m; không tự sửa khoảng cách")
    pl = c["pathloss"]
    return Layout(public, receivers, d_ar, d_ru,
                  float(path_loss(d_ar, pl["beta0_ar"], pl["exponent_ar"], pl["reference_distance_m"])),
                  path_loss(d_ru, pl["beta0_ru"], pl["exponent_ru"], pl["reference_distance_m"]))
