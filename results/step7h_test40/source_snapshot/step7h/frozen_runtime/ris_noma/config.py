"""Cấu hình JSON; không âm thầm điền giá trị còn thiếu của baseline."""
from pathlib import Path
import json
import numpy as np


def dbm_to_w(dbm):
    """Công thức (4.3). Hàm nhận cả số và mảng NumPy."""
    return 10.0 ** ((np.asarray(dbm, dtype=float) - 30.0) / 10.0)


def _missing(value, prefix=""):
    if value is None:
        return [prefix]
    if isinstance(value, dict):
        return [p for k, v in value.items() for p in _missing(v, f"{prefix}.{k}".strip("."))]
    if isinstance(value, list):
        return [p for i, v in enumerate(value) for p in _missing(v, f"{prefix}[{i}]")]
    return []


def validate_config(c):
    required = ["profile", "n_ris", "n_public", "n_fading", "rician_k", "kappa",
                "total_dbm", "root_seed", "layout_id", "positions", "public_half_width_m",
                "min_link_distance_m", "pathloss", "noise_dbm", "circuit_w", "los",
                "requirements", "bounds", "reference_allocation"]
    absent = [k for k in required if k not in c]
    if absent:
        raise ValueError(f"Thiếu trường cấu hình: {absent}")
    pending = _missing({k: c[k] for k in required if k != "los"})
    if c["los"]["mode"] != "unit_phasor_demo":
        pending += _missing(c["los"], "los")
    if pending:
        raise ValueError("Cần điền tham số baseline, không dùng số demo thay thế: " + ", ".join(pending))
    for k in ["n_ris", "n_public", "n_fading", "root_seed", "layout_id"]:
        if type(c[k]) is not int or c[k] < (2 if k == "n_fading" else 1 if k in ["n_ris", "n_public"] else 0):
            raise ValueError(f"{k}: cần số nguyên trong miền hợp lệ")
    K, N = c["n_public"], c["n_ris"]
    if type(c.get("moment_ddof", 1)) is not int or c.get("moment_ddof", 1) not in [0, 1]:
        raise ValueError("moment_ddof phải là 0 hoặc 1")
    if c["profile"] not in ["demo", "baseline"]:
        raise ValueError("profile phải là demo hoặc baseline")
    for k in ["rician_k", "kappa", "public_half_width_m"]:
        if not np.isfinite(c[k]) or c[k] < 0:
            raise ValueError(f"{k} phải hữu hạn và không âm")
    if not np.isfinite(c["min_link_distance_m"]) or c["min_link_distance_m"] <= 0:
        raise ValueError("min_link_distance_m phải dương")
    for name in ["alice", "ris", "bob", "willie", "public_center"]:
        a = np.asarray(c["positions"][name], float)
        if a.shape != (2,) or not np.all(np.isfinite(a)):
            raise ValueError(f"positions.{name}: cần hai tọa độ hữu hạn, đơn vị mét")
    if not all(np.isfinite(v) and v > 0 for v in c["pathloss"].values()):
        raise ValueError("Các tham số pathloss phải hữu hạn và dương")
    noise = c["noise_dbm"]
    if len(noise["public"]) != K or not np.all(np.isfinite([noise["ris"], noise["bob"], noise["willie"], *noise["public"]])):
        raise ValueError("noise_dbm: cần đủ công suất nhiễu cho từng user")
    if not all(np.isfinite(v) and v >= 0 for v in c["circuit_w"].values()):
        raise ValueError("Công suất mạch phải hữu hạn và không âm")
    if not np.isfinite(c["total_dbm"]) or dbm_to_w(c["total_dbm"]) <= N * sum(c["circuit_w"].values()):
        raise ValueError("P0 = Ptot - N(PS + PDC) phải dương")
    for key in ["mb", "s"]:
        lo, hi = c["bounds"][key]
        if not 0 < lo < hi < 1:
            raise ValueError(f"bounds.{key}: cần 0 < min < max < 1")
    req = c["requirements"]
    for key in ["qos_public", "target_public"]:
        if len(req[key]) != K or not np.all(np.isfinite(req[key])) or np.any(np.asarray(req[key]) <= 0):
            raise ValueError(f"requirements.{key}: cần K giá trị hữu hạn, dương")
    if not np.isfinite(req["target_bob"]) or req["target_bob"] <= 0 or not 0 < req["outage_bob_max"] < 1:
        raise ValueError("Target Bob phải dương và 0 < outage_bob_max < 1")
    if not req["xi"] or any(not 0 < x < 1 for x in req["xi"]):
        raise ValueError("Mỗi xi phải nằm trong (0,1)")
    los = c["los"]
    if los["mode"] == "unit_phasor_demo":
        if c["profile"] != "demo":
            raise ValueError("LoS=1 chỉ là quy ước demo; cần LoS baseline cho profile baseline")
    elif los["mode"] == "explicit_phases":
        for key, shape in [("ar_phase_rad", (N,)), ("ru_phase_rad", (K + 2, N))]:
            phase = np.asarray(los[key], float)
            if phase.shape != shape or not np.all(np.isfinite(phase)):
                raise ValueError(f"los.{key}: cần shape {shape}, pha tính theo radian")
    else:
        raise ValueError("los.mode chưa được hỗ trợ")
    allocation = c["reference_allocation"]
    fractions = np.asarray([allocation["mb"], *allocation["public"]], float)
    if len(allocation["public"]) != K or not np.all(np.isfinite(fractions)) or np.any(fractions < 0) or not np.isclose(fractions.sum(), 1, atol=1e-12, rtol=0):
        raise ValueError("reference_allocation: cần mb + tổng public = 1 và mọi thành phần không âm")
    if not c["bounds"]["mb"][0] <= allocation["mb"] <= c["bounds"]["mb"][1] or not c["bounds"]["s"][0] <= allocation["s"] <= c["bounds"]["s"][1]:
        raise ValueError("reference_allocation nằm ngoài bounds")
    return c


def load_config(path):
    with Path(path).open(encoding="utf-8") as f:
        return validate_config(json.load(f))
