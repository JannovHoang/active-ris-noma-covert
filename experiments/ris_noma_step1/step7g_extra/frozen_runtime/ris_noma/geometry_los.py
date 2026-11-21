"""7d geometry-dependent, unit-modulus LoS; legacy channel/noise code is reused.

Only the relative propagation PHASE is extended. Center-distance path loss,
i.i.d. Rician NLoS, Bob alignment and scalar RIS noise are unchanged. This is
not a full electromagnetic or calibrated three-dimensional deployment model.
"""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json

import numpy as np

from .channels import build_bank
from .config import validate_config
from .geometry import path_loss
from .multik_data_core import geometry

SPEED_OF_LIGHT = 299792458.0
PROFILES = ("legacy", "geometry", "geometry_friis")
STREAMS = {"search": 301, "selection": 302, "audit": 303, "pso": 304}


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def array_elements(center, count, carrier_hz, spacing_wavelengths, axis):
    center, axis = np.asarray(center, float), np.asarray(axis, float)
    if center.shape != (2,) or axis.shape != (2,) or not np.all(np.isfinite(np.r_[center, axis])):
        raise ValueError("Array center and axis require finite 2D coordinates")
    if type(count) is not int or count < 1 or not np.isfinite(carrier_hz) or carrier_hz <= 0:
        raise ValueError("Positive element count and carrier frequency required")
    if not np.isfinite(spacing_wavelengths) or not 0 < spacing_wavelengths <= .5:
        raise ValueError("Use a declared positive spacing at most half a wavelength")
    norm = float(np.linalg.norm(axis))
    if norm == 0:
        raise ValueError("Array axis cannot be zero")
    wavelength = SPEED_OF_LIGHT / carrier_hz
    offsets = (np.arange(count) - (count - 1) / 2) * spacing_wavelengths * wavelength
    return center + offsets[:, None] * (axis / norm)[None, :]


def relative_los_phases(points, center, elements, wavelength):
    points, center, elements = np.asarray(points, float), np.asarray(center, float), np.asarray(elements, float)
    if points.ndim != 2 or points.shape[1] != 2 or not np.all(np.isfinite(points)):
        raise ValueError("LoS endpoints require finite 2D coordinates")
    distances = np.linalg.norm(points[:, None, :] - elements[None, :, :], axis=-1)
    reference = np.linalg.norm(points - center, axis=-1)
    if np.any(distances <= 0) or np.any(reference <= 0) or wavelength <= 0:
        raise ValueError("LoS propagation distances and wavelength must be positive")
    # A common center-reference phase is removed per link. No conjugation is
    # introduced: the inherited effective channel multiplies h_ar*h_ru.
    return np.angle(np.exp(-2j * np.pi * (distances - reference[:, None]) / wavelength))


def apply_profile(c, layout, profile, deployment, settings):
    if profile not in PROFILES:
        raise ValueError("Unknown 7d channel profile")
    c = deepcopy(c)
    wavelength = SPEED_OF_LIGHT / settings["carrier_hz"]
    axis = deployment.get("array_axis", settings["array_axis"])
    elements = array_elements(c["positions"]["ris"], c["n_ris"], settings["carrier_hz"], settings["spacing_wavelengths"], axis)
    if profile != "legacy":
        endpoints = np.vstack([c["positions"]["alice"], layout.receiver_xy])
        phases = relative_los_phases(endpoints, c["positions"]["ris"], elements, wavelength)
        c["los"] = {"mode": "explicit_phases", "ar_phase_rad": phases[0].tolist(), "ru_phase_rad": phases[1:].tolist()}
    if profile == "geometry_friis":
        d0 = c["pathloss"]["reference_distance_m"]
        beta0 = float((wavelength / (4 * np.pi * d0)) ** 2)
        c["pathloss"].update(beta0_ar=beta0, beta0_ru=beta0)
        pl = c["pathloss"]
        layout = replace(layout,
            loss_ar=float(path_loss(layout.distance_ar_m, beta0, pl["exponent_ar"], d0)),
            loss_ru=path_loss(layout.distance_ru_m, beta0, pl["exponent_ru"], d0))
    validate_config(c)
    metadata = {"profile": profile, "deployment_id": deployment["id"],
        "alice": c["positions"]["alice"], "ris": c["positions"]["ris"],
        "receiver_xy": layout.receiver_xy.tolist(), "elements_xy_m": elements.tolist(),
        "carrier_hz": settings["carrier_hz"], "wavelength_m": wavelength,
        "spacing_wavelengths": settings["spacing_wavelengths"], "array_axis": list(axis),
        "los_sha256": fingerprint(c["los"]), "los": c["los"],
        "loss_ar": layout.loss_ar, "loss_ru": layout.loss_ru.tolist(), "pathloss": c["pathloss"],
        "phase_model": "preserved_explicit_vectors" if profile == "legacy" else "unit_phasors_from_exact_element_distance_minus_center_distance",
        "scope": "phase-only geometric extension; center-distance amplitude; no element pattern, shadowing, polarization or full EM near-field model"}
    return c, layout, metadata


def channel_context(plan, source, dataset_plan, job, role):
    from types import SimpleNamespace
    import time
    settings = plan["config"]
    samples = settings[{"search": "search_samples", "selection": "selection_samples", "audit": "audit_samples"}[role]]
    adapted = deepcopy(source)
    adapted["base_config"]["positions"].update(alice=job["deployment"]["alice"], ris=job["deployment"]["ris"])
    c, layout = geometry(dataset_plan, adapted, job, samples)
    c, layout, metadata = apply_profile(c, layout, job["profile"], job["deployment"], settings)
    entropy = [c["root_seed"], job["layout_id"], job["n_public"], STREAMS[role]]
    # Profile/deployment are intentionally absent: comparisons use common
    # normalized NLoS draws, while each bank ROLE is independent.
    started = time.perf_counter()
    full = build_bank(c, layout, np.random.default_rng(np.random.SeedSequence(entropy)))
    seconds = time.perf_counter() - started
    names = ("h_ar", "h_tilde", "g_scalar_tilde", "signal_gain_tilde", "noise_gain_tilde", "mean_h_ar_power")
    bank = SimpleNamespace(**{key: getattr(full, key) for key in names})
    metadata.update(bank_role=role, bank_entropy=entropy, bank_samples=samples,
                    bank_generation_seconds=seconds)
    return c, layout, bank, metadata


def pso_rng(root_seed, job, replicate_seed, power_index, xi_index):
    # Same PSO random trajectory across methods, margins and paired profiles.
    return np.random.default_rng(np.random.SeedSequence([
        root_seed, job["layout_id"], job["n_public"], STREAMS["pso"], replicate_seed, power_index, xi_index]))
