"""Compare K=1 physics on channels exported by the supplied Octave model.

The original grid/rate conventions are checked separately from intentional v4
changes. This does not compare PSO trajectories or certify Gamma accuracy.
"""
import numpy as np
from scipy.special import gammaincc

from .channels import ChannelBank
from .config import dbm_to_w
from .evaluator import evaluate_with_dep
from .geometry import sample_layout


def bank_from_reference(c, layout, data):
    names = ["har", "hrb", "hrc", "hrw"]
    arrays = [np.asarray(data[name], complex) for name in names]
    shape = (c["n_fading"], c["n_ris"])
    if any(a.shape != shape or not np.all(np.isfinite(a)) for a in arrays):
        raise ValueError(f"Exported channels must all have shape {shape} and be finite")
    har, hrb, hrc, hrw = arrays
    ru = np.stack([hrb, hrc, hrw], axis=1)
    phase = np.exp(-1j * (np.angle(har) + np.angle(hrb)))
    h = np.sum(har[:, None, :] * ru * phase[:, None, :], axis=2)
    g = np.sum(ru * phase[:, None, :], axis=2)
    checks = {}
    for name, actual in [("phase", phase), ("X", np.abs(h) ** 2), ("Y", np.abs(g) ** 2)]:
        expected = np.asarray(data[name])
        if expected.shape != actual.shape or not np.all(np.isfinite(expected)):
            raise ValueError(f"Invalid exported {name}")
        scale = max(float(np.max(np.abs(expected))), np.finfo(float).tiny)
        error = float(np.max(np.abs(actual - expected)) / scale)
        checks[name] = {"max_absolute_error_over_global_scale": error, "pass": error <= 1e-10}
    signal = np.abs(h) ** 2 / (layout.loss_ar * layout.loss_ru[None, :])
    noise = np.abs(g) ** 2 / layout.loss_ru[None, :]
    return ChannelBank(har, ru, phase, h, g, signal, noise, float(np.mean(np.abs(har) ** 2))), checks


def compare_reference(c, data):
    if c["n_public"] != 1 or c.get("moment_ddof", 1) != 0 or c["public_half_width_m"] != 0:
        raise ValueError("Use legacy_physics_k1_v4.json: K=1, moment_ddof=0, fixed Carol")
    if int(np.asarray(data["schema_version"]).item()) != 1:
        raise ValueError("Unsupported reference schema")
    c = dict(c, n_fading=int(np.asarray(data["har"]).shape[0]))
    layout = sample_layout(c, np.random.default_rng(0))
    bank, channel_checks = bank_from_reference(c, layout, data)
    candidates = np.asarray(data["candidates"], float)
    expected = np.asarray(data["expected"], float)
    if candidates.ndim != 2 or candidates.shape[1] != 2 or len(candidates) < 1 or expected.shape != (len(candidates), 5):
        raise ValueError("Expected candidates (n,2) and results (n,5)")
    if not np.all(np.isfinite(candidates)) or not np.all(np.isfinite(expected)):
        raise ValueError("Reference results must be finite")
    rows = []
    columns = ["rho2", "bob_rate", "carol_self_rate", "dep_original_grid", "lambda_original_grid_w"]
    for (mb, s), reference in zip(candidates, expected):
        summary, samples, detail = evaluate_with_dep(c, layout, bank, [0], float(mb), [1 - mb], float(s), .9)
        h0, h1 = detail["h0"], detail["h1"]
        upper = max(h.mean_w + 8 * np.sqrt(h.shape) * h.scale_w for h in [h0, h1])
        shifts = np.linspace(0, upper, 300)
        # Match the original upper-Gamma subtraction and the ORIGINAL grid.
        values = gammaincc(h0.shape, shifts / h0.scale_w) + 1 - gammaincc(h1.shape, shifts / h1.scale_w)
        index = int(np.argmin(values))
        carol_self = float(np.log2(1 + samples["gamma_public"][:, 1, 0]).mean())
        actual = np.array([summary["power"]["rho2"], summary["bob_mean_rate_bit_s_hz"], carol_self,
                           values[index], shifts[index] + dbm_to_w(c["noise_dbm"]["willie"])])
        reference_w = reference.copy()
        reference_w[-1] /= 1000  # Octave's lambda is mW; Python's is W.
        passed = np.isclose(actual, reference_w, rtol=5e-9, atol=0)
        gap = float(actual[3] - summary["dep_min"])
        if gap < -1e-9:
            raise ArithmeticError("Original grid DEP is below stationary minimum")
        rows.append({"mb": float(mb), "s": float(s),
                     "original_convention_checks": {name: {"python": float(a), "octave_reference_in_python_units": float(e),
                                                             "absolute_difference": float(abs(a - e)), "pass": bool(ok)}
                                                    for name, a, e, ok in zip(columns, actual, reference_w, passed)},
                     "v4_method_differences": {"public_all_required_decoders_rate": summary["public_mean_rate_bit_s_hz"][0],
                                                "public_rate_reduction_from_carol_only": carol_self - summary["public_mean_rate_bit_s_hz"][0],
                                                "dep_stationary": summary["dep_min"], "grid_minus_stationary_dep": gap,
                                                "bob_outage_sample_estimate": summary["bob_outage_estimate"]},
                     "all_original_conventions_pass": bool(passed.all())})
    return {"scope": "Common-channel fixed-candidate comparison; not an exact PSO port or a population/Gamma accuracy certificate",
            "power_units": "W in this report; original Octave lambda converted from mW",
            "variance_ddof": 0, "channel_checks": channel_checks, "candidates": rows,
            "all_checks_pass": bool(all(x["pass"] for x in channel_checks.values()) and all(x["all_original_conventions_pass"] for x in rows))}
