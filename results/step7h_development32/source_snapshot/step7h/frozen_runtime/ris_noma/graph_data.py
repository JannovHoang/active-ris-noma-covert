"""Pre-allocation statistical graph features, following derivation v4 section 18.

Node order is [Bob, public original ID 0, public original ID 1]. Willie,
Alice and RIS enter through global features. Edges are decoding/interference
relations, not physical user-to-user propagation channels.
"""
import numpy as np
from .config import dbm_to_w
from .metrics import power_budget
from .objective import decode_k2

NODE_FEATURES = ["is_bob", "is_public", "ris_relative_x_over_100m", "ris_relative_y_over_100m",
                 "log10_loss_ru", "log10_receiver_noise_w", "qos_public_bit_s_hz", "target_rate_bit_s_hz",
                 "target_sinr_linear", "sic_rank_over_K_plus_1", "log10_mean_signal_gain_tilde",
                 "signal_gain_cv", "log10_mean_noise_gain_tilde", "noise_gain_cv", "signal_noise_correlation"]
GLOBAL_FEATURES = ["xi", "log10_total_power_w", "log10_p0_w", "kappa", "n_ris", "rician_k",
                   "log10_loss_ar", "log10_loss_rw", "log10_ris_noise_w", "log10_willie_noise_w",
                   "bob_outage_max", "mb_min", "mb_max", "s_min", "s_max", "n_public",
                   "log10_mean_h_ar_power", "willie_log10_mean_signal_gain_tilde", "willie_signal_gain_cv",
                   "willie_log10_mean_noise_gain_tilde", "willie_noise_gain_cv", "willie_signal_noise_correlation",
                   "log10_circuit_ps_w", "log10_circuit_pdc_w"]
EDGE_FEATURES = ["source_stream_decoded_at_destination", "source_stream_interferes_at_destination"]
SCHEMA_VERSION = "statistical_sic_graph_k2_v1"


def log_positive(x):
    # Explicit numerical floor, also covering zero circuit power in other configs.
    return np.log10(np.maximum(np.asarray(x, dtype=float), 1e-300))


def gain_statistics(signal, noise):
    signal, noise = np.asarray(signal, float), np.asarray(noise, float)
    if signal.shape != noise.shape or signal.ndim != 2 or len(signal) < 2:
        raise ValueError("Need paired (M, receivers) gains, M >= 2")
    if not np.all(np.isfinite(signal)) or not np.all(np.isfinite(noise)) or np.any(signal < 0) or np.any(noise < 0):
        raise ValueError("Gains must be finite and nonnegative")
    mg, mq = signal.mean(axis=0), noise.mean(axis=0)
    sg, sq = signal.std(axis=0, ddof=0), noise.std(axis=0, ddof=0)
    covariance = ((signal - mg) * (noise - mq)).mean(axis=0)
    corr = np.divide(covariance, sg * sq, out=np.zeros_like(mg), where=sg * sq > 0)
    return np.column_stack([log_positive(mg), np.divide(sg, mg, out=np.zeros_like(mg), where=mg > 0),
                            log_positive(mq), np.divide(sq, mq, out=np.zeros_like(mq), where=mq > 0),
                            np.clip(corr, -1, 1)])


def statistical_graph(c, layout, bank, order, xi):
    """No allocation, PSO output, validation or heldout argument is accepted."""
    if c["n_public"] != 2 or not np.isfinite(xi) or not 0 < xi < 1:
        raise ValueError("This schema supports K=2 and 0 < xi < 1")
    order = np.asarray(order, int)
    if order.shape != (2,) or not np.array_equal(np.sort(order), [0, 1]):
        raise ValueError("Invalid SIC mapping")
    ranks = np.empty(3, dtype=int)
    ranks[0] = 3
    ranks[order + 1] = [1, 2]
    stats = gain_statistics(bank.signal_gain_tilde, bank.noise_gain_tilde)
    relative = (layout.receiver_xy[:3] - np.asarray(c["positions"]["ris"])) / 100.
    targets = np.r_[c["requirements"]["target_bob"], c["requirements"]["target_public"]]
    qos = np.r_[0., c["requirements"]["qos_public"]]
    nodes = np.column_stack([np.array([1., 0., 0.]), np.array([0., 1., 1.]), relative,
                             log_positive(layout.loss_ru[:3]),
                             log_positive(dbm_to_w(np.r_[c["noise_dbm"]["bob"], c["noise_dbm"]["public"]])),
                             qos, targets, np.expm1(np.log(2) * targets), ranks / 3., stats[:3]])
    # For src -> dst, ask if dst decodes src's stream, or treats it as interference.
    pairs = [(src, dst) for src in range(3) for dst in range(3) if src != dst]
    edge_index = np.asarray(pairs, int).T
    edge_attr = np.asarray([[float(src != 0 and ranks[src] < ranks[dst]),
                             float(src == 0 or ranks[src] > ranks[dst])] for src, dst in pairs])
    budget = power_budget(c, layout.loss_ar, c["bounds"]["s"][0])
    global_features = np.asarray([xi, float(log_positive(dbm_to_w(c["total_dbm"]))),
        float(log_positive(budget["p0_w"])), c["kappa"], c["n_ris"], c["rician_k"],
        float(log_positive(layout.loss_ar)), float(log_positive(layout.loss_ru[-1])),
        float(log_positive(dbm_to_w(c["noise_dbm"]["ris"]))),
        float(log_positive(dbm_to_w(c["noise_dbm"]["willie"]))), c["requirements"]["outage_bob_max"],
        *c["bounds"]["mb"], *c["bounds"]["s"], c["n_public"], float(log_positive(bank.mean_h_ar_power)),
        *stats[-1], float(log_positive(c["circuit_w"]["switch"])),
        float(log_positive(c["circuit_w"]["dc"]))], float)
    if nodes.shape != (3, len(NODE_FEATURES)) or global_features.shape != (len(GLOBAL_FEATURES),):
        raise ArithmeticError("Feature schema size mismatch")
    return {"node_raw": nodes, "global_raw": global_features, "edge_index": edge_index, "edge_attr": edge_attr}


def original_allocation(allocation_sic, order, bounds):
    mb, s = float(allocation_sic["mb"]), float(allocation_sic["s"])
    mp_sic = np.asarray(allocation_sic["public"], float)
    order = np.asarray(order, int)
    if mp_sic.shape != (2,) or not np.array_equal(np.sort(order), [0, 1]):
        raise ValueError("Invalid allocation or SIC order")
    mp_original = np.empty(2)
    mp_original[order] = mp_sic
    y = np.r_[mb, mp_original, s]
    if not np.all(np.isfinite(y)) or np.any(y[:3] < 0) or not np.isclose(y[:3].sum(), 1, rtol=0, atol=1e-12):
        raise ValueError("Invalid power simplex")
    if not bounds["mb"][0] <= mb <= bounds["mb"][1] or not bounds["s"][0] <= s <= bounds["s"][1]:
        raise ValueError("Allocation outside declared bounds")
    z = np.array([(mb - bounds["mb"][0]) / np.ptp(bounds["mb"]),
                  mp_sic[0] / (1 - mb), (s - bounds["s"][0]) / np.ptp(bounds["s"])])
    decoded_mb, decoded_mp, decoded_s = decode_k2(z, bounds)
    if not np.allclose([decoded_mb, *decoded_mp, decoded_s], [mb, *mp_sic, s], rtol=0, atol=1e-12):
        raise ArithmeticError("Allocation did not round-trip to PSO coordinates")
    return y, z


def allocation_head(bob_logit, public_logits, s_logit, bounds):
    """NumPy reference for the bounded output head to be used by the later GNN."""
    p = np.asarray(public_logits, float)
    if p.shape != (2,) or not np.all(np.isfinite([bob_logit, *p, s_logit])):
        raise ValueError("Need finite K=2 head logits")
    def sigmoid(x):
        return .5 * (1 + np.tanh(.5 * float(x)))
    mb = bounds["mb"][0] + np.ptp(bounds["mb"]) * sigmoid(bob_logit)
    s = bounds["s"][0] + np.ptp(bounds["s"]) * sigmoid(s_logit)
    weights = np.exp(p - p.max());weights /= weights.sum()
    return np.r_[mb, (1 - mb) * weights, s]


def fit_normalizer(node, global_features):
    node, global_features = np.asarray(node, float), np.asarray(global_features, float)
    if node.ndim != 3 or global_features.ndim != 2 or not len(node) or len(node) != len(global_features):
        raise ValueError("Need nonempty training graphs")
    if not np.all(np.isfinite(node)) or not np.all(np.isfinite(global_features)):
        raise ValueError("Features must be finite")
    def params(x, axis):
        mean, scale = x.mean(axis=axis), x.std(axis=axis, ddof=0)
        scale = np.where(scale < 1e-12, 1., scale)
        return {"mean": mean.tolist(), "scale": scale.tolist()}
    return {"node": params(node, (0, 1)), "global": params(global_features, 0),
            "method": "shared-node z-score; fitted on train only; constant scale becomes 1",
            "training_graph_count": len(node)}


def transform_graphs(node, global_features, normalizer):
    result = []
    for x, key in [(node, "node"), (global_features, "global")]:
        p = normalizer[key]
        result.append((np.asarray(x, float) - np.asarray(p["mean"])) / np.asarray(p["scale"]))
    return tuple(result)
