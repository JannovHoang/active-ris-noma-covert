"""Variable-size version of the existing statistical SIC graph.

The feature meanings (15 node / 24 global / 2 edge) are preserved. Padding is
an in-memory batching detail only; records on disk retain their true size.
"""
import numpy as np

from .config import dbm_to_w
from .graph_data import NODE_FEATURES, GLOBAL_FEATURES, EDGE_FEATURES, gain_statistics, log_positive
from .metrics import power_budget
from .multik_core import public_count, sic_order

SCHEMA_VERSION = "statistical_sic_graph_variable_k_v1"


# Mỗi đồ thị biểu diễn một scenario/layout/K, không phải một mạng riêng.
# Các nút gồm Bob và K public; Willie đi vào đặc trưng toàn cục.
def statistical_graph(c, layout, bank, order, xi):
    k = public_count(c["n_public"])
    order = sic_order(order, k)
    if not np.isfinite(xi) or not 0 < xi < 1:
        raise ValueError("Need 0 < xi < 1")
    n = k + 1
    ranks = np.empty(n, int)
    # Bob có hạng mạnh nhất; order ánh xạ ID public gốc sang hạng SIC yếu-mạnh.
    ranks[0], ranks[order + 1] = n, np.arange(1, n)
    # Chỉ dùng thống kê bank đầu vào; không lấy kết quả audit hoặc nhãn vào feature.
    stats = gain_statistics(bank.signal_gain_tilde, bank.noise_gain_tilde)
    targets = np.r_[c["requirements"]["target_bob"], c["requirements"]["target_public"]]
    roles = np.zeros(n); roles[0] = 1
    # Các cột nút mã hóa vai trò, tọa độ tương đối, suy hao, nhiễu, yêu cầu rate,
    # hạng SIC và thống kê độ lợi; tên/ý nghĩa cột nằm trong graph_data.py.
    nodes = np.column_stack([roles, 1 - roles,
        (layout.receiver_xy[:n] - np.asarray(c["positions"]["ris"])) / 100.,
        log_positive(layout.loss_ru[:n]),
        log_positive(dbm_to_w(np.r_[c["noise_dbm"]["bob"], c["noise_dbm"]["public"]])),
        np.r_[0., c["requirements"]["qos_public"]], targets, np.expm1(np.log(2) * targets),
        ranks / n, stats[:n]])
    # Đồ thị có mọi cạnh có hướng giữa hai nút khác nhau. Hai thuộc tính cạnh
    # phân biệt quan hệ giải mã SIC và phần tín hiệu gây giao thoa.
    pairs = [(src, dst) for src in range(n) for dst in range(n) if src != dst]
    edges = np.asarray(pairs, int).T
    attributes = np.asarray([[float(src != 0 and ranks[src] < ranks[dst]),
                             float(src == 0 or ranks[src] > ranks[dst])] for src, dst in pairs])
    budget = power_budget(c, layout.loss_ar, c["bounds"]["s"][0])
    # Feature toàn cục gồm yêu cầu DEP, công suất, phần cứng, K và thống kê
    # Willie. Log dùng để giảm chênh lệch thang đo giữa các đại lượng dương.
    globals_ = np.asarray([xi, float(log_positive(dbm_to_w(c["total_dbm"]))),
        float(log_positive(budget["p0_w"])), c["kappa"], c["n_ris"], c["rician_k"],
        float(log_positive(layout.loss_ar)), float(log_positive(layout.loss_ru[-1])),
        float(log_positive(dbm_to_w(c["noise_dbm"]["ris"]))),
        float(log_positive(dbm_to_w(c["noise_dbm"]["willie"]))), c["requirements"]["outage_bob_max"],
        *c["bounds"]["mb"], *c["bounds"]["s"], k, float(log_positive(bank.mean_h_ar_power)),
        *stats[-1], float(log_positive(c["circuit_w"]["switch"])),
        float(log_positive(c["circuit_w"]["dc"]))], float)
    graph = {"n_public": k, "node_raw": nodes.tolist(), "global_raw": globals_.tolist(),
             "edge_index": edges.tolist(), "edge_attr": attributes.tolist()}
    validate_graph(graph)
    return graph


def validate_graph(graph):
    k = public_count(graph["n_public"])
    n, e = k + 1, k * (k + 1)
    node, glob = np.asarray(graph["node_raw"], float), np.asarray(graph["global_raw"], float)
    edges, attr = np.asarray(graph["edge_index"]), np.asarray(graph["edge_attr"], float)
    if (node.shape != (n, len(NODE_FEATURES)) or glob.shape != (len(GLOBAL_FEATURES),)
            or edges.shape != (2, e) or attr.shape != (e, len(EDGE_FEATURES))
            or not np.issubdtype(edges.dtype, np.integer)
            or np.any(edges < 0) or np.any(edges >= n)
            or not all(np.all(np.isfinite(a)) for a in (node, glob, attr))):
        raise ValueError("Variable-K graph dimensions/values are invalid")
    if not np.array_equal(node[:, :2], np.c_[np.arange(n) == 0, np.arange(n) != 0]):
        raise ValueError("Bob must be node zero followed by public nodes")
    pairs = list(map(tuple, edges.T.tolist()))
    if len(set(pairs)) != e or any(a == b for a, b in pairs):
        raise ValueError("Need all directed non-self SIC relations")
    if glob[15] != k:
        raise ValueError("Global public count differs from the graph")
    ranks = np.rint(node[:, 9] * n).astype(int)
    if (ranks[0] != n or not np.array_equal(np.sort(ranks[1:]), np.arange(1, n))
            or not np.allclose(node[:, 9], ranks / n, atol=1e-12, rtol=0)):
        raise ValueError("Invalid normalized SIC ranks")
    expected = np.asarray([[float(src != 0 and ranks[src] < ranks[dst]),
                            float(src == 0 or ranks[src] > ranks[dst])] for src, dst in pairs])
    if not np.array_equal(attr, expected):
        raise ValueError("Edge features disagree with the SIC ranks")
    return k


# Chỉ truyền các đồ thị train có nhãn vào đây. Fit bằng cả validation/test
# sẽ làm thông tin phân phối đánh giá lọt vào quá trình huấn luyện.
def fit_normalizer(graphs):
    if not graphs:
        raise ValueError("Need nonempty training graphs")
    for g in graphs:
        validate_graph(g)
    def parameters(a):
        mean, scale = a.mean(0), a.std(0, ddof=0)
        # Cột hằng có scale=1 để tránh chia 0; mean và scale sau đó được giữ cố định.
        return {"mean": mean.tolist(), "scale": np.where(scale < 1e-12, 1., scale).tolist()}
    return {"node": parameters(np.concatenate([g["node_raw"] for g in graphs])),
            "global": parameters(np.asarray([g["global_raw"] for g in graphs])),
            "method": "shared-node z-score; train graphs only; nodes equally weighted",
            "training_graph_count": len(graphs), "training_public_counts": sorted({g["n_public"] for g in graphs})}


# Ghép các K khác nhau vào batch bằng padding tới K lớn nhất trong batch.
# Mask phân biệt user/cạnh thật với ô đệm, không sinh thêm người dùng vật lý.
def pack_graphs(graphs, normalizer, targets=None):
    """Pad to this batch's largest K; true graph edges and users get masks."""
    if not graphs:
        raise ValueError("Need a nonempty graph batch")
    ks = [validate_graph(g) for g in graphs]
    b, n = len(graphs), max(ks) + 1
    e = n * (n - 1)
    data = {"node": np.zeros((b, n, len(NODE_FEATURES)), np.float32),
            "global_features": np.zeros((b, len(GLOBAL_FEATURES)), np.float32),
            "edge_index": np.zeros((b, 2, e), np.int64),
            "edge_attr": np.zeros((b, e, len(EDGE_FEATURES)), np.float32),
            "node_mask": np.zeros((b, n), bool), "edge_mask": np.zeros((b, e), bool)}
    if targets is not None:
        if len(targets) != b:
            raise ValueError("One target allocation is required per graph")
        data["target"] = np.zeros((b, n + 1), np.float32)
    for i, (g, k) in enumerate(zip(graphs, ks)):
        ni, ei = k + 1, k * (k + 1)
        data["node"][i, :ni] = ((np.asarray(g["node_raw"]) - normalizer["node"]["mean"])
                               / normalizer["node"]["scale"])
        data["global_features"][i] = ((np.asarray(g["global_raw"]) - normalizer["global"]["mean"])
                                     / normalizer["global"]["scale"])
        data["edge_index"][i, :, :ei], data["edge_attr"][i, :ei] = g["edge_index"], g["edge_attr"]
        # Chỉ các nút và cạnh thực được phép đóng góp vào forward, pooling và loss.
        data["node_mask"][i, :ni], data["edge_mask"][i, :ei] = True, True
        if targets is not None:
            y = np.asarray(targets[i], float)
            if y.shape != (k + 2,) or not np.all(np.isfinite(y)):
                raise ValueError("Target dimension must be K+2")
            data["target"][i, :ni], data["target"][i, -1] = y[:-1], y[-1]
    return data


# Bỏ padding sau suy luận; trả mỗi scenario đúng K+2 giá trị [mb, public, s].
def unpack_predictions(prediction, graphs):
    values = np.asarray(prediction)
    if values.ndim != 2 or values.shape[0] != len(graphs) or not np.all(np.isfinite(values)):
        raise ValueError("Invalid padded prediction array")
    result = []
    for y, g in zip(values, graphs):
        k = g["n_public"]
        if len(y) < k + 2:
            raise ValueError("Prediction is shorter than its true graph")
        result.append(np.r_[y[:k + 1], y[-1]].astype(float))
    return result
