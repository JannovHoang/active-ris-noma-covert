"""7c models: unchanged variable-K GNN and a parameter-matched pooled MLP."""
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .learning_models import dense_network
from .multik_models import VariableAllocationGNN, balanced_allocation_loss


# Đối chứng MLP dùng cùng phép biến đổi cho từng user; tham số edges và
# attributes được nhận để giữ giao diện, nhưng không được dùng trong tính toán.
class PointwiseLayer(nn.Module):
    """Shared per-user residual MLP, with no edge aggregation/messages."""
    def __init__(self, hidden, inner):
        super().__init__()
        self.network = dense_network([hidden, inner, hidden])

    def forward(self, h, context, edges, attributes, node_mask, edge_mask):
        return F.relu(h + self.network(h)) * node_mask.unsqueeze(-1)


# MLP vẫn nhận đặc trưng user/hạng SIC và pooling toàn cục. Khác biệt chính
# với GNN là không tổng hợp thông điệp theo cạnh, không phải thiếu mọi thông tin SIC.
class PooledAllocationMLP(VariableAllocationGNN):
    """DeepSets-style baseline: same encoders/heads, pointwise MLP blocks.

    Inherits only the shared masked input/readout implementation. Edge values
    do not influence this baseline. No maximum K or untrained padded slot is
    introduced, so an unseen K is a meaningful test for both architectures.
    """
    def __init__(self, bounds, hidden=64, layers=3, inner=224):
        # Dùng chung encoder/readout/head với GNN để đối chứng có cùng loại thông tin
        # và ràng buộc đầu ra; thay phần xử lý giữa bằng các khối pointwise.
        super().__init__(bounds, hidden, layers=0)
        self.layers = nn.ModuleList([PointwiseLayer(hidden, inner) for _ in range(layers)])


# Chọn kiến trúc từ cấu hình đã lưu; hidden/layers không được đổi tùy ý
# khi nạp một checkpoint đã huấn luyện.
def build_model(kind, bounds, config):
    if kind == "gnn":
        return VariableAllocationGNN(bounds, config["hidden"], config["layers"])
    if kind == "mlp":
        return PooledAllocationMLP(bounds, config["hidden"], config["layers"], config["mlp_inner"])
    raise ValueError("7c model kind must be gnn or mlp")


def error_metrics(prediction, target, node_mask, weights):
    errors = (prediction - target).square()
    public_mask = node_mask[:, 1:].to(errors.dtype)
    k = public_mask.sum(-1)
    # MSE theo mỗi scenario chỉ tính K+2 hệ số thực; không dùng ô đệm
    # để làm sai số trung bình giảm giả.
    case_mse = (errors[:, 0] + (errors[:, 1:-1]*public_mask).sum(-1) + errors[:, -1]) / (k+2)
    weighted = balanced_allocation_loss(prediction, target, node_mask, weights["mb"], weights["public_total"], weights["s"])
    return {"case_mse": float(case_mse.mean()), "weighted_loss": float(weighted),
            "mb_mae": float((prediction[:, 0]-target[:, 0]).abs().mean()),
            "public_mae": float((((prediction[:, 1:-1]-target[:, 1:-1]).abs()*public_mask).sum(-1)/k).mean()),
            "s_mae": float((prediction[:, -1]-target[:, -1]).abs().mean())}


# Lưu cả trọng số và nguồn gốc huấn luyện: normalizer, K, seed, epoch
# và cấu hình. Các trường này giúp kiểm tra checkpoint không dùng test để chọn.
def checkpoint(model, state, kind, seed, epoch, plan, bounds, validation):
    return {"format": "variable_k_learning_7c_v1", "state_dict": state, "model_kind": kind,
            "seed": seed, "epoch": epoch, "training_config": plan["config"], "bounds": bounds,
            "normalization": plan["normalization"], "trained_public_counts": plan["training_public_counts"],
            "validation_errors": validation, "test_used_for_training_or_selection": False,
            "physical_gradient": False, "plan_sha256": plan["plan_sha256"]}


def save_torch(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


# Nạp trọng số vào kiến trúc khớp cấu hình đã lưu, rồi bật chế độ suy luận.
def load_model(path):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved.get("format") != "variable_k_learning_7c_v1" or saved["test_used_for_training_or_selection"]:
        raise ValueError("Not a supported frozen 7c checkpoint")
    model = build_model(saved["model_kind"], saved["bounds"], saved["training_config"])
    model.load_state_dict(saved["state_dict"], strict=True)
    model.eval()
    return model, saved
