"""One parameter set for variable K; masked dense batching, pure PyTorch."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .learning_models import dense_network
from .learning_train import load_checkpoint
from .multik_graph import pack_graphs, unpack_predictions


# Đầu ra dùng chung tham số cho mọi K; số logit public đi theo số nút thật.
# Head bảo toàn miền biến/simplex, chưa bảo đảm DEP, QoS hoặc outage.
class VariablePowerHead(nn.Module):
    def __init__(self, bounds):
        super().__init__()
        self.register_buffer("intervals", torch.tensor([bounds["mb"], bounds["s"]], dtype=torch.float32))

    def forward(self, bob_logit, public_logits, s_logit, public_mask):
        # Sigmoid ánh xạ logit Bob vào khoảng mb đã định, không cần cắt gradient.
        mb = self.intervals[0, 0] + (self.intervals[0, 1] - self.intervals[0, 0]) * torch.sigmoid(bob_logit)
        s = self.intervals[1, 0] + (self.intervals[1, 1] - self.intervals[1, 0]) * torch.sigmoid(s_logit)
        # Đặt logit ô đệm thành -inf để softmax chỉ chia công suất cho public thật.
        weights = torch.softmax(public_logits.masked_fill(~public_mask, -torch.inf), dim=-1)
        # Các public cùng chia phần còn lại 1-mb nên tổng hệ số công suất bằng 1.
        public = (1 - mb).unsqueeze(-1) * weights
        return torch.cat([mb.unsqueeze(-1), public, s.unsqueeze(-1)], dim=-1)


# Một lớp GNN: tạo thông điệp từ nút nguồn, nút đích và quan hệ SIC,
# rồi tổng hợp tại nút đích trước khi cập nhật biểu diễn nút.
class VariableMessageLayer(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.message = dense_network([2 * hidden + 2, hidden, hidden], final_relu=True)
        self.update = dense_network([3 * hidden, hidden, hidden])

    def forward(self, h, context, edges, attributes, node_mask, edge_mask):
        # Chỉ số source/destination của từng cạnh, giữ riêng cho mỗi đồ thị trong batch.
        src, dst = edges[:, 0], edges[:, 1]
        hs = torch.gather(h, 1, src.unsqueeze(-1).expand(-1, -1, h.shape[-1]))
        hd = torch.gather(h, 1, dst.unsqueeze(-1).expand(-1, -1, h.shape[-1]))
        # MLP thông điệp nhận cả biểu diễn hai đầu cạnh và hai đặc trưng quan hệ SIC.
        messages = self.message(torch.cat([hs, hd, attributes], dim=-1))
        mask = edge_mask.unsqueeze(-1).to(h.dtype)
        destination = F.one_hot(dst, num_classes=h.shape[1]).to(h.dtype).transpose(1, 2)
        destination = destination * edge_mask.unsqueeze(1)
        # Lấy trung bình thông điệp đi vào từng nút, bỏ cạnh đệm; chuẩn hóa theo
        # số cạnh giúp quy mô biểu diễn không tăng đơn thuần do K lớn hơn.
        aggregate = torch.bmm(destination, messages * mask) / destination.sum(-1, keepdim=True).clamp_min(1.)
        c = context.unsqueeze(1).expand(-1, h.shape[1], -1)
        # Cập nhật residual từ trạng thái cũ, thông điệp và context toàn cục;
        # mask bảo đảm nút đệm luôn đóng góp bằng 0.
        return F.relu(h + self.update(torch.cat([h, aggregate, c], dim=-1))) * node_mask.unsqueeze(-1)


# Một bộ trọng số chung cho nhiều layout và K. K thay đổi kích thước đồ thị,
# không tạo một bộ tham số mới cho từng scenario.
class VariableAllocationGNN(nn.Module):
    """Same named weights as K=2 GNN; node/edge counts are runtime inputs."""
    def __init__(self, bounds, hidden=64, layers=3):
        super().__init__()
        self.global_encoder = dense_network([24, hidden], final_relu=True)
        self.node_encoder = dense_network([15 + hidden, hidden], final_relu=True)
        self.layers = nn.ModuleList([VariableMessageLayer(hidden) for _ in range(layers)])
        self.bob_head = dense_network([3 * hidden, hidden, 2])
        self.public_head = dense_network([4 * hidden, hidden, 1])
        self.power_head = VariablePowerHead(bounds)

    def forward(self, node, global_features, edge_index, edge_attr, node_mask=None, edge_mask=None):
        b, n, d = node.shape
        if n < 2 or d != 15 or global_features.shape != (b, 24):
            raise ValueError("Need Bob plus at least one public node and the fixed feature schema")
        if node_mask is None:
            node_mask = torch.ones((b, n), dtype=torch.bool, device=node.device)
        if edge_mask is None:
            edge_mask = torch.ones(edge_attr.shape[:2], dtype=torch.bool, device=node.device)
        if (node_mask.dtype != torch.bool or node_mask.shape != (b, n)
                or edge_mask.dtype != torch.bool or edge_mask.shape != edge_attr.shape[:2]
                or edge_index.shape != (b, 2, edge_attr.shape[1]) or edge_attr.shape[-1] != 2
                or not bool(node_mask[:, 0].all()) or not bool((node_mask[:, 1:].sum(-1) > 0).all())):
            raise ValueError("Invalid variable-graph masks or edge dimensions")
        if bool((edge_index < 0).any()) or bool((edge_index >= n).any()):
            raise ValueError("Edge endpoint outside the padded graph")
        valid_endpoints = torch.gather(node_mask, 1, edge_index[:, 0]) & torch.gather(node_mask, 1, edge_index[:, 1])
        if bool((edge_mask & ~valid_endpoints).any()):
            raise ValueError("An active edge touches a padded node")
        # Mã hóa thông tin chung như xi, tổng công suất và thống kê Willie.
        context = self.global_encoder(global_features)
        h = self.node_encoder(torch.cat([node, context.unsqueeze(1).expand(-1, n, -1)], dim=-1))
        h = h * node_mask.unsqueeze(-1)
        # Truyền thông điệp qua số lớp đã chọn; MLP baseline thay lớp này bằng
        # khối pointwise và không dùng các cạnh.
        for layer in self.layers:
            h = layer(h, context, edge_index, edge_attr, node_mask, edge_mask)
        # Pooling trung bình các nút thật tạo vector kích thước cố định cho mọi K.
        pool = h.sum(1) / node_mask.sum(1, keepdim=True)
        bob = h[:, 0]
        logits = self.bob_head(torch.cat([bob, pool, context], dim=-1))
        shared = torch.cat([h[:, 1:], bob.unsqueeze(1).expand(-1, n - 1, -1),
                            pool.unsqueeze(1).expand(-1, n - 1, -1), context.unsqueeze(1).expand(-1, n - 1, -1)], dim=-1)
        # Cùng một public_head áp cho từng user, giúp đầu ra đi theo hoán vị nút.
        public = self.public_head(shared).squeeze(-1)
        return self.power_head(logits[:, 0], public, logits[:, 1], node_mask[:, 1:])


def tensor_batch(data):
    return {k: torch.as_tensor(v, dtype=torch.long if k == "edge_index" else torch.bool
                             if k in ("node_mask", "edge_mask") else torch.float32)
            for k, v in data.items()}


# Suy luận thuần: dùng normalizer train đã lưu, tắt gradient, không chạy PSO.
def predict(model, graphs, normalizer):
    model.eval()
    data = tensor_batch(pack_graphs(graphs, normalizer))
    with torch.inference_mode():
        padded = model(**data).cpu().numpy()
    return unpack_predictions(padded, graphs)


# Loss hồi quy có trọng số cho ba nhóm mb/public/s. Nhãn là nghiệm tham chiếu
# PSO; loss này không phải xác suất đúng hoặc phép kiểm định vật lý.
def balanced_allocation_loss(prediction, target, node_mask, mb_weight=32., public_weight=2., s_weight=1.):
    """Equal case weight across K; public coefficients have fixed total weight.

    Defaults equal the previous [32,1,1,1] weighted MSE for unpadded K=2.
    Padded users have exactly zero loss and zero gradient contribution.
    """
    if (prediction.shape != target.shape or prediction.ndim != 2
            or node_mask.shape != (len(prediction), prediction.shape[1] - 1)
            or node_mask.dtype != torch.bool or not bool((node_mask[:, 1:].sum(-1) > 0).all())
            or any(not np.isfinite(w) or w <= 0 for w in (mb_weight, public_weight, s_weight))):
        raise ValueError("Invalid allocation loss dimensions, mask or weights")
    error = (prediction - target).square()
    mask = node_mask[:, 1:].to(error.dtype)
    # Lấy MSE trung bình trên public thật để tổng trọng số nhóm public
    # không tự tăng khi K lớn; ô đệm không tạo loss hoặc gradient.
    public_error = (error[:, 1:-1] * mask).sum(-1) / mask.sum(-1)
    return (mb_weight * error[:, 0] + public_weight * public_error + s_weight * error[:, -1]).mean() / (mb_weight + public_weight + s_weight)


def import_legacy_checkpoint(path):
    old, checkpoint = load_checkpoint(path)
    if checkpoint["model_kind"] != "gnn":
        raise ValueError("Only a GNN checkpoint can be imported")
    c = checkpoint["config"]
    model = VariableAllocationGNN(checkpoint["bounds"], c["gnn_hidden"], c["gnn_layers"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    return model, checkpoint, old
