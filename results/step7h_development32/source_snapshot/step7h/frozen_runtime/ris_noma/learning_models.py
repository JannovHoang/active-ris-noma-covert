"""Pure PyTorch K=2 GNN and fixed-order MLP; no PyTorch Geometric needed."""
import torch
from torch import nn
from torch.nn import functional as F


class PowerHead(nn.Module):
    def __init__(self, bounds):
        super().__init__()
        self.register_buffer("intervals", torch.tensor([bounds["mb"], bounds["s"]], dtype=torch.float32))

    def forward(self, bob_logit, public_logits, s_logit):
        mb = self.intervals[0, 0] + (self.intervals[0, 1] - self.intervals[0, 0]) * torch.sigmoid(bob_logit)
        s = self.intervals[1, 0] + (self.intervals[1, 1] - self.intervals[1, 0]) * torch.sigmoid(s_logit)
        public = (1. - mb).unsqueeze(-1) * torch.softmax(public_logits, dim=-1)
        return torch.cat([mb.unsqueeze(-1), public, s.unsqueeze(-1)], dim=-1)


def dense_network(sizes, final_relu=False):
    layers = []
    for k in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[k], sizes[k+1]))
        if k < len(sizes) - 2 or final_relu:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class MessageLayer(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.message = dense_network([2 * hidden + 2, hidden, hidden], final_relu=True)
        self.update = dense_network([3 * hidden, hidden, hidden])

    def forward(self, h, context, edges, attributes):
        src, dst = edges[:, 0], edges[:, 1]
        hs = torch.gather(h, 1, src.unsqueeze(-1).expand(-1, -1, h.shape[-1]))
        hd = torch.gather(h, 1, dst.unsqueeze(-1).expand(-1, -1, h.shape[-1]))
        messages = self.message(torch.cat([hs, hd, attributes], dim=-1))
        # Fixed-size dense aggregation avoids scatter/index_add CUDA atomics.
        destination = F.one_hot(dst, num_classes=3).to(h.dtype).transpose(1, 2)
        aggregate = torch.bmm(destination, messages) / destination.sum(-1, keepdim=True).clamp_min(1.)
        c = context.unsqueeze(1).expand(-1, 3, -1)
        return F.relu(h + self.update(torch.cat([h, aggregate, c], dim=-1)))


class AllocationGNN(nn.Module):
    """Shared public head; invariant Bob/s and equivariant public allocations."""
    def __init__(self, bounds, hidden=64, layers=3):
        super().__init__()
        self.global_encoder = dense_network([24, hidden], final_relu=True)
        self.node_encoder = dense_network([15 + hidden, hidden], final_relu=True)
        self.layers = nn.ModuleList([MessageLayer(hidden) for _ in range(layers)])
        self.bob_head = dense_network([3 * hidden, hidden, 2])
        self.public_head = dense_network([4 * hidden, hidden, 1])
        self.power_head = PowerHead(bounds)

    def forward(self, node, global_features, edge_index, edge_attr):
        context = self.global_encoder(global_features)
        h = self.node_encoder(torch.cat([node, context.unsqueeze(1).expand(-1, 3, -1)], dim=-1))
        for layer in self.layers:
            h = layer(h, context, edge_index, edge_attr)
        pool, bob = h.mean(dim=1), h[:, 0]
        logits = self.bob_head(torch.cat([bob, pool, context], dim=-1))
        shared = torch.cat([h[:, 1:], bob.unsqueeze(1).expand(-1, 2, -1),
                            pool.unsqueeze(1).expand(-1, 2, -1), context.unsqueeze(1).expand(-1, 2, -1)], dim=-1)
        public = self.public_head(shared).squeeze(-1)
        return self.power_head(logits[:, 0], public, logits[:, 1])


class AllocationMLP(nn.Module):
    """Same node/global/edge information, canonical fixed original-user order."""
    def __init__(self, bounds, hidden=128, layers=3):
        super().__init__()
        self.network = dense_network([81] + [hidden] * layers + [4])
        self.power_head = PowerHead(bounds)

    def forward(self, node, global_features, edge_index, edge_attr):
        # Put each relation in its canonical source,destination slot, even if
        # the edge list is reordered. Self edges are absent and remain zero.
        slots = edge_index[:, 0] * 3 + edge_index[:, 1]
        destination = F.one_hot(slots, num_classes=9).to(edge_attr.dtype).transpose(1, 2)
        dense = torch.bmm(destination, edge_attr)
        relation = dense[:, [1, 2, 3, 5, 6, 7]].flatten(start_dim=1)
        x = torch.cat([node.flatten(start_dim=1), global_features, relation], dim=-1)
        raw = self.network(x)
        return self.power_head(raw[:, 0], raw[:, 1:3], raw[:, 3])


def make_model(kind, bounds, config):
    if kind == "gnn":
        return AllocationGNN(bounds, config["gnn_hidden"], config["gnn_layers"])
    if kind == "mlp":
        return AllocationMLP(bounds, config["mlp_hidden"], config["mlp_layers"])
    raise ValueError("model must be gnn or mlp")
