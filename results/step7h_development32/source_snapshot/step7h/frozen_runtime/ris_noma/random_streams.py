"""Luồng ngẫu nhiên theo vai trò; không dùng np.random.seed toàn cục."""
import numpy as np

STREAM = {"geometry": 0, "label": 1, "evaluation": 2, "pso": 3, "training": 4}


def make_rng(root_seed, layout_id, role):
    # Giữ role code cố định để thêm thuật toán mới không đổi channel bank cũ.
    sequence = np.random.SeedSequence([root_seed, layout_id, STREAM[role]])
    return np.random.default_rng(sequence)
