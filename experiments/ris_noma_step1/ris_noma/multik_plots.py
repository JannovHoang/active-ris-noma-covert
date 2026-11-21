"""Read saved 7a records and plot without importing Torch in this process."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot(output):
    output = Path(output)
    result = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    folder = output / "figures"; folder.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    shapes = result["model_shapes"]
    by_k = {r["n_public"]: r for r in shapes}
    ks = sorted(by_k)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(ks, [by_k[k]["nodes"] for k in ks], "o-", label="Số nút người dùng")
    axes[0].plot(ks, [by_k[k]["outputs"] for k in ks], "s-", label="Số hệ số đầu ra")
    axes[0].set(title="Kích thước đầu vào/đầu ra thay đổi", xlabel="Số người dùng công khai K", ylabel="Số lượng", xticks=ks)
    axes[0].legend()
    axes[1].bar(ks, [by_k[k]["parameters"] for k in ks], color="#4477aa")
    axes[1].set(title="Cùng một mạng, cùng số tham số", xlabel="K", ylabel="Số tham số học được", xticks=ks)
    for k in ks:
        axes[1].text(k, by_k[k]["parameters"] * 1.02, str(by_k[k]["parameters"]), ha="center", fontsize=9)
    axes[1].set_ylim(0, max(r["parameters"] for r in shapes) * 1.2)
    fig.text(.5, .015, "Kiểm chứng kích thước và dùng chung trọng số; chưa chứng minh chất lượng dự đoán ngoài phạm vi train.", ha="center", fontsize=9, color="#555555")
    fig.tight_layout(rect=(0, .065, 1, 1)); fig.savefig(folder / "shared_model.png", dpi=170); plt.close(fig)
    with (output / "training_history.csv").open(encoding="utf-8-sig") as f:
        history = list(csv.DictReader(f))
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for key, label in (("train_loss", "Train"), ("validation_loss", "Validation")):
        ax.plot([int(r["epoch"]) for r in history], [float(r[key]) for r in history], label=label)
    ax.axvline(result["shared_training"]["selected_epoch"], linestyle="--", color="#777777", label="Checkpoint được chọn")
    ax.set(title="Lần train nhỏ để kiểm tra quy trình nhiều K", xlabel="Epoch", ylabel="Loss phân bổ công suất", yscale="log")
    ax.legend(); ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(folder / "training_demo.png", dpi=170); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    coverage = result["coverage"]
    train_ks = sorted({r["n_public"] for r in coverage})
    for offset, split in enumerate(("train", "validation", "test")):
        rr = {r["n_public"]: r for r in coverage if r["split"] == split}
        x = np.arange(len(train_ks)) + (offset - 1) * .25
        axes[0].bar(x, [rr[k]["labels"] for k in train_ks], width=.25, label=split)
    axes[0].set(title="Số đồ thị có nhãn trong bộ chạy nhỏ", xlabel="K", ylabel="Số đồ thị", xticks=np.arange(len(train_ks)), xticklabels=train_ks)
    axes[0].set_ylim(0, max(r["labels"] for r in coverage) * 1.35)
    axes[0].legend()
    physical = result["smoke_physics"]
    gnn = {r["n_public"]: r for r in physical if r["method"] == "shared_gnn_demo"}
    axes[1].bar(ks, [gnn[k]["passed"] for k in ks], color=["#4477aa" if not gnn[k]["unseen_K"] else "#ddaa33" for k in ks])
    for k in ks:
        axes[1].text(k, gnn[k]["passed"] + .08, f"{gnn[k]['passed']}/{gnn[k]['cases']}", ha="center")
    axes[1].set(title="Kiểm định vật lý của GNN chạy thử", xlabel="K (màu vàng: chưa train)", ylabel="Số tình huống đạt", xticks=ks,
                ylim=(0, max(r["cases"] for r in gnn.values()) + 1))
    fig.text(.5, .015, "Bộ thử rất nhỏ; hình này mô tả quy trình và số lượt thực tế, không phải bảng chứng minh ưu thế phương pháp.", ha="center", fontsize=9, color="#555555")
    fig.tight_layout(rect=(0, .065, 1, 1)); fig.savefig(folder / "coverage_and_physics_demo.png", dpi=170); plt.close(fig)
    files = ["shared_model.png", "training_demo.png", "coverage_and_physics_demo.png"]
    (folder / "figure_manifest.json").write_text(json.dumps({"step": "7a", "figures": files}, indent=2) + "\n")
    return {"figures": files, "foundation_only": True}
