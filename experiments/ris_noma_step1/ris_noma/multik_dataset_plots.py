"""Plot saved 7b dataset counts and label audits; never import Torch."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot(output):
    output = Path(output)
    plan = json.loads((output / "plan.json").read_text(encoding="utf-8"))
    result = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    if not result["complete"]:
        raise ValueError("Complete the dataset before plotting")
    rows = [json.loads(line) for line in (output / "records.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    folder = output / "figures"
    folder.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    def role(r):
        return "test_unseen_k" if r["regime"] == "unseen_k" else r["split"]
    def mean(values):
        return float(np.mean(values)) if values else np.nan
    ks = sorted(plan["config"]["training_public_counts"] + plan["config"]["unseen_public_counts"])
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, name, title in zip(axes.flat, ("train", "validation", "test", "test_unseen_k"),
                               ("Train", "Validation", "Test: K đã có trong train", "Test: K chưa có trong train")):
        present = [k for k in ks if any(role(r) == name and r["n_public"] == k for r in rows)]
        total = [sum(role(r) == name and r["n_public"] == k for r in rows) for k in present]
        labels = [sum(role(r) == name and r["n_public"] == k and r["label_available"] for r in rows) for k in present]
        ax.bar(present, total, color="#dbe5ef", label="Tình huống dự kiến")
        ax.bar(present, labels, width=.55, color="#4477aa", label="Có nhãn")
        for k, a, b in zip(present, labels, total):
            ax.text(k, b + max(total)*.025, f"{a}/{b}", ha="center", fontsize=9)
        ax.set(title=title, xlabel="K", ylabel="Số đồ thị", xticks=present, ylim=(0, max(total)*1.22))
        ax.legend(fontsize=8)
    fig.suptitle("Độ phủ nhãn nhiều K — giữ cả tình huống không có nhãn")
    fig.tight_layout(rect=(0, 0, 1, .96))
    fig.savefig(folder / "label_coverage.png", dpi=170)
    plt.close(fig)
    xis = plan["config"]["xi"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7))
    for k in ks:
        by_xi = [[r for r in rows if r["n_public"] == k and r["xi"] == xi] for xi in xis]
        availability = [100*mean([r["label_available"] for r in group]) for group in by_xi]
        passing = [100*mean([r["heldout"]["passed"] for r in group if r["label_available"]]) for group in by_xi]
        label = f"K={k}" + (" (chỉ test)" if k in plan["config"]["unseen_public_counts"] else "")
        axes[0].plot(xis, availability, "o-", label=label)
        axes[1].plot(xis, passing, "o-", label=label)
    axes[0].set(title="Có nhãn / toàn bộ tình huống", ylabel="Tỷ lệ (%)")
    axes[1].set(title="Qua heldout / số nhãn đã chọn", ylabel="Tỷ lệ (%)")
    for ax in axes:
        ax.set(xlabel="DEP yêu cầu", xticks=xis, ylim=(-3, 105))
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.text(.5, .015, "Hai biểu đồ có mẫu số khác nhau. Nhãn trượt heldout vẫn giữ trong dữ liệu; chưa đánh giá GNN.", ha="center", fontsize=9, color="#555555")
    fig.tight_layout(rect=(0, .065, 1, 1))
    fig.savefig(folder / "label_audit_by_dep.png", dpi=170)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for k in ks:
        rates = [mean([0. if not r["label_available"] else r["heldout"]["gated_bob_rate"]
                       for r in rows if r["n_public"] == k and r["xi"] == xi]) for xi in xis]
        ax.plot(xis, rates, "o-", label=f"K={k}" + (" (chỉ test)" if k in plan["config"]["unseen_public_counts"] else ""))
    ax.set(title="Rate của nhãn PSO sau kiểm định", xlabel="DEP yêu cầu", ylabel="Rate Bob bình quân (bit/s/Hz)", xticks=xis)
    ax.legend(fontsize=8)
    ax.grid(alpha=.2)
    fig.text(.5, .015, "Bình quân trên toàn bộ tình huống đã lên kế hoạch; không có nhãn hoặc trượt heldout đóng góp 0.", ha="center", fontsize=8, color="#555555")
    fig.tight_layout(rect=(0, .065, 1, 1))
    fig.savefig(folder / "label_rates.png", dpi=170)
    plt.close(fig)
    files = ["label_coverage.png", "label_audit_by_dep.png", "label_rates.png"]
    (folder / "figure_manifest.json").write_text(json.dumps({"step": "7b", "figures": files}, indent=2) + "\n", encoding="utf-8")
    return {"step": "7b", "figures": files, "gnn_trained": False}
