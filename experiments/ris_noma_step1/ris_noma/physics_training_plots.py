"""Read-only plots for the 6f ablation; no Torch import in the plotting process."""
import csv
from pathlib import Path

from filelock import FileLock

from .benchmark_plots import plt, save_figure
from .expanded_data import save_json
from .learning_data import read_json, sha256


LABELS = {
    "uniform_mse": ("MSE + chọn bằng MSE", "#4177a4"),
    "uniform_physics": ("MSE + chọn bằng vật lý", "#23856a"),
    "weighted_mse": ("MSE trọng số + chọn bằng MSE", "#b68830"),
    "weighted_physics": ("MSE trọng số + chọn bằng vật lý", "#875aa2"),
    "pso_short": ("PSO ngắn ngẫu nhiên", "#ca7b27"),
    "baseline_pso_short": ("GNN mốc + PSO ngắn", "#8594a5"),
    "selected_pso_short": ("GNN chọn trên validation + PSO ngắn", "#b2537e"),
}


def comparison(result, folder):
    table = {r["method"]: r for r in result["methods"]}
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.1), sharey=True)
    labels, colors = zip(*LABELS.values())
    for ax, title, key, factor, unit in (
        (axes[0], "Qua kiểm định độc lập", "pass_fraction", 100., "%"),
        (axes[1], "Tốc độ Bob có tính lượt trượt", "gated_bob_rate", 1., "bit/s/Hz"),
        (axes[2], "Thời gian trả nghiệm", "search_wall_ms", 1., "ms/lượt")):
        values = [table[m][key] * factor for m in LABELS]
        bars = ax.barh(labels, values, color=colors, height=.64)
        ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=8)
        ax.set(title=title, xlabel=unit, xlim=(0, max(values, default=1.) * 1.25 or 1.))
        ax.grid(axis="x", alpha=.2)
        ax.set_axisbelow(True)
    axes[0].invert_yaxis()
    save_figure(fig, folder / "test_comparison.png",
        f"{result['test_layouts']} bố trí mới; bốn nhóm GNN dùng cùng nhãn có dư. "
        "Lượt trượt tính tốc độ bằng 0. Thời gian với bank và cache đã dựng.")


def by_dep(result, folder):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for method in ("uniform_mse", "uniform_physics", "weighted_mse", "weighted_physics", "pso_short", "selected_pso_short"):
        rows = [r for r in result["by_xi"] if r["method"] == method]
        label, color = LABELS[method]
        xs = [r["xi"] for r in rows]
        axes[0].plot(xs, [100 * r["pass_fraction"] for r in rows], marker="o", label=label, color=color)
        axes[1].plot(xs, [r["gated_bob_rate"] for r in rows], marker="o", color=color)
    axes[0].set(title="Qua kiểm định theo DEP", ylabel="Tỷ lệ (%)", ylim=(0, 104))
    axes[1].set(title="Tốc độ Bob có tính lượt trượt", ylabel="bit/s/Hz")
    for ax in axes:
        ax.set(xlabel="DEP yêu cầu", xticks=xs)
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=7.5, loc="best")
    save_figure(fig, folder / "test_by_dep.png", "Mỗi ngưỡng DEP là một đầu vào khác. GNN tổng hợp ba seed huấn luyện; PSO tổng hợp ba seed tìm kiếm.")


def checkpoint_selection(output, selection, folder):
    names = list(selection["selected"])
    scores = {r["candidate"]: r for r in selection["candidate_scores"]}
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6), sharey=True)
    labels, passes, errors, colors = [], [], [], []
    for name in names:
        ref = selection["selected"][name]
        score = scores[ref["candidate"]]
        short = {"uniform_mse": "MSE / MSE", "uniform_physics": "MSE / vật lý",
                 "weighted_mse": "Trọng số / MSE", "weighted_physics": "Trọng số / vật lý"}[ref["group"]]
        labels.append(f"{short}, seed {ref['seed']} (epoch {ref['epoch']})")
        passes.append(100 * score["passed"] / score["runs"])
        errors.append(ref["validation_mse"])
        colors.append(LABELS[ref["group"]][1])
    for ax, values, title, unit in ((axes[0], passes, "Kiểm định vật lý validation", "Tỷ lệ đạt (%)"),
                                   (axes[1], errors, "Sai số dự đoán validation", "MSE")):
        bars = ax.barh(labels, values, color=colors, height=.66)
        ax.bar_label(bars, fmt="%.3g", padding=3, fontsize=8)
        ax.set(title=title, xlabel=unit, xlim=(0, max(values, default=1.) * 1.25 or 1.))
        ax.grid(axis="x", alpha=.2)
        ax.set_axisbelow(True)
    axes[0].invert_yaxis()
    save_figure(fig, folder / "validation_selection.png", "Checkpoint được chọn bằng validation. Kiểm định vật lý là đánh giá xuôi, không truyền gradient qua mô phỏng.")


def learning_curves(output, plan, folder):
    seeds = plan["training_config"]["seeds"]
    fig, axes = plt.subplots(1, len(seeds), figsize=(4.2 * len(seeds), 3.9), squeeze=False)
    for ax, seed in zip(axes[0], seeds):
        for loss, color in (("uniform", "#4177a4"), ("weighted", "#875aa2")):
            with (output / f"fits/{loss}_s{seed}/history.csv").open(encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))
            xs = [int(r["epoch"]) for r in rows]
            ax.plot(xs, [float(r["train_mse"]) for r in rows], color=color, linestyle="--", label=f"{loss}: train")
            ax.plot(xs, [float(r["validation_mse"]) for r in rows], color=color, label=f"{loss}: validation")
        ax.set(title=f"Seed {seed}", xlabel="Epoch", ylabel="MSE không trọng số", yscale="log")
        ax.grid(alpha=.2)
        ax.legend(fontsize=7)
    save_figure(fig, folder / "learning_curves.png", "Hai loss dùng cùng khởi tạo theo seed, cùng batch và cùng số epoch. MSE trên hình giữ cùng định nghĩa để so sánh.")


def plot_experiment(output):
    output = Path(output).resolve()
    hashes = read_json(output / "completion.json")["sha256"]
    for name in ("summary.json", "selection_frozen.json", "plan.json"):
        if sha256(output / name) != hashes[name]:
            raise ValueError(f"Step 6f numerical output changed: {name}")
    result, selection, plan = (read_json(output / n) for n in ("summary.json", "selection_frozen.json", "plan.json"))
    if result["step"] != "6f" or not result["complete"]:
        raise ValueError("Need a completed step 6f run")
    folder = output / "figures"
    manifest = {"step": "6f", "figures": ["test_comparison.png", "test_by_dep.png", "validation_selection.png", "learning_curves.png"]}
    with FileLock(str(output.with_name(f".{output.name}.plot.lock")), timeout=0):
        folder.mkdir(exist_ok=True)
        plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                            "axes.spines.top": False, "axes.spines.right": False})
        comparison(result, folder)
        by_dep(result, folder)
        checkpoint_selection(output, selection, folder)
        learning_curves(output, plan, folder)
        save_json(folder / "figure_manifest.json", manifest)
    return manifest
