"""Plot numerical step 6d outputs without importing Torch."""
from pathlib import Path

from filelock import FileLock

from .benchmark_plots import COLORS, LABELS, MARKERS, plt, read_json, save_figure
from .budget_analysis import METHODS
from .expanded_data import save_json
from .learning_data import sha256


def curves(summary, folder):
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2))
    panels = (
        ("Qua kiểm định độc lập", "Tỷ lệ (%)", "audit_pass_fraction", 100.),
        ("Tốc độ Bob có tính lượt trượt", "bit/s/Hz", "gated_bob_rate_mean", 1.),
        ("Thời gian tìm nghiệm", "ms/lượt", "search_wall_seconds_mean", 1000.),
    )
    for ax, (title, ylabel, key, scale) in zip(axes, panels):
        for method in METHODS:
            rows = [row for row in summary["curves"] if row["method"] == method]
            budgets = [row["budget"] for row in rows]
            ax.plot(budgets, [row[key] * scale for row in rows], color=COLORS[method],
                    marker=MARKERS[method], label=LABELS[method])
        ax.set(title=title, xlabel="Số đánh giá PSO", ylabel=ylabel, xticks=budgets)
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8, loc="best")
    save_figure(fig, folder / "budget_curves.png",
                f"Kiểm định trên {summary['eligible_layouts']} bố trí mới đủ điều kiện. "
                "Tốc độ tính bằng 0 cho lượt trượt. Thời gian đo với bank đã dựng.")


def paired_differences(summary, folder):
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.2))
    rows = [row for row in summary["comparisons"] if row["gnn_budget"] == row["random_budget"]]
    budgets = [row["gnn_budget"] for row in rows]
    primary = summary["primary"]["gnn_budget"]
    for ax, (title, ylabel, key) in zip(axes, (
        ("Chênh lệch tỷ lệ qua kiểm định", "Điểm phần trăm", "pass_delta_pp"),
        ("Chênh lệch tốc độ Bob có tính lượt trượt", "bit/s/Hz", "gated_rate_delta"),
    )):
        ax.axhline(0., color="#475569", linewidth=.9, linestyle="--")
        ax.axvline(primary, color="#cbd5e1", linewidth=1.5)
        ax.plot(budgets, [row[key] for row in rows], marker="o", color=COLORS["gnn_pso_short"])
        for x, row in zip(budgets, rows):
            lo, hi = row[f"{key}_ci_low"], row[f"{key}_ci_high"]
            if lo is not None:
                ax.vlines(x, lo, hi, color=COLORS["gnn_pso_short"], alpha=.6, linewidth=2.)
                ax.scatter([x, x], [lo, hi], marker="_", color=COLORS["gnn_pso_short"], s=45)
        ax.set(title=title, xlabel="Ngân sách chung", ylabel=ylabel, xticks=budgets)
        ax.grid(axis="y", alpha=.2)
    save_figure(fig, folder / "paired_differences.png",
                f"GNN + PSO trừ PSO ngẫu nhiên. Khoảng bootstrap 95% theo bố trí. Mốc chính là {primary} đánh giá.")


def plot_budget(output):
    output = Path(output).resolve()
    expected = read_json(output / "completion.json")["sha256"]["summary.json"]
    path = output / "summary.json"
    if sha256(path) != expected:
        raise ValueError("Summary changed after the benchmark")
    summary = read_json(path)
    if summary["step"] != "6d" or not summary["complete"]:
        raise ValueError("Need a completed step 6d benchmark")
    folder = output / "figures"
    result = {"step": "6d", "figures": ["budget_curves.png", "paired_differences.png"]}
    with FileLock(str(output.with_name(f".{output.name}.plot.lock")), timeout=0):
        folder.mkdir(exist_ok=True)
        plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                            "axes.spines.top": False, "axes.spines.right": False})
        curves(summary, folder)
        paired_differences(summary, folder)
        save_json(folder / "figure_manifest.json", result)
    return result
