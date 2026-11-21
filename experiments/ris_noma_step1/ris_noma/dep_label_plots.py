"""Plot step 6e summaries without loading Torch."""
from pathlib import Path

from filelock import FileLock

from .benchmark_plots import plt, read_json, save_figure
from .expanded_data import save_json
from .learning_data import sha256


METHODS = {
    "gnn_6b": ("GNN 6b", "#8594a5"),
    "standard_direct_mean": ("GNN nhãn chuẩn", "#4177a4"),
    "robust_direct_mean": ("GNN nhãn có dư", "#23856a"),
    "pso_short": ("PSO ngắn ngẫu nhiên", "#ca7b27"),
    "standard_pso_short": ("GNN nhãn chuẩn + PSO", "#7163a5"),
    "robust_pso_short": ("GNN nhãn có dư + PSO", "#b2537e"),
}


def plot_coverage(labels, folder):
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2))
    rows = [row for row in labels["coverage"] if row["split"] == "train"]
    slacks = sorted({row["dep_slack"] for row in rows})
    for slack in slacks:
        group = [row for row in rows if row["dep_slack"] == slack]
        xs = [row["xi"] for row in group]
        coverage = [100 * row["labels"] / row["eligible_cases"] if row["eligible_cases"] else None for row in group]
        axes[0].plot(xs, coverage, marker="o", label=f"δ = {slack:g}")
        axes[1].plot(xs, [row["bob_rate_delta_on_matched_labels"] for row in group], marker="o")
    axes[0].set(title="Độ phủ nhãn tập train", ylabel="Tỷ lệ có nhãn (%)", ylim=(0, 104))
    axes[1].set(title="Tốc độ Bob so với nhãn chuẩn", ylabel="Chênh lệch (bit/s/Hz)")
    for ax in axes:
        ax.set(xlabel="DEP yêu cầu", xticks=xs)
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=9)
    save_figure(fig, folder / "label_coverage.png",
                "Hai cách chọn nhãn dùng cùng ứng viên PSO. Chênh lệch tốc độ tính trên các tình huống có đủ nhãn.")


def plot_comparison(result, folder):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.6), sharey=True)
    table = {row["method"]: row for row in result["methods"]}
    labels, colors = zip(*METHODS.values())
    for ax, (title, unit, key, scale) in zip(axes, (
        ("Qua kiểm định độc lập", "Tỷ lệ (%)", "pass_fraction", 100.),
        ("Tốc độ Bob có tính lượt trượt", "bit/s/Hz", "gated_bob_rate", 1.),
        ("Thời gian trả nghiệm", "ms/lượt", "search_wall_ms", 1.),
    )):
        values = [table[name][key] * scale for name in METHODS]
        bars = ax.barh(labels, values, color=colors, height=.63)
        ax.set(title=title, xlabel=unit, xlim=(0, max(values, default=1.) * 1.25 or 1.))
        ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=8)
        ax.grid(axis="x", alpha=.2)
        ax.set_axisbelow(True)
    axes[0].invert_yaxis()
    save_figure(fig, folder / "test_comparison.png",
                f"Test trên {result['test_layouts']} bố trí mới. "
                "Tốc độ tính bằng 0 cho lượt trượt. Thời gian đo với bank tìm kiếm đã dựng.")


def plot_by_xi(result, folder):
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.2))
    methods = ("standard_direct_mean", "robust_direct_mean", "pso_short", "robust_pso_short")
    for method in methods:
        group = [row for row in result["by_xi"] if row["method"] == method]
        name, color = METHODS[method]
        xs = [row["xi"] for row in group]
        axes[0].plot(xs, [100 * row["pass_fraction"] for row in group], color=color, marker="o", label=name)
        axes[1].plot(xs, [row["gated_bob_rate"] for row in group], color=color, marker="o")
    axes[0].set(title="Qua kiểm định theo ngưỡng DEP", ylabel="Tỷ lệ (%)", ylim=(0, 104))
    axes[1].set(title="Tốc độ Bob có tính lượt trượt", ylabel="bit/s/Hz")
    for ax in axes:
        ax.set(xlabel="DEP yêu cầu", xticks=xs)
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8, loc="best")
    save_figure(fig, folder / "test_by_dep.png",
                "GNN trực tiếp tổng hợp theo ba seed huấn luyện. Các nhánh PSO dùng cùng ngân sách và ba seed tìm kiếm.")


def plot_experiment(output):
    output = Path(output).resolve()
    hashes = read_json(output / "completion.json")["sha256"]
    for name in ("summary.json", "label_summary.json"):
        if sha256(output / name) != hashes[name]:
            raise ValueError(f"Summary changed: {name}")
    result = read_json(output / "summary.json")
    labels = read_json(output / "label_summary.json")
    if result["step"] != "6e" or not result["complete"]:
        raise ValueError("Need a completed step 6e result")
    folder = output / "figures"
    manifest = {"step": "6e", "figures": ["label_coverage.png", "test_comparison.png", "test_by_dep.png"]}
    with FileLock(str(output.with_name(f".{output.name}.plot.lock")), timeout=0):
        folder.mkdir(exist_ok=True)
        plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                            "axes.spines.top": False, "axes.spines.right": False})
        plot_coverage(labels, folder)
        plot_comparison(result, folder)
        plot_by_xi(result, folder)
        save_json(folder / "figure_manifest.json", manifest)
    return manifest
