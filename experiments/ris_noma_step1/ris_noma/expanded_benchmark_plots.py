"""Figures for step 6c, with a plotting process that does not import Torch."""
from pathlib import Path

from filelock import FileLock
import numpy as np

from .benchmark_plots import COLORS, LABELS, MARKERS, METHODS, plt, read_json, read_results, save_figure
from .expanded_data import save_json


def overview(summary, folder):
    table = summary["summary"]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.5))
    panels = (
        ("Tỷ lệ đạt kiểm định vật lý", "Tỷ lệ (%)",
         [100 * row["bound_pass_fraction"] for row in table],
         [f"{row['bound_pass_count']}/{row['runs']}" for row in table]),
        ("Tốc độ Bob có tính lượt trượt", "bit/s/Hz",
         [row["audit_gated_bob_rate_mean"] for row in table],
         [f"{row['audit_gated_bob_rate_mean']:.3f}" for row in table]),
        ("Có nghiệm khả thi ở lượt quét đầu", "Tỷ lệ (%)",
         [100 * row["initial_swarm_pass_count"] / row["runs"] for row in table],
         [f"{row['initial_swarm_pass_count']}/{row['runs']}" for row in table]),
        ("Ngân sách tìm kiếm mỗi lượt", "Số lần đánh giá phương án",
         [row["evaluations_mean"] for row in table],
         [f"{row['evaluations_mean']:.0f}" for row in table]),
    )
    for ax, (title, unit, values, annotations) in zip(axes.flat, panels):
        bars = ax.barh([LABELS[m] for m in METHODS], values, color=[COLORS[m] for m in METHODS], height=.62)
        maximum = max(values) or 1.
        ax.bar_label(bars, labels=annotations, padding=4, fontsize=9)
        ax.set(title=title, xlabel=unit, xlim=(0, maximum * 1.25))
        if unit == "Tỷ lệ (%)":
            ax.set(xlim=(0, 114), xticks=range(0, 101, 20))
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=.2)
        ax.set_axisbelow(True)
    runs, layouts = table[0]["runs"], table[0]["layouts"]
    save_figure(fig, folder / "overview.png",
                f"Đợt chạy thử trên {layouts} bố trí, {runs} lượt mỗi phương pháp. Quy ước tốc độ bằng 0 khi trượt kiểm định.")


def by_power(rows, folder):
    split = "test" if any(row["split"] == "test" for row in rows) else rows[0]["split"]
    focus = [row for row in rows if row["split"] == split]
    powers = sorted({row["total_power_dbm"] for row in focus})
    xis = sorted({row["xi"] for row in focus})
    fig, axes = plt.subplots(len(powers), 2, figsize=(12, 2.7 * len(powers) + .7), squeeze=False)
    for power, (rate_ax, pass_ax) in zip(powers, axes):
        counts = np.zeros((len(METHODS), len(xis)), int)
        sizes = np.zeros_like(counts)
        for index, method in enumerate(METHODS):
            groups = [[row for row in focus if row["total_power_dbm"] == power
                       and row["xi"] == xi and row["method"] == method] for xi in xis]
            counts[index] = [sum(row["bound_pass"] for row in group) for group in groups]
            sizes[index] = [len(group) for group in groups]
            rate_ax.plot(xis, [np.mean([row["audit_gated_bob_rate"] for row in group]) for group in groups],
                         color=COLORS[method], marker=MARKERS[method], fillstyle="none", label=LABELS[method])
        rate_ax.set(title=f"{power:g} dBm · Tốc độ Bob", ylabel="bit/s/Hz", xticks=xis)
        rate_ax.grid(alpha=.2)
        pass_ax.imshow(counts / sizes, cmap="Blues", vmin=0., vmax=1., aspect="auto")
        pass_ax.set(title=f"{power:g} dBm · Số lượt qua kiểm định",
                    xticks=range(len(xis)), xticklabels=[f"{xi:.2f}" for xi in xis],
                    yticks=range(len(METHODS)), yticklabels=[LABELS[m] for m in METHODS])
        for index, column in np.ndindex(counts.shape):
            count, size = counts[index, column], sizes[index, column]
            pass_ax.text(column, index, f"{count}/{size}", ha="center", va="center",
                         color="white" if count / size > .55 else "#1f2c37")
    axes[0, 0].legend(fontsize=7, ncol=2, loc="best")
    for ax in axes[-1]:
        ax.set_xlabel("DEP yêu cầu")
    layouts = len({row["layout_id"] for row in focus})
    save_figure(fig, folder / "rate_and_feasibility_by_power.png",
                f"Tập {split} gồm {layouts} bố trí. Mỗi ô tổng hợp các seed PSO. Quy ước tốc độ bằng 0 cho lượt trượt.")


def convergence(searches, folder):
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.4))
    common = min(row["evaluations"] for row in searches)
    for method in METHODS:
        runs = [[point for point in row["history"] if point["evaluations"] <= common]
                for row in searches if row["method"] == method]
        ticks = [point["evaluations"] for point in runs[0]]
        feasible = np.array([[point["gbest_feasible"] for point in run] for run in runs])
        rates = np.array([[point["gbest_score"] for point in run] for run in runs])
        for ax, values in zip(axes, (100 * feasible.mean(axis=0), (rates * feasible).mean(axis=0))):
            ax.plot(ticks, values, label=LABELS[method], color=COLORS[method],
                    marker=MARKERS[method], fillstyle="none", markersize=5)
    axes[0].set(title="Đã tìm được nghiệm khả thi", ylabel="Tỷ lệ lượt (%)")
    axes[0].set_ylim(bottom=max(0., axes[0].get_ylim()[0]), top=101.)
    axes[1].set(title="Tốc độ tốt nhất tìm được", ylabel="bit/s/Hz")
    for ax in axes:
        ax.set_xlabel("Số lần đánh giá phương án")
        ax.grid(alpha=.2)
        ax.legend(fontsize=8, loc="best")
    save_figure(fig, folder / "convergence.png",
                "Đo trên bank tìm kiếm trong ngân sách chung. Tốc độ tính bằng 0 đến khi tìm được nghiệm khả thi.")


def timing(summary, folder):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    table = summary["summary"]
    names = [LABELS[m] for m in METHODS]
    search = np.array([row["search_seconds_mean"] for row in table])
    audit = np.array([row["audit_seconds_mean"] for row in table])
    setup = np.array([row["standalone_seconds_mean"] - row["total_seconds_mean"] for row in table])
    for index, (ax, title) in enumerate(zip(axes, ("Dùng bank đã dựng", "Ước tính nếu dựng lại bank"))):
        ax.barh(names, search, color="#24557c", label="Tìm nghiệm")
        ax.barh(names, audit, left=search, color="#24805d", label="Chấm nghiệm")
        if index == 1:
            ax.barh(names, setup, left=search + audit, color="#ca7b27", label="Dựng bank và bộ tính trước")
            ax.set_xlim(0, 1.2 * max(search + audit + setup))
        ax.set(title=title, xlabel="Thời gian trung bình mỗi lượt (s)")
        ax.invert_yaxis()
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(axis="x", alpha=.2)
        ax.set_axisbelow(True)
    save_figure(fig, folder / "time_and_speedup.png",
                "Chi phí tìm nghiệm gồm trích đặc trưng, suy luận và PSO khi áp dụng. Thời gian huấn luyện được tính riêng ở 6b.")


def plot_benchmark(output):
    output = Path(output).resolve()
    searches, rows = read_results(output)
    summary = read_json(output / "benchmark_summary.json")
    if summary["step"] != "6c":
        raise ValueError("Need a completed step 6c benchmark")
    folder = output / "figures"
    with FileLock(str(output.with_name(f".{output.name}.plot.lock")), timeout=0):
        folder.mkdir(exist_ok=True)
        plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
        overview(summary, folder)
        by_power(rows, folder)
        convergence(searches, folder)
        timing(summary, folder)
        result = {"step": "6c", "scope": summary["scope"],
                  "figures": ["overview.png", "rate_and_feasibility_by_power.png", "convergence.png", "time_and_speedup.png"]}
        save_json(folder / "figure_manifest.json", result)
    return result
