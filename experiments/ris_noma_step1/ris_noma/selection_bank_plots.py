"""Read-only scientific plots; isolated from Torch in a separate process."""
from pathlib import Path

from filelock import FileLock

from .benchmark_plots import plt, save_figure
from .expanded_data import save_json
from .learning_data import read_json, sha256


LABELS = {
    "gnn_raw": ("GNN trực tiếp", "#4177a4"),
    "gnn_screened": ("GNN qua bank lựa chọn", "#5d9fba"),
    "cold_raw": ("PSO ngẫu nhiên: chọn cũ", "#ca7b27"),
    "cold_screened": ("PSO ngẫu nhiên: chọn mới", "#e3aa4b"),
    "warm_raw": ("GNN + PSO: chọn cũ", "#7163a5"),
    "warm_screened": ("GNN + PSO: chọn mới", "#23856a"),
    "full_raw": ("PSO đầy đủ: chọn cũ", "#8594a5"),
    "full_screened": ("PSO đầy đủ: chọn mới", "#b2537e"),
}


def overview(result, folder):
    table = {r["method"]: r for r in result["primary_methods"]}
    labels, colors = zip(*LABELS.values())
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.7), sharey=True)
    for ax, title, key, factor, unit in (
        (axes[0], "Trả nghiệm đạt kiểm định", "pass_fraction", 100., "% tổng lượt"),
        (axes[1], "Tốc độ Bob có tính lượt trượt", "gated_bob_rate", 1., "bit/s/Hz"),
        (axes[2], "Tìm và lựa chọn nghiệm", "online_ms", 1., "ms, bank đã dựng")):
        values = [table[m][key] * factor for m in LABELS]
        bars = ax.barh(labels, values, color=colors, height=.65)
        ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=8)
        ax.set(title=title, xlabel=unit, xlim=(0, 108 if key == "pass_fraction" else max(values, default=1.) * 1.25 or 1.))
        ax.grid(axis="x", alpha=.2)
        ax.set_axisbelow(True)
    axes[0].invert_yaxis()
    save_figure(fig, folder / "overview.png", f"{result['test_layouts']} bố trí mới; PSO ngắn {result['primary_budget']} lần chấm, "
        f"PSO đầy đủ {result['full_evaluations']}. Không trả nghiệm cũng tính là không đạt. Thời gian gồm chấm bank lựa chọn.")


def coverage(result, folder):
    table = {r["method"]: r for r in result["primary_methods"]}
    labels = [LABELS[m][0] for m in LABELS]
    fig, ax = plt.subplots(figsize=(10, 5.3))
    passed = [100 * table[m]["passed"] / table[m]["runs"] for m in LABELS]
    failed = [100 * table[m]["audit_failed_after_return"] / table[m]["runs"] for m in LABELS]
    abstained = [100 * table[m]["abstained"] / table[m]["runs"] for m in LABELS]
    ax.barh(labels, passed, color="#23856a", label="Trả nghiệm và đạt")
    ax.barh(labels, failed, left=passed, color="#be655e", label="Trả nghiệm nhưng trượt")
    ax.barh(labels, abstained, left=[a + b for a, b in zip(passed, failed)], color="#b7bec6", label="Không trả nghiệm")
    ax.set(xlabel="Tỷ lệ trên toàn bộ lượt (%)", xlim=(0, 100), title="Độ phủ và kết quả kiểm định")
    ax.invert_yaxis()
    ax.legend(loc="lower center", bbox_to_anchor=(.5, -.25), ncol=3, fontsize=8)
    save_figure(fig, folder / "coverage.png", "Tỷ lệ đạt chính giữ mọi lượt trong mẫu số; không dùng tỷ lệ có điều kiện để che các lượt không trả nghiệm.")


def budgets(result, folder):
    rows = result["methods"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
    for method in ("cold_raw", "cold_screened", "warm_raw", "warm_screened"):
        group = sorted([r for r in rows if r["method"] == method], key=lambda r: r["budget"])
        label, color = LABELS[method]
        xs = [r["budget"] for r in group]
        for ax, key, factor in ((axes[0], "pass_fraction", 100.), (axes[1], "gated_bob_rate", 1.), (axes[2], "online_ms", 1.)):
            ax.plot(xs, [factor * r[key] for r in group], marker="o", label=label, color=color)
            ax.set(xticks=xs, xlabel="Số lần đánh giá mục tiêu PSO")
            ax.grid(alpha=.2)
    axes[0].set(title="Trả nghiệm đạt kiểm định", ylabel="% tổng lượt", ylim=(0, 104))
    axes[1].set(title="Tốc độ Bob có tính lượt trượt", ylabel="bit/s/Hz")
    axes[2].set(title="Tìm và lựa chọn nghiệm", ylabel="ms, bank đã dựng")
    axes[0].legend(fontsize=7)
    save_figure(fig, folder / "budget_curves.png", "Các ngân sách ngắn lấy trên cùng một quỹ đạo PSO, không cộng các ngân sách lại thành chi phí thực chạy.")


def times(result, folder):
    table = {r["method"]: r for r in result["primary_methods"]}
    labels = [LABELS[m][0] for m in LABELS]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.7), sharey=True)
    a = [table[m]["search_ms"] for m in LABELS]
    b = [table[m]["selection_pool_ms"] for m in LABELS]
    axes[0].barh(labels, a, color="#4177a4", label="GNN/PSO + lưu ứng viên")
    axes[0].barh(labels, b, left=a, color="#b2537e", label="Chấm bank lựa chọn")
    axes[0].set(title="Bank đã dựng", xlabel="ms/quyết định")
    axes[0].legend(fontsize=7)
    left = [0.] * len(labels)
    for key, label, color in (("online_ms", "Tìm + chọn", "#4177a4"), ("search_bank_ms", "Dựng bank tìm", "#ca7b27"),
            ("selection_bank_ms", "Dựng bank lựa chọn", "#b2537e"), ("objective_cache_ms", "Cache PSO", "#8594a5")):
        values = [table[m][key] for m in LABELS]
        axes[1].barh(labels, values, left=left, label=label, color=color)
        left = [a + b for a, b in zip(left, values)]
    axes[1].set(title="Ước tính chạy riêng từng quyết định", xlabel="ms, tính đầy đủ dựng bank")
    axes[1].legend(fontsize=7)
    axes[0].invert_yaxis()
    for ax in axes:
        ax.grid(axis="x", alpha=.2)
        ax.set_axisbelow(True)
    save_figure(fig, folder / "latency_breakdown.png", "Ước tính chạy riêng cộng các thành phần đo riêng, tính đủ bank lựa chọn; không gồm train/nạp model và audit nghiên cứu cuối.")


def by_dep(result, folder):
    primary, full = result["primary_budget"], result["full_evaluations"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for method, budget in (("gnn_raw", 0), ("cold_screened", primary), ("warm_raw", primary),
                           ("warm_screened", primary), ("full_screened", full)):
        group = sorted([r for r in result["by_xi"] if r["method"] == method and r["budget"] == budget], key=lambda r: r["xi"])
        label, color = LABELS[method]
        xs = [r["xi"] for r in group]
        axes[0].plot(xs, [100 * r["pass_fraction"] for r in group], marker="o", color=color, label=label)
        axes[1].plot(xs, [r["gated_bob_rate"] for r in group], marker="o", color=color)
    axes[0].set(title="Trả nghiệm đạt theo DEP", ylabel="% tổng lượt", ylim=(0, 104))
    axes[1].set(title="Tốc độ Bob có tính lượt trượt", ylabel="bit/s/Hz")
    for ax in axes:
        ax.set(xlabel="DEP yêu cầu", xticks=xs)
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=7)
    save_figure(fig, folder / "test_by_dep.png", "Bố trí và bank audit giống nhau giữa các phương pháp. Quy tắc chọn nghiệm được cố định trước bank audit.")


def plot_experiment(output):
    output = Path(output).resolve()
    hashes = read_json(output / "completion.json")["sha256"]
    if sha256(output / "summary.json") != hashes["summary.json"]:
        raise ValueError("Summary changed after completion")
    result = read_json(output / "summary.json")
    if result["step"] != "6g" or not result["complete"]:
        raise ValueError("Need completed 6g numerical results")
    folder = output / "figures"
    manifest = {"step": "6g", "figures": ["overview.png", "coverage.png", "budget_curves.png", "latency_breakdown.png", "test_by_dep.png"]}
    with FileLock(str(output.with_name(f".{output.name}.plot.lock")), timeout=0):
        folder.mkdir(exist_ok=True)
        plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                            "axes.spines.top": False, "axes.spines.right": False})
        overview(result, folder)
        coverage(result, folder)
        budgets(result, folder)
        times(result, folder)
        by_dep(result, folder)
        save_json(folder / "figure_manifest.json", manifest)
    return manifest
