"""Standalone, reproducible 7d figures; original numerical artifacts untouched."""
from pathlib import Path
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .learning_data import read_json
from .multik_dataset import read_committed, write_json
from .multik_benchmark_core import METHODS

LABELS = {"full_pso": "PSO đầy đủ", "random_short_pso": "PSO ngắn ngẫu nhiên",
    "lookup_sic": "Bảng tra + SIC", "lookup_sic_short_pso": "Bảng tra + SIC + PSO",
    "gnn": "GNN trực tiếp", "mlp": "MLP trực tiếp", "gnn_short_pso": "GNN + PSO", "mlp_short_pso": "MLP + PSO"}
PROFILE_LABELS = {"legacy": "LoS tường minh cũ", "geometry": "LoS theo hình học", "geometry_friis": "LoS hình học + tham chiếu Friis"}


def plot(output):
    output = Path(output)
    # Check numerical completion before producing a deliverable figure.
    from .multik_benchmark import check
    summary = check(output)
    plan = read_json(output / "plan.json"); c = plan["config"]
    folder = output / "figures"; folder.mkdir(exist_ok=True)
    profiles = c["benchmark_profiles"]
    deployment = c["benchmark_deployments"][0]["id"]
    margin = max(c["search_margins"])
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    saved = []
    def save(fig, name):
        fig.savefig(folder / name, dpi=160, bbox_inches="tight"); plt.close(fig); saved.append(name)
    def subset(profile, selector="search_best"):
        return [r for r in summary["overview"] if r["stage"] == "benchmark" and r["profile"] == profile
                and r["deployment_id"] == deployment and r["search_margin"] == margin and r["selector"] == selector]
    fig, axes = plt.subplots(len(profiles), 3, figsize=(17, 5 * len(profiles)), squeeze=False)
    for i, profile in enumerate(profiles):
        rows = {r["method"]: r for r in subset(profile)}
        y = np.arange(len(METHODS))
        for j, (field, scale, label) in enumerate((("pass_fraction", 100, "Tỷ lệ qua kiểm định (%)"),
                ("gated_rate", 1, "Rate Bob có tính lượt trượt (bit/s/Hz)"),
                ("return_ms_cached_banks", 1, "Thời gian trả nghiệm, bank có sẵn (ms)"))):
            ax = axes[i, j]; values = [rows[m][field] * scale for m in METHODS]
            ax.barh(y, values, color=plt.get_cmap("tab10")(np.arange(len(METHODS))))
            ax.set_yticks(y, [LABELS[m] for m in METHODS] if j == 0 else [""] * len(METHODS)); ax.invert_yaxis()
            ax.set_xlabel(label); ax.grid(axis="x", alpha=.2)
            ax.set_title(PROFILE_LABELS[profile])
            if j == 0: ax.set_xlim(0, 108)
            for yy, value in zip(y, values): ax.text(value, yy, f" {value:.2f}", va="center", fontsize=8)
    fig.suptitle(f"Đối chứng 7d — chọn theo bank tìm kiếm, biên DEP {margin:g}")
    fig.text(.5, .005, "Trọng số 7c giữ nguyên; LoS hình học là kiểm tra chuyển môi trường. Lượt trượt tính rate bằng 0. Thời gian gồm chi phí đầu vào/forward và tìm kiếm.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .03, 1, .97)); save(fig, "benchmark_overview.png")

    fig, axes = plt.subplots(2, len(profiles), figsize=(7 * len(profiles), 8), squeeze=False)
    chosen = ("full_pso", "random_short_pso", "gnn_short_pso", "mlp_short_pso")
    for i, selector in enumerate(("search_best", "independent_bank")):
        for j, profile in enumerate(profiles):
            ax = axes[i, j]
            for method in chosen:
                rows = sorted([r for r in summary["overview"] if r["stage"] == "benchmark" and r["profile"] == profile
                    and r["deployment_id"] == deployment and r["method"] == method and r["selector"] == selector], key=lambda r:r["search_margin"])
                ax.plot([r["search_margin"] for r in rows], [100*r["pass_fraction"] for r in rows], "o-", label=LABELS[method])
            ax.set_title(PROFILE_LABELS[profile] + (" — chọn cũ" if i == 0 else " — bank độc lập"))
            ax.set_xlabel("Biên DEP khi tìm kiếm"); ax.set_ylabel("Tỷ lệ đạt audit (%)"); ax.set_ylim(-3, 103); ax.grid(alpha=.2)
            if i == j == 0: ax.legend(fontsize=8)
    fig.suptitle("Tách ảnh hưởng của biên DEP và bank chọn nghiệm")
    fig.tight_layout(rect=(0, 0, 1, .95)); save(fig, "margin_selection_effect.png")

    fig, axes = plt.subplots(1, len(profiles), figsize=(8 * len(profiles), 5), squeeze=False)
    for j, profile in enumerate(profiles):
        ax = axes[0, j]; x = np.arange(len(METHODS)); width = .38
        old = {r["method"]: r for r in subset(profile)}; new = {r["method"]: r for r in subset(profile, "independent_bank")}
        ax.bar(x - width/2, [old[m]["return_ms_cached_banks"] for m in METHODS], width, label="Chọn cũ, bank có sẵn")
        ax.bar(x + width/2, [new[m]["return_ms_cached_banks"] for m in METHODS], width, label="Chọn bằng bank độc lập")
        ax.set_xticks(x, [LABELS[m] for m in METHODS], rotation=40, ha="right", fontsize=8)
        ax.set_ylabel("ms/lượt"); ax.set_title(PROFILE_LABELS[profile]); ax.grid(axis="y", alpha=.2)
        if j == 0: ax.legend(fontsize=8)
    fig.suptitle("Chi phí chấm kho ứng viên được tính vào thời gian chọn mới")
    fig.tight_layout(rect=(0, 0, 1, .94)); save(fig, "selection_cost.png")

    with (output / "link_budget_probes.csv").open(encoding="utf-8-sig", newline="") as stream: probes = list(csv.DictReader(stream))
    ks = sorted(set(c["public_counts"]))
    displayed_ks = sorted({ks[0], ks[-1]})
    fig, axes = plt.subplots(len(displayed_ks), 2, figsize=(13, 4 * len(displayed_ks)), squeeze=False)
    for i, k in enumerate(displayed_ks):
        for profile in c["calibration_profiles"]:
            part = [r for r in probes if r["stage"] == "calibration" and r["profile"] == profile
                and r["deployment_id"] == c["calibration_deployments"][0]["id"] and int(r["n_public"]) == k]
            for j, field in enumerate(("median_kappa2_U_bob", "bob_rate_to_own_hardware_ceiling")):
                values = [np.mean([float(r[field]) for r in part if float(r["power_dbm"]) == power]) for power in c["power_dbm"]]
                axes[i, j].plot(c["power_dbm"], values, "o-", label=PROFILE_LABELS[profile])
                axes[i, j].set_xlabel("Công suất tổng (dBm)"); axes[i, j].grid(alpha=.2)
                axes[i, j].set_title(f"K={k}")
        axes[i, 0].set_yscale("log"); axes[i, 0].set_ylabel("Trung bình theo layout của median κ²U Bob")
        axes[i, 1].set_ylabel("Rate Bob / trần của cùng phân bổ"); axes[i, 1].set_ylim(0, 1.05)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Chẩn đoán link budget với phân bổ cố định mb=0,28; s=0,5 — không khẳng định nghiệm khả thi")
    fig.tight_layout(rect=(0, 0, 1, .94)); save(fig, "link_budget.png")

    fig, axes = plt.subplots(2, len(c["calibration_profiles"]), figsize=(6 * len(c["calibration_profiles"]), 7), squeeze=False)
    for j, profile in enumerate(c["calibration_profiles"]):
        jobs = [job for job in plan["jobs"] if job["stage"] == "calibration" and job["profile"] == profile
            and job["deployment"]["id"] == c["calibration_deployments"][0]["id"] and job["n_public"] == ks[0]]
        for job in jobs:
            meta = read_committed(output / "search" / (job["id"] + ".json"))["metadata"]
            for i, index in enumerate((0, -1)):
                phases = meta["los"]["ru_phase_rad"][index]
                axes[i, j].plot(np.arange(1, len(phases)+1), phases, ".-", label=f"layout {job['layout_id']}", alpha=.8)
                axes[i, j].set_xlabel("Chỉ số phần tử RIS"); axes[i, j].set_ylabel("Pha LoS tương đối (radian)"); axes[i, j].grid(alpha=.2)
                axes[i, j].set_title(PROFILE_LABELS[profile] + (" — Bob" if i == 0 else " — Willie"))
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout(); save(fig, "los_by_layout.png")

    fig, axes = plt.subplots(1, len(profiles), figsize=(7 * len(profiles), 4), squeeze=False)
    for j, profile in enumerate(profiles):
        ax = axes[0, j]
        for method in ("full_pso", "random_short_pso", "gnn", "mlp", "gnn_short_pso", "mlp_short_pso"):
            rows = sorted([r for r in summary["by_k"] if r["stage"] == "benchmark" and r["profile"] == profile
                and r["deployment_id"] == deployment and r["search_margin"] == margin and r["selector"] == "search_best" and r["method"] == method], key=lambda r:r["n_public"])
            ax.plot([r["n_public"] for r in rows], [100*r["pass_fraction"] for r in rows], "o-", label=LABELS[method])
        ax.set_xlabel("Số người dùng công khai K"); ax.set_ylabel("Tỷ lệ đạt audit (%)"); ax.set_ylim(-3, 103); ax.grid(alpha=.2)
        ax.set_title(PROFILE_LABELS[profile])
        if j == 0: ax.legend(fontsize=7)
    fig.tight_layout(); save(fig, "benchmark_by_k.png")
    write_json(folder / "figure_manifest.json", {"figures": saved, "scope": summary["scope"],
        "deployment_shown": deployment, "main_margin_shown": margin,
        "note": "All deployments, margins and selectors remain in CSV; figures show the first declared deployment."})
    return {"figures": saved, "scope": summary["scope"]}
