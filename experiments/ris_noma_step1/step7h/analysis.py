"""Layout-paired summaries; failed returns remain in every declared denominator."""
from collections import defaultdict, Counter
from pathlib import Path
import statistics
import numpy as np

from .common import METHODS, SHORT, DIRECT, PRIMARY, read, write, table

LABELS = {"full_pso": "Full PSO", "random_short_pso": "PSO ngắn ngẫu nhiên",
          "mlp_direct": "MLP trực tiếp", "gnn_direct": "GNN trực tiếp",
          "mlp_short_pso": "MLP + PSO ngắn", "gnn_short_pso": "GNN + PSO ngắn"}


# Tổng hợp mọi lượt đã khai báo, kể cả trượt. pass_fraction đo khả thi
# vật lý, không đo dự đoán trùng nhãn PSO hay accuracy phân loại.
def metrics(rows):
    if not rows: raise ValueError("Empty benchmark denominator")
    good = [r for r in rows if not r.get("invalid_layout_assumption", False)]
    # Chỉ dùng tập này cho rate_on_passed. Gated rate bên dưới lấy trung bình
    # trên toàn rows, trong đó ca trượt đóng góp 0.
    passed = [r for r in rows if r["passed"]]
    return {"allocations": len(rows), "scenarios": len({r["case_id"] for r in rows}),
            "layouts": len({r["layout_id"] for r in rows}), "passed": len(passed),
            "pass_fraction": len(passed)/len(rows), "gated_rate": statistics.mean(r["gated_bob_rate"] for r in rows),
            "raw_rate": statistics.mean(r["bob_rate"] for r in rows),
            "rate_on_passed": statistics.mean(r["bob_rate"] for r in passed) if passed else None,
            "eligible_allocations": len(good), "eligible_pass_fraction": sum(r["passed"] for r in good)/len(good) if good else None,
            "mean_online_ms": 1000*statistics.mean(r["online_seconds"] for r in rows),
            "median_online_ms": 1000*statistics.median(r["online_seconds"] for r in rows),
            "p90_online_ms": 1000*float(np.quantile([r["online_seconds"] for r in rows], .9)),
            "mean_pso_ms": 1000*statistics.mean(r["pso_seconds"] for r in rows),
            "objective_evaluations_per_return": statistics.mean(r["objective_evaluations"] for r in rows),
            "objective_evaluations_sum": sum(r["objective_evaluations"] for r in rows),
            "fallback_count": 0,
            "failure_counts": dict(Counter(c for r in rows for c in r["failed_checks"]))}


def grouped(rows, fields):
    groups = defaultdict(list)
    for row in rows: groups[tuple(row[f] for f in fields)].append(row)
    return [dict(zip(fields, key), **metrics(part)) for key, part in sorted(groups.items(), key=lambda x: str(x[0]))]


# Bootstrap ở cấp layout để giữ các case công suất/DEP và seed lặp tương quan
# trong cùng bố trí; số fading không phải số bố trí độc lập.
def layout_intervals(rows, resamples, seed):
    """Resample entire layouts, retaining correlated scenarios and seed repeats.

    Weighted sums preserve the row-mean estimand when a CV stratum contains
    different numbers of K/cases from different layouts.
    """
    blocks = defaultdict(list)
    for row in rows: blocks[row["layout_id"]].append(row)
    n = len(blocks)
    result = {"layout_clusters": n, "allocations": len(rows), "small_cluster_count": n < 10,
              "interval_kind": "pointwise95 layout-cluster percentile bootstrap; not simultaneous significance tests"}
    if n < 2:
        return dict(result, pass_fraction_ci95=None, gated_rate_ci95=None, mean_online_ms_ci95=None)
    values = np.asarray([[len(part), sum(r["passed"] for r in part),sum(r["gated_bob_rate"] for r in part),
                          1000*sum(r["online_seconds"] for r in part)] for _,part in sorted(blocks.items())],float)
    rng = np.random.default_rng(seed)
    # Mỗi hàng là một lần lấy lại n bố trí có hoàn lại; giữ nguyên mọi ca bên trong.
    indices = rng.integers(0,n,size=(resamples,n))
    totals = values[indices].sum(axis=1)
    for j,name in enumerate(("pass_fraction","gated_rate","mean_online_ms"),1):
        result[name+"_ci95"] = np.quantile(totals[:,j]/totals[:,0],[.025,.975]).tolist()
    return result


# Ghép đúng case_id và replicate trước khi tính chênh lệch phương pháp.
# Thiếu hoặc lặp một ca sẽ bị từ chối để tránh so hai mẫu khác nhau.
def paired(left, right, resamples, seed):
    def identity(row): return row["case_id"], row["replicate"]
    a, b = {identity(r): r for r in left}, {identity(r): r for r in right}
    if len(a) != len(left) or len(b) != len(right) or set(a) != set(b):
        raise ValueError("Paired comparison requires exactly the same scenarios and seed replicates")
    blocks = defaultdict(list)
    for key in a:
        x, y = a[key], b[key]
        if x["layout_id"] != y["layout_id"]: raise ValueError("Paired layout changed")
        blocks[x["layout_id"]].append((float(x["passed"])-float(y["passed"]),
            x["gated_bob_rate"]-y["gated_bob_rate"], x["online_seconds"]-y["online_seconds"],
            x["online_seconds"], y["online_seconds"]))
    # Primary design has equal scenario/replicate counts per layout. Cluster means avoid pseudo-replication.
    sizes = {len(v) for v in blocks.values()}
    if len(sizes) != 1: raise ValueError("Primary paired blocks must have equal predeclared sizes")
    # Mỗi layout đóng góp một vector chênh lệch trung bình; các seed lặp không
    # được coi là bố trí mới độc lập.
    values = np.asarray([np.mean(blocks[k], axis=0) for k in sorted(blocks)])
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(resamples, len(values)))
    boot = values[indices].mean(axis=1)
    result = {"layout_clusters": len(values), "paired_allocations": len(a),
              "bootstrap_resamples": resamples, "interval_kind": "pointwise 95% paired layout bootstrap; descriptive, no multiple-testing p-values"}
    for j, name in enumerate(("pass_difference", "gated_rate_difference", "online_seconds_difference")):
        result[name] = float(values[:, j].mean())
        result[name+"_ci95"] = np.quantile(boot[:, j], [.025, .975]).tolist()
    result["right_over_left_time_ratio"] = float(values[:, 4].mean()/values[:, 3].mean()) if values[:, 3].mean() > 0 else None
    result["small_cluster_count"] = len(values) < 10
    return result


# Chọn ngân sách nhỏ nhất mà cả GNN và MLP hybrid cùng đạt mục tiêu
# đã khai báo so với Full PSO trên validation; không xem test để chọn.
def budget_selection(rows, c):
    full = metrics([r for r in rows if r["method"] == "full_pso"])
    choices = []
    for budget in c["short_budgets"]:
        networks = {kind: metrics([r for r in rows if r["method"] == kind+"_short_pso" and r["budget"] == budget])
                    for kind in ("gnn", "mlp")}
        meets = full["passed"] > 0 and full["gated_rate"] > 0 and all(m["pass_fraction"] >= full["pass_fraction"]-c["budget_max_pass_loss"]
                    and m["gated_rate"] >= c["budget_min_rate_ratio"]*full["gated_rate"] for m in networks.values())
        choices.append({"budget": budget, "both_networks_meet_reference_target": meets, "networks": networks})
    passing = [r["budget"] for r in choices if r["both_networks_meet_reference_target"]]
    # Nếu không ngân sách nào đạt, vẫn ghi nhận target chưa đạt; dùng mức lớn
    # nhất theo quy tắc định trước, không đổi tiêu chí sau khi xem kết quả.
    selected = min(passing) if passing else max(c["short_budgets"])
    random = metrics([r for r in rows if r["method"] == "random_short_pso" and r["budget"] == selected])
    direct = {kind: metrics([r for r in rows if r["method"] == kind+"_direct"]) for kind in ("gnn", "mlp")}
    return {"selected_short_budget": selected, "reference_target_met": bool(passing),
            "reference_has_feasible_return": full["passed"] > 0 and full["gated_rate"] > 0, "full_reference": full,
            "candidate_budgets": choices, "random_at_selected_budget": random, "direct": direct,
            "rule": "Smallest budget at which BOTH GNN and MLP hybrids lose at most 1 percentage point feasibility and retain >=95% gated rate relative to Full PSO; if none, use largest and record target not met",
            "claim": "Validation-only engineering target; neither a statistical noninferiority proof nor proof of learned-initialization superiority",
            "test_used": False}


# Xuất bảng chung, bảng theo K/DEP/công suất/CV và so sánh ghép cặp.
# K đã thấy và K ngoại suy được tổng hợp riêng để không che lỗi zero-shot.
def summarize(output, plan, rows):
    output = Path(output)
    enriched = []
    for row in rows:
        r = dict(row)
        r["dep_bin"] = min(3, max(0, int((r["xi"]-.75)/.05)))
        r["power_bin"] = 0 if r["power_dbm"] < 35 else 1
        enriched.append(r)
    overall = grouped(enriched, ("regime", "method", "budget"))
    table(output / "metrics.csv", overall)
    table(output / "direct_metrics.csv", [r for r in overall if r["method"] in DIRECT])
    table(output / "pso_metrics.csv", [r for r in overall if r["method"] not in DIRECT])
    for name, field in (("by_k", "n_public"), ("by_dep", "dep_bin"), ("by_power", "power_bin"),
                        ("by_cv", "cv_stratum"), ("by_layout", "layout_id"), ("by_seed", "replicate")):
        table(output / f"{name}.csv", grouped(enriched, ("regime", field, "method", "budget")))
    table(output / "failure_rows.csv", [{k:v for k,v in r.items() if k != "history"} for r in enriched if not r["passed"]])
    table(output / "physics_rows.csv", [{k:v for k,v in r.items() if k != "history"} for r in enriched])
    contrasts = []
    for regime in sorted({r["regime"] for r in enriched}):
        subset = [r for r in enriched if r["regime"] == regime]
        for budget in plan["short_budgets"]:
            for left, right in PRIMARY:
                a = [r for r in subset if r["method"] == left and r["budget"] == budget]
                b = [r for r in subset if r["method"] == right and (right == "full_pso" or r["budget"] == budget)]
                contrasts.append({"regime": regime, "budget": budget, "left": left, "right": right,
                    **paired(a, b, plan["settings"]["bootstrap_resamples"], plan["settings"]["bootstrap_seed"])})
    table(output / "paired_contrasts.csv", contrasts)
    intervals = []
    for stratum_kind in ("overall","cv"):
        groups = defaultdict(list)
        for r in enriched:
            stratum = r["cv_stratum"] if stratum_kind == "cv" else "all"
            groups[(r["regime"],stratum,r["method"],r["budget"])].append(r)
        for (regime,stratum,method,budget),part in sorted(groups.items()):
            intervals.append({"stratum_kind":stratum_kind,"regime":regime,"stratum":stratum,"method":method,"budget":budget,
                **layout_intervals(part,plan["settings"]["bootstrap_resamples"],plan["settings"]["bootstrap_seed"])})
    table(output / "layout_confidence_intervals.csv",intervals)
    totals = {"scope": plan["settings"]["scope"], "stage": plan["stage"], "methods": list(METHODS),
        "layouts": len({r["layout_id"] for r in enriched}), "unique_scenarios": len({r["case_id"] for r in enriched}),
        "allocation_rows": len(enriched), "objective_evaluations": sum(r["objective_evaluations"] for r in enriched),
        "metrics": overall, "paired_contrasts": contrasts,"layout_confidence_intervals":intervals,
        "test_opened": plan["stage"] == "final", "test_used_for_selection": False,
        "scope_note": "K=2..5 and K=6 are reported separately; no pooling as an in-domain headline; intervals cluster by layout",
        "timing_note": "Measured CPU warm-return time including amortized input-bank generation, graph, forward, delta, objective setup and PSO; independent audit excluded; cold model loading recorded separately",
        "certificate_note": "Finite frozen proposal family under the adopted simulator, conditional on fixed geometry and iid fading; no global optimum or real-world guarantee"}
    if plan["stage"] == "development":
        totals["budget_selection"] = budget_selection(enriched, plan["settings"])
        write(output / "budget_selection.json", totals["budget_selection"], immutable=True)
    write(output / "summary.json", totals)
    plot(output, plan, enriched)
    return totals


# Vẽ từ các bản ghi đã lưu; bước vẽ không chạy PSO hoặc huấn luyện lại.
def plot(output, plan=None, rows=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output = Path(output)
    if plan is None: plan = read(output / "plan.json")
    if rows is None:
        from .benchmark import all_rows
        rows = all_rows(output, plan)
    folder = output / "figures"
    folder.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    if plan["stage"] == "development":
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        for method in SHORT:
            ms = [metrics([r for r in rows if r["method"] == method and r["budget"] == b]) for b in plan["short_budgets"]]
            for ax, field, scale in zip(axes, ("pass_fraction", "gated_rate", "mean_online_ms"), (100, 1, 1)):
                ax.plot(plan["short_budgets"], [m[field]*scale for m in ms], "o-", label=LABELS[method])
        reference = metrics([r for r in rows if r["method"] == "full_pso"])
        for ax, field, scale, title in zip(axes, ("pass_fraction", "gated_rate", "mean_online_ms"), (100, 1, 1),
            ("Tỷ lệ qua kiểm định (%)", "Rate có tính lượt trượt (bit/s/Hz)", "Thời gian trả nghiệm (ms)")):
            ax.axhline(reference[field]*scale, color="black", linestyle="--", label="Full PSO")
            ax.set_title(title); ax.set_xlabel("Số phép đánh giá mục tiêu")
            ax.grid(alpha=.25); ax.legend(fontsize=7)
        fig.suptitle("Validation: mỗi ngân sách là một lượt PSO riêng")
        fig.tight_layout(); fig.savefig(folder / "budget_curve.png", dpi=160); plt.close(fig)
    budget = (read(output / "budget_selection.json")["selected_short_budget"] if plan["stage"] == "development"
              else plan["short_budgets"][0])
    for regime in sorted({r["regime"] for r in rows}):
        part = [r for r in rows if r["regime"] == regime and (r["method"] not in SHORT or r["budget"] == budget)]
        figure, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
        summaries = [metrics([r for r in part if r["method"] == name]) for name in METHODS]
        y = np.arange(len(METHODS))
        for ax, field, scale, title in zip(axes, ("pass_fraction", "gated_rate", "mean_online_ms"), (100, 1, 1),
            ("Qua kiểm định (%)", "Rate có tính lượt trượt", "Thời gian trả nghiệm (ms)")):
            values = [r[field]*scale for r in summaries]
            ax.barh(y, values, color=plt.get_cmap("tab10")(np.arange(len(y))))
            ax.set_yticks(y, [LABELS[m] for m in METHODS]); ax.set_title(title); ax.grid(axis="x", alpha=.2)
            for i, v in enumerate(values): ax.text(v, i, f" {v:.2f}", va="center", fontsize=8)
        axes[0].invert_yaxis()
        figure.suptitle(f"{regime}: {len({r['layout_id'] for r in part})} bố trí; PSO ngắn {budget} phép đánh giá")
        figure.tight_layout(); figure.savefig(folder / f"comparison_{regime}.png", dpi=160); plt.close(figure)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    ks = sorted({r["n_public"] for r in rows})
    for method in METHODS:
        mm = [metrics([r for r in rows if r["n_public"] == k and r["method"] == method
              and (method not in SHORT or r["budget"] == budget)]) for k in ks]
        axes[0].plot(ks, [100*m["pass_fraction"] for m in mm], "o-", label=LABELS[method])
        axes[1].plot(ks, [m["gated_rate"] for m in mm], "o-", label=LABELS[method])
    for ax, title in zip(axes, ("Qua kiểm định (%)", "Rate có tính lượt trượt (bit/s/Hz)")):
        ax.set_title(title); ax.set_xlabel("Số người dùng công khai K"); ax.set_xticks(ks)
        ax.grid(alpha=.25); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(folder / "by_k.png", dpi=160); plt.close(fig)
    for field, filename, title in (("cv_stratum", "by_cv.png", "Nhóm CV Willie từ bank đầu vào"),
                                   ("dep_bin", "by_dep.png", "Khoảng DEP yêu cầu")):
        selected = [r for r in rows if r["regime"] == "seen_k" and (r["method"] not in SHORT or r["budget"] == budget)]
        for r in selected:
            if field == "dep_bin": r["dep_bin"] = min(3,max(0,int((r["xi"]-.75)/.05)))
        groups = sorted({r[field] for r in selected},key=str)
        figure, axes = plt.subplots(1,2,figsize=(12,4.2))
        for method in METHODS:
            values = [metrics([r for r in selected if r["method"]==method and r[field]==group]) for group in groups]
            axes[0].plot(range(len(groups)),[100*m["pass_fraction"] for m in values],"o-",label=LABELS[method])
            axes[1].plot(range(len(groups)),[m["gated_rate"] for m in values],"o-",label=LABELS[method])
        ticks = (["CV < 0,9" if g=="low" else "CV ≥ 0,9" if g=="regular" else str(g) for g in groups]
                 if field=="cv_stratum" else [f"{.75+.05*g:.2f}–{.80+.05*g:.2f}" for g in groups])
        for ax, subtitle in zip(axes,("Qua kiểm định (%)","Rate có tính lượt trượt (bit/s/Hz)")):
            ax.set_xticks(range(len(groups)),ticks);ax.set_xlabel(title);ax.set_title(subtitle)
            ax.grid(alpha=.25);ax.legend(fontsize=7)
        figure.suptitle("K=2–5; nhóm trống không được bổ sung bằng lấy mẫu lại")
        figure.tight_layout();figure.savefig(folder/filename,dpi=160);plt.close(figure)
