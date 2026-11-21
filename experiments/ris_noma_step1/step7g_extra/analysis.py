from collections import Counter,defaultdict
from pathlib import Path
import statistics
import numpy as np
from .common import read,write,table,digest,metrics,allocation_errors
from .physics import phase_rows,variants
from .fits import read_models


def original_failures(out,s):
    out=Path(out);targets={r["case_id"]:r for r in s["validation"]}
    selected={v["candidate_id"] for v in s["source_views"].values()};records=[]
    weights=s["stage_plan"]["config"]["training"]["loss_weights"]
    for path in sorted((s["stage"]/"validation_audit").glob("*.json")):
        if path.name.endswith(".receipt.json"):continue
        for r in read(path)["rows"]:
            if r["candidate_id"] not in selected:continue
            target=targets[r["case_id"]];y=r["allocation"]
            records.append({"architecture":r["architecture_name"],"kind":r["kind"],**{k:r[k] for k in
                ("case_id","layout_id","n_public","power_dbm","xi","seed","passed","bob_rate","gated_bob_rate","failed_checks")},
                "invalid_layout_assumption":not r["point_checks_without_dep"]["layout_bob_mean_gain_rule"],
                "dep_gamma_margin":r["dep_gamma"]-r["xi"],"dep_empirical_margin":r["dep_empirical"]-r["xi"],
                "dep_lower_margin":r["dep_lower"]-r["xi"],"qos_lower_margin":r["qos_lower_margin"],
                "public_lower_margins":[v-.2 for v in r["public_rate_lower_sic"]],
                "mb_prediction":y[0],"mb_label":target["allocation"][0] if target["has_label"] else None,
                **allocation_errors(y,target["allocation"] if target["has_label"] else None,weights),
                "xi_bin":min(3,int((r["xi"]-.75)/.05)),"power_bin":min(1,int((r["power_dbm"]-30)/5))})
    table(out/"source_all_rows.csv",records)
    table(out/"source_failures.csv",[r for r in records if not r["passed"]])
    breakdown=[]
    for field in ("n_public","xi_bin","power_bin","layout_id"):
        groups=defaultdict(list)
        for r in records:groups[(r["architecture"],r["kind"],r["seed"],field,r[field])].append(r)
        for (arch,kind,seed,field,value),part in groups.items():
            breakdown.append({"architecture":arch,"kind":kind,"seed":seed,"grouping":field,"value":value,
                "cases":len(part),"passed":sum(r["passed"] for r in part),"failures":sum(not r["passed"] for r in part),
                "invalid_assumption_cases":sum(r["invalid_layout_assumption"] for r in part),
                "violation_counts":dict(Counter(n for r in part for n in r["failed_checks"]))})
    table(out/"source_failure_breakdown.csv",breakdown)
    result={"scope":"original Windows stage-32 validation; existing audit only", "prediction_rows":len(records),
        "failed_rows":sum(not r["passed"] for r in records),"layout_clusters":len({r["layout_id"] for r in records}),
        "counts_overlap":True,"xi_bins":[.75,.80,.85,.90,.95],"power_dbm_bins":[30,35,40],
        "missing_labels_have_no_mb_error":True,"new_physics_evaluations":0,"new_pso_evaluations":0,
        "note":"All seed/case outputs are dependent within layout; invalid Bob mean-gain cases retained in denominator"}
    write(out/"source_failure_summary.json",result)
    return result


def grouped(rows):
    result=defaultdict(list)
    for r in rows:result[r["view_id"]].append(r)
    return result


def select(out,plan,save=True):
    out=Path(out);refs=read_models(out);views=variants(refs,plan["settings"])
    val=phase_rows(out,plan,"validation");groups=grouped(val)
    if any(r["phase"]!="validation" or r["split"]!="validation" for r in val):
        raise ValueError("Train/test rows cannot select a model")
    scores={k:metrics(v) for k,v in groups.items()}
    expected=len({r["case_id"] for r in val})
    if set(groups)!=set(views) or any(v["cases"]!=expected for v in scores.values()):
        raise ValueError("Validation candidate/case denominator changed")
    def key(name):
        m=scores[name];v=views[name]
        return (-m["passed"],-m["gated_rate"],m["weighted_loss"] if m["weighted_loss"] is not None else float("inf"),v["epoch"],name)
    frozen={}
    for model,ref in refs.items():
        if ref["family"]!="frozen32":continue
        candidates=[k for k,v in views.items() if v["model_id"]==model]
        frozen[model]={"raw":next(k for k in candidates if views[k]["delta"]==0),"corrected":min(candidates,key=key)}
    raw_baselines=[k for k,v in views.items() if v["family"]=="baseline" and v["delta"]==0]
    best_raw=min(raw_baselines,key=key);best_model=views[best_raw]["model_id"]
    best_delta=min([k for k,v in views.items() if v["model_id"]==best_model],key=key)
    curves={}
    for size in plan["settings"]["sizes"]:
        for kind in ("gnn","mlp"):
            for seed in plan["settings"]["seeds"]:
                candidates=[k for k,v in views.items() if v["family"]=="curve" and
                            (v["size"],v["kind"],v["seed"])==(size,kind,seed)]
                curves[f"{size}::{kind}::{seed}"]=min(candidates,key=key)
    teacher=next(k for k,v in views.items() if v["family"]=="teacher")
    train_views=sorted({teacher,best_raw,best_delta,*curves.values(),
                         *(v for pair in frozen.values() for v in pair.values())})
    result={"frozen":frozen,"baseline":{"raw":best_raw,"corrected":best_delta},"curve":curves,
        "teacher":teacher,"train_views":train_views,"scores":scores,"test_used_for_selection":False,
        "source":"fresh diagnostic validation only; no train audit used for selection",
        "plan_sha256":digest(out/"plan.json"),"validation_audit_sha256":digest(out/"validation_audit_frozen.json")}
    if save:
        write(out/"selection.json",result,immutable=True)
        table(out/"validation_candidates.csv",[{"view_id":k,**{n:views[k][n] for n in
              ("family","size","kind","architecture","seed","epoch","delta")},**v} for k,v in scores.items()])
    return result


def bootstrap_difference(a,b,resamples,seed):
    # Average seed outputs inside each layout, then bootstrap paired layouts.
    x={};y={}
    for rows,target in ((a,x),(b,y)):
        groups=defaultdict(list)
        for r in rows:groups[r["layout_id"]].append(r)
        for lid,part in groups.items():target[lid]=metrics(part)
    if set(x)!=set(y):raise ValueError("Paired comparison is missing layouts")
    ids=sorted(x);rng=np.random.default_rng(seed);draws=rng.integers(0,len(ids),size=(resamples,len(ids)))
    result={"layout_clusters":len(ids),"interpretation":"conditional validation diagnostic after checkpoint selection"}
    for metric in ("pass_fraction","gated_rate","case_mse"):
        values=np.array([y[lid][metric]-x[lid][metric] for lid in ids])
        lo,hi=np.quantile(values[draws].mean(1),[.025,.975])
        result.update({metric+"_difference":float(values.mean()),metric+"_ci_low":float(lo),metric+"_ci_high":float(hi)})
    return result


def bootstrap_interval(rows,resamples,seed):
    groups=defaultdict(list)
    for r in rows:groups[r["layout_id"]].append(r)
    parts=[groups[k] for k in sorted(groups)]
    rng=np.random.default_rng(seed);draws=rng.integers(0,len(parts),(resamples,len(parts)))
    result={"layout_clusters":len(parts)}
    for metric in ("pass_fraction","gated_rate","case_mse"):
        nums=[];dens=[]
        for part in parts:
            if metric=="case_mse":a=[r["case_mse"] for r in part if r["case_mse"] is not None]
            else:a=[r["passed"] if metric=="pass_fraction" else r["gated_bob_rate"] for r in part]
            nums.append(sum(a));dens.append(len(a))
        nums,dens=np.asarray(nums),np.asarray(dens)
        values=nums[draws].sum(1)/dens[draws].sum(1)
        result[metric+"_ci_low"],result[metric+"_ci_high"]=map(float,np.quantile(values,[.025,.975]))
    return result


def summarize(out,plan):
    out=Path(out);selection=read(out/"selection.json");refs=read_models(out);views=variants(refs,plan["settings"])
    val=phase_rows(out,plan,"validation");train=phase_rows(out,plan,"train")
    vg,tg=grouped(val),grouped(train);rows=[];curve_rows=[];contrasts=[]
    selected={selection["teacher"],*selection["baseline"].values(),*selection["curve"].values(),
              *(v for pair in selection["frozen"].values() for v in pair.values())}
    for view in sorted(selected):
        v=views[view]
        for phase,g in (("train",tg),("validation",vg)):
            if view not in g:continue
            record={"view_id":view,"phase":phase,**{n:v[n] for n in
                ("family","size","kind","architecture","seed","epoch","delta")},**metrics(g[view]),
                "parameters":v["parameters"],"training_seconds":v.get("training_seconds"),
                "training_time_origin":v.get("training_time_origin",plan["versions"]["system"])}
            rows.append(record)
            if v["family"]=="curve":curve_rows.append(record)
    table(out/"selected_metrics.csv",rows);table(out/"learning_curve.csv",curve_rows)
    cfg=plan["settings"]
    for name,pair in {**selection["frozen"],"baseline":selection["baseline"]}.items():
        contrast=bootstrap_difference(vg[pair["raw"]],vg[pair["corrected"]],cfg["bootstrap_resamples"],cfg["bootstrap_seed"])
        contrasts.append({"comparison":name+" corrected_minus_raw",**contrast})
    curve_averages=[]
    for kind in ("gnn","mlp"):
        by_size={}
        for size in cfg["sizes"]:
            part=[r for r in val if r["view_id"] in [v for k,v in selection["curve"].items() if k.startswith(f"{size}::{kind}::")]]
            by_size[size]=part
            curve_averages.append({"kind":kind,"size":size,"seed_outputs":len(part),**metrics(part),
                **bootstrap_interval(part,cfg["bootstrap_resamples"],cfg["bootstrap_seed"])})
        for a,b in zip(cfg["sizes"],cfg["sizes"][1:]):
            contrasts.append({"comparison":f"{kind} curve {b}_minus_{a}",**bootstrap_difference(by_size[a],by_size[b],cfg["bootstrap_resamples"],cfg["bootstrap_seed"])})
    table(out/"learning_curve_seed_average.csv",curve_averages);table(out/"paired_layout_contrasts.csv",contrasts)
    rawfail=[r for r in val if r["family"]=="frozen32" and r["delta"]==0 and not r["passed"]]
    table(out/"fresh_validation_failures.csv",rawfail)
    delta_rows=[]
    for model,pair in {**selection["frozen"],"baseline":selection["baseline"]}.items():
        before={r["case_id"]:r for r in vg[pair["raw"]]};after={r["case_id"]:r for r in vg[pair["corrected"]]}
        for case,r in before.items():
            q=after[case]
            delta_rows.append({"model":model,"case_id":case,"layout_id":r["layout_id"],"n_public":r["n_public"],
                "xi":r["xi"],"power_dbm":r["power_dbm"],"selected_delta":q["delta"],
                "passed_before":r["passed"],"passed_after":q["passed"],"raw_rate_change":q["bob_rate"]-r["bob_rate"],
                "gated_rate_change":q["gated_bob_rate"]-r["gated_bob_rate"],
                "dep_lower_margin_before":r["dep_lower_margin"],"dep_lower_margin_after":q["dep_lower_margin"],
                "qos_lower_margin_before":r["qos_lower_margin"],"qos_lower_margin_after":q["qos_lower_margin"],
                "mb_before":r["allocation"][0],"mb_after":q["allocation"][0]})
    table(out/"delta_tradeoffs.csv",delta_rows)
    methods=defaultdict(lambda:{"raw":[],"corrected":[]})
    for model,pair in selection["frozen"].items():
        ref=refs[model];name=ref["kind"].upper()+" "+ref["architecture"]
        for policy,view in pair.items():methods[name][policy].append(view)
    methods["Scalar ridge"]= {policy:[view] for policy,view in selection["baseline"].items()}
    comparison=[]
    for method,policies in methods.items():
        for policy,identities in policies.items():
            for phase,g in (("train",tg),("validation",vg)):
                part=[r for identity in identities for r in g[identity]]
                comparison.append({"method":method,"policy":policy,"phase":phase,"model_runs":len(identities),
                    "deltas":[views[v]["delta"] for v in identities],**metrics(part),
                    **bootstrap_interval(part,cfg["bootstrap_resamples"],cfg["bootstrap_seed"])})
    table(out/"method_comparison.csv",comparison)
    result={"scope":plan["scope"],"complete":True,"reserved_test_generated":False,"pso_evaluations":0,
        "original_stage_modified":False,"validation_candidates":len(vg),"validation_cases_per_view":len(next(iter(vg.values()))),
        "train_physical_rows":len(train),"validation_physical_rows":len(val),"selected_metrics":rows,
        "curve_seed_average":curve_averages,"paired_layout_contrasts":contrasts,"method_comparison":comparison,
        "point_vs_bound_note":"Point acceptance and bound acceptance are reported separately; neither alone proves network superiority",
        "baseline_note":"Ridge uses only xi, power in dBm, K and normalized SIC rank, never coordinates/gains/audit fields",
        "selection_note":"All hyperparameters/checkpoints/deltas selected on diagnostic validation; final test remains unopened",
        "timing_note":"32-layout training times were reused from source runtime; do not compare cross-machine wall times",
        "decision":"Review 8/16/32 gains and scalar baseline first; no automatic claim that 32 is sufficient or 64 unnecessary"}
    write(out/"summary.json",result)
    plot(out)
    report(out,result)
    return result


def plot(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out=Path(out);s=read(out/"summary.json");figdir=out/"figures";figdir.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,3,figsize=(12.6,4.1),layout="constrained")
    for kind in ("gnn","mlp"):
        rows=sorted([r for r in s["curve_seed_average"] if r["kind"]==kind],key=lambda r:r["size"])
        for ax,key,label in zip(axes,("pass_fraction","gated_rate","case_mse"),("Physical pass fraction","Gated Bob rate (bit/s/Hz)","Case MSE")):
            line=ax.plot([r["size"] for r in rows],[r[key] for r in rows],"o-",label=kind.upper())[0]
            ax.fill_between([r["size"] for r in rows],[r[key+"_ci_low"] for r in rows],
                [r[key+"_ci_high"] for r in rows],alpha=.13,color=line.get_color())
            ax.set_xlabel("Train layouts");ax.set_ylabel(label);ax.grid(alpha=.2);ax.legend()
    scope=s["scope"].upper()
    fig.suptitle(f"{scope}: 8/16/32 — validation, mean across seeds; conditional layout CIs")
    fig.savefig(figdir/"learning_curve.png",dpi=160);plt.close(fig)
    # Keep per-seed and conditional paired CIs in CSV; don't pretend seed outputs are independent layouts.
    metrics_rows=[r for r in s["method_comparison"] if r["phase"]=="validation"]
    names=[r["method"]+" "+r["policy"] for r in metrics_rows]
    passes=[r["pass_fraction"] for r in metrics_rows];rates=[r["gated_rate"] for r in metrics_rows]
    fig,axes=plt.subplots(1,2,figsize=(12,max(4,.34*len(names))),layout="constrained")
    for ax,values,label in zip(axes,(passes,rates),("Physical pass fraction","Gated Bob rate (bit/s/Hz)")):
        ax.barh(names,values);ax.set_xlabel(label);ax.grid(axis="x",alpha=.2)
    fig.suptitle(f"{scope}: frozen stage-32 models / scalar baseline / delta selected on validation")
    fig.savefig(figdir/"baseline_and_delta.png",dpi=160);plt.close(fig)


def report(out,s):
    out=Path(out)
    text="# Báo cáo bổ sung 7g — dữ liệu 32 bố trí\n\n"
    text+=f"Phạm vi: **{s['scope']}**. Đây là validation phát triển, chưa phải test cuối. Không chạy PSO mới và không sửa mốc 32.\n\n"
    text+="## Đường cong học mạng nhỏ\n\n| Mạng | Bố trí train | Tỷ lệ đạt | Rate có tính trượt | MSE |\n|---|---:|---:|---:|---:|\n"
    for r in s["curve_seed_average"]:text+=f"| {r['kind'].upper()} | {r['size']} | {100*r['pass_fraction']:.2f}% | {r['gated_rate']:.5f} | {r['case_mse']:.7f} |\n"
    text+="\nKết quả được lấy trung bình trên các seed; các lượt trong cùng bố trí có liên quan. Khoảng tin cậy ghép cặp theo bố trí nằm trong paired_layout_contrasts.csv và là mô tả có điều kiện sau lựa chọn trên validation.\n\n"
    text+="## Cách đọc các tệp\n\n- source_failures.csv: ca trượt đã có trong kết quả Windows, không chấm lại.\n- selected_metrics.csv: cả train và validation mới, cùng chính sách cận thống kê; giữ ca thiếu nhãn/bố trí không đạt giả định trong mẫu số vật lý.\n- validation_candidates.csv: toàn bộ ứng viên đã chấm, gồm cả biến thể không được chọn.\n- delta_tradeoffs.csv: từng ca trước/sau hiệu chỉnh; raw_rate_change là giảm rate thật, gated_rate_change còn chịu ảnh hưởng đạt/trượt.\n- fresh_validation_failures.csv: các ca trượt của mô hình mốc 32 trên bank chẩn đoán mới.\n- selection.json: lựa chọn chỉ từ validation; train không được dùng chọn checkpoint/alpha/delta.\n\n"
    text+="## Đối chứng và hiệu chỉnh\n\n| Phương pháp | Chính sách | Tỷ lệ đạt validation | Rate có tính trượt | MSE |\n|---|---|---:|---:|---:|\n"
    for r in s["method_comparison"]:
        if r["phase"]=="validation":text+=f"| {r['method']} | {r['policy']} | {100*r['pass_fraction']:.2f}% | {r['gated_rate']:.5f} | {r['case_mse']:.7f} |\n"
    text+="\nRaw và corrected luôn dùng cùng các lượt train/seed; delta có thể được chọn khác nhau theo seed. Xem deltas trong method_comparison.csv.\n\n"
    text+="## Giới hạn\n\nMô hình baseline chỉ dùng DEP, công suất dBm, K và hạng SIC. Phân bổ public bằng hồi quy theo hạng rồi softmax, không phải quy tắc SIC giải tích. So sánh này không thay thế Full PSO/short PSO của 7h. Không lấy tỷ lệ đạt cao làm bằng chứng không overfit; không lấy đường cong phẳng 8/16/32 làm bằng chứng 64 chắc chắn vô ích.\n\n"
    text+="Bank chẩn đoán mới và cận dùng chung khác hồ sơ Windows cũ, nên mọi thay đổi tỷ lệ so với báo cáo mốc 32 phải tách ảnh hưởng bank/cách chứng nhận khỏi hiệu chỉnh phương pháp. Nhãn PSO là nghiệm tham chiếu hữu hạn, không phải tối ưu toàn cục. Chưa đo sàn nhiễu nhãn bằng các seed PSO mới.\n\n"
    text+="Thời gian train mốc 32 được dùng lại từ hồ sơ nguồn; thời gian 8/16 lấy từ máy đang chạy phần bổ sung. Chỉ so trực tiếp thời gian nếu thực sự cùng máy/môi trường. Run/check không đóng seal 7g gốc; tập test 5000–5039 và K=6 vẫn chưa mở.\n"
    (out/"BAO_CAO_KET_QUA_BO_SUNG7G.md").write_text(text,encoding="utf-8")
