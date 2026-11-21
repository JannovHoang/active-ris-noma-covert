"""Learning curves and paired layout diagnostics; validation is not a final test."""
from collections import Counter
from pathlib import Path
import json
import math
import numpy as np

from .continuous_dataset import read_lines
from .expanded_data import write_table
from .learning_data import read_json
from .multik_dataset import write_json
from .curve_physics import read_record


def table(path,rows):
    rows=list(rows)
    keys=list(dict.fromkeys(key for row in rows for key in row))
    write_table(path,[{key:row.get(key) for key in keys} for row in rows])


def no_errors():
    return {"labeled_error_cases":0,"case_mse":None,"weighted_loss":None,
            "mb_rmse":None,"public_rmse":None,"s_rmse":None}


def physical_score(rows):
    dep=[r["dep_lower"]-r["xi"] for r in rows if r.get("dep_lower") is not None]
    qos=[r["qos_lower_margin"] for r in rows if r.get("qos_lower_margin") is not None]
    eligible=[r for r in rows if r.get("point_checks_without_dep",{}).get("layout_bob_mean_gain_rule") is True]
    evaluable=[r for r in rows if "layout_bob_mean_gain_rule" in r.get("point_checks_without_dep",{})]
    return {"cases":len(rows),"passed":sum(r["passed"] for r in rows),
            "pass_fraction":float(np.mean([r["passed"] for r in rows])),
            "raw_rate":float(np.mean([r["bob_rate"] for r in rows])),
            "gated_rate":float(np.mean([r["gated_bob_rate"] for r in rows])),
            "minimum_dep_lower_margin":float(min(dep)) if dep else None,
            "minimum_qos_lower_margin":float(min(qos)) if qos else None,
            "qos_lower_margin_p05":float(np.quantile(qos,.05)) if qos else None,
            "layout_gain_rule_evaluable_cases":len(evaluable),
            "layout_gain_rule_passed_cases":len(eligible),
            "pass_fraction_on_layout_gain_rule_passed":float(np.mean([r["passed"] for r in eligible])) if eligible else None,
            "gated_rate_on_layout_gain_rule_passed":float(np.mean([r["gated_bob_rate"] for r in eligible])) if eligible else None}


def allocation_errors(rows, lookup, weights):
    terms=[]
    for row in rows:
        source=lookup[row["case_id"]]
        if not source["has_label"] or not row["returned"]:
            continue
        e=(np.asarray(row["allocation"])-np.asarray(source["allocation"]))**2
        k=source["n_public"]
        terms.append({"case_mse":float(e.mean()),
            "weighted_loss":float((weights["mb"]*e[0]+weights["public_total"]*e[1:-1].mean()+weights["s"]*e[-1])/sum(weights.values())),
            "mb_sq":float(e[0]),"public_sq":float(e[1:-1].mean()),"s_sq":float(e[-1])})
    if not terms:
        return {"labeled_error_cases":0,"case_mse":None,"weighted_loss":None,
                "mb_rmse":None,"public_rmse":None,"s_rmse":None}
    mean=lambda key:float(np.mean([r[key] for r in terms]))
    return {"labeled_error_cases":len(terms),"case_mse":mean("case_mse"),"weighted_loss":mean("weighted_loss"),
            "mb_rmse":float(np.sqrt(mean("mb_sq"))),"public_rmse":float(np.sqrt(mean("public_sq"))),
            "s_rmse":float(np.sqrt(mean("s_sq")))}


def selected_rows(folder,stage,selection,phase):
    audited=[r for j in stage[phase+"_jobs"] for r in read_record(folder,phase,"audit",j)["rows"]]
    result=[]
    for view_id,ref in selection["views"].items():
        for row in audited:
            matches=(row["candidate_id"]==ref["candidate_id"] if phase=="validation" else row["view_id"]==view_id)
            if matches:
                result.append(dict(row,view_id=view_id,selection_rule=ref["selection_rule"]))
    result.extend(row for row in audited if row["kind"]=="pso_label_reference")
    return result


def stage_metrics(folder,stage,selection):
    folder=Path(folder)
    output=[]
    for phase,name in (("validation","validation"),("train_probe","train")):
        lookup={r["case_id"]:r for r in read_lines(folder/"inputs"/(name+".jsonl"))}
        rows=selected_rows(folder,stage,selection,phase)
        teachers={r["case_id"]:r for r in rows if r["kind"]=="pso_label_reference"}
        views=sorted({r["view_id"] for r in rows})
        for view in views:
            part=[r for r in rows if r["view_id"]==view]
            first=part[0]
            fields={k:first[k] for k in ("view_id","kind","architecture_name","seed","selection_rule")}
            errors=allocation_errors(part,lookup,stage["config"]["training"]["loss_weights"]) if first["kind"]!="pso_label_reference" else no_errors()
            paired=[r for r in part if r["case_id"] in teachers and teachers[r["case_id"]]["passed"]]
            output.append(dict(fields,phase=phase,**physical_score(part),**errors,
                teacher_feasible_cases=len(paired),paired_gated_rate_gap_vs_teacher=(
                    float(np.mean([r["gated_bob_rate"]-teachers[r["case_id"]]["bob_rate"] for r in paired])) if paired else None),
                failure_breakdown=dict(Counter(name for r in part for name in r["failed_checks"]))))
    return {"step":"7g","stage_complete":True,"train_layouts":stage["train_layouts"],
            "planned_counts":stage["planned_counts"],"labeled_counts":stage["labeled_counts"],
            "coverage":stage["coverage"],
            "validation_layouts":len(stage["config"]["validation_layout_ids"]),
            "train_probe_layouts":stage["config"]["physics"]["train_probe_layout_count"],
            "model_parameters":stage["model_parameters"],"metrics_per_view":output,
            "reserved_test_generated":False,"test_used_for_selection":False,
            "scope":"development learning curve; train probe is a fixed subset, validation is used for selection"}


def export_stage(folder,stage,refs,selection):
    folder=Path(folder)
    summary=stage_metrics(folder,stage,selection)
    write_json(folder/"summary.json",summary)
    table(folder/"metrics_per_view.csv",summary["metrics_per_view"])
    physical=[]
    layout_rows=[]
    by_k=[]
    for phase,name in (("validation","validation"),("train_probe","train")):
        rows=selected_rows(folder,stage,selection,phase)
        lookup={r["case_id"]:r for r in read_lines(folder/"inputs"/(name+".jsonl"))}
        for row in rows:
            physical.append({k:row.get(k) for k in ("view_id","kind","architecture_name","seed","selection_rule",
                "case_id","layout_id","n_public","power_dbm","xi","returned","passed","bob_rate","gated_bob_rate",
                "dep_lower","qos_lower_margin","bob_outage_upper","failed_checks")}|{"phase":phase})
        for view in sorted({r["view_id"] for r in rows}):
            part=[r for r in rows if r["view_id"]==view]
            first=part[0]
            fields={k:first[k] for k in ("view_id","kind","architecture_name","seed","selection_rule")}
            for lid in sorted({r["layout_id"] for r in part}):
                group=[r for r in part if r["layout_id"]==lid]
                errors=allocation_errors(group,lookup,stage["config"]["training"]["loss_weights"]) if first["kind"]!="pso_label_reference" else no_errors()
                layout_rows.append(dict(fields,phase=phase,layout_id=lid,**physical_score(group),**errors))
            for k in sorted({r["n_public"] for r in part}):
                group=[r for r in part if r["n_public"]==k]
                by_k.append(dict(fields,phase=phase,n_public=k,**physical_score(group)))
    table(folder/"physical_rows.csv",physical)
    table(folder/"metrics_by_layout.csv",layout_rows)
    table(folder/"metrics_by_k.csv",by_k)
    write_json(folder/"layout_metrics.json",layout_rows)
    candidate_scores=[dict(candidate_id=key,**value) for key,value in selection["candidate_validation_scores"].items()]
    table(folder/"candidate_validation_scores.csv",candidate_scores)
    return summary


def paired_bootstrap(a,b,settings,entropy):
    if set(a)!=set(b) or not a:
        raise ValueError("A paired layout comparison requires identical nonempty layout sets")
    ids=sorted(a)
    rng=np.random.default_rng(np.random.SeedSequence([settings["bootstrap_seed"],*entropy]))
    result={"layout_clusters":len(ids)}
    indices={}
    for metric in ("pass_fraction","gated_rate","case_mse"):
        available=[i for i in ids if a[i].get(metric) is not None and b[i].get(metric) is not None]
        result[metric+"_layout_clusters"]=len(available)
        if not available:
            for suffix in ("difference","ci_low","ci_high","baseline"):
                result[metric+"_"+suffix]=None
            continue
        n=len(available)
        if n not in indices:
            indices[n]=rng.integers(0,n,(settings["bootstrap_resamples"],n))
        delta=np.array([b[i][metric]-a[i][metric] for i in available])
        sample=delta[indices[n]].mean(axis=1)
        low,high=np.quantile(sample,[.025,.975])
        result[metric+"_difference"]=float(delta.mean())
        result[metric+"_ci_low"]=float(low)
        result[metric+"_ci_high"]=float(high)
        result[metric+"_baseline"]=float(np.mean([a[i][metric] for i in available]))
    return result


def average_layouts(rows,architecture,kind):
    selected=[r for r in rows if r["phase"]=="validation" and r["architecture_name"]==architecture
              and r["kind"]==kind and r["selection_rule"]=="physical"]
    result={}
    for lid in sorted({r["layout_id"] for r in selected}):
        group=[r for r in selected if r["layout_id"]==lid]
        result[lid]={metric:(float(np.mean([r[metric] for r in group if r.get(metric) is not None]))
                            if any(r.get(metric) is not None for r in group) else None)
                     for metric in ("pass_fraction","gated_rate","case_mse")}
    return result


def summarize(output):
    from .curve_learning import completed_stages,folder_for
    output=Path(output)
    master=read_json(output/"plan.json")
    stages=completed_stages(output)
    summaries={n:read_json(folder_for(output,n)/"summary.json") for n in stages}
    layout={n:read_json(folder_for(output,n)/"layout_metrics.json") for n in stages}
    curve=[]
    for n in stages:
        summary=summaries[n]
        for architecture in ("reference","small"):
            for kind in ("gnn","mlp"):
                for rule in ("physical","weighted_loss"):
                    for phase in ("validation","train_probe"):
                        group=[r for r in summary["metrics_per_view"] if (r["architecture_name"],r["kind"],r["selection_rule"],r["phase"])==(architecture,kind,rule,phase)]
                        metrics={metric+"_mean":float(np.mean([r[metric] for r in group])) for metric in
                                 ("pass_fraction","gated_rate","case_mse","weighted_loss","mb_rmse","public_rmse","s_rmse")}
                        spread={metric+"_seed_std":float(np.std([r[metric] for r in group],ddof=1)) if len(group)>1 else 0.
                                for metric in ("pass_fraction","gated_rate","case_mse")}
                        fits=[read_json(folder_for(output,n)/"cells"/architecture/"fits"/f"{kind}_s{s}"/"fit_manifest.json")
                              for s in master["config"]["training"]["seeds"]]
                        curve.append({"train_layouts":n,"train_cases":summary["planned_counts"]["train"],
                            "labeled_train_cases":summary["labeled_counts"]["train"],
                            "labeled_train_layouts":summary["coverage"]["train"]["labeled_layouts"],
                            "architecture_name":architecture,"kind":kind,"selection_rule":rule,"phase":phase,
                            "parameters":master["model_parameters"][architecture][kind],
                            "optimizer_updates":math.ceil(summary["labeled_counts"]["train"]/master["config"]["training"]["batch_size"])*master["config"]["training"]["epochs"],
                            "training_seconds_mean":float(np.mean([f["training_seconds"] for f in fits])),
                            **metrics,**spread})
    contrasts=[]
    for index,n in enumerate(stages):
        for ki,kind in enumerate(("gnn","mlp")):
            a=average_layouts(layout[n],"reference",kind)
            b=average_layouts(layout[n],"small",kind)
            contrasts.append({"contrast":"small_minus_reference","kind":kind,"from_size":n,"to_size":n,
                **paired_bootstrap(a,b,master["config"]["curve"],[1,n,ki])})
            if index:
                previous=stages[index-1]
                for ai,architecture in enumerate(("reference","small")):
                    a=average_layouts(layout[previous],architecture,kind)
                    b=average_layouts(layout[n],architecture,kind)
                    contrasts.append({"contrast":"larger_train_minus_smaller_train","architecture_name":architecture,
                        "kind":kind,"from_size":previous,"to_size":n,
                        **paired_bootstrap(a,b,master["config"]["curve"],[2,n,ki,ai])})
    remaining=[n for n in master["config"]["training_sizes"] if n not in stages]
    recommendation={"next_predeclared_size":remaining[0] if remaining else None,
                    "decision":"need_first_two_sizes" if len(stages)<2 else "inspect_validation_gains",
                    "automatic_proof_of_sufficient_data":False}
    if len(stages)>=2:
        latest=[r for r in contrasts if r["contrast"]=="larger_train_minus_smaller_train" and r["to_size"]==stages[-1]]
        cfg=master["config"]["curve"]
        no_gain=all(r["pass_fraction_ci_high"]<=cfg["minimum_pass_fraction_gain"]
            and r["gated_rate_ci_high"]<=max(abs(r["gated_rate_baseline"]),1e-12)*cfg["minimum_rate_relative_gain"]
            and r["case_mse_ci_low"] is not None
            and -r["case_mse_ci_low"]<=max(abs(r["case_mse_baseline"]),1e-12)*cfg["minimum_mse_relative_gain"]
            for r in latest)
        recommendation["decision"]=("current_validation_plateau_may_seal_or_expand_validation" if no_gain
                                    else "improvement_or_uncertainty_supports_next_size")
        recommendation["practical_thresholds"]={k:cfg[k] for k in cfg if k.startswith("minimum_")}
    return {"step":"7g","completed_sizes":stages,"learning_curve":curve,"paired_layout_contrasts":contrasts,
            "recommendation":recommendation,"reserved_test_generated":False,
            "interpretation":"Conditional validation diagnostics after checkpoint selection; not final-test confidence intervals or a guarantee against overfitting."}


def plot(output,summary=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output=Path(output)
    summary=summarize(output) if summary is None else summary
    target=output/"figures";target.mkdir(exist_ok=True)
    rows=[r for r in summary["learning_curve"] if r["selection_rule"]=="physical" and r["phase"]=="validation"]
    fig,axes=plt.subplots(1,3,figsize=(15,4.5))
    metrics=[("pass_fraction_mean","Tỷ lệ đạt validation (%)",100),
             ("gated_rate_mean","Rate Bob, lượt trượt tính 0",1),
             ("case_mse_mean","MSE validation",1)]
    for architecture,style in (("reference","-o"),("small","--s")):
        for kind,color in (("gnn","tab:blue"),("mlp","tab:orange")):
            group=sorted((r for r in rows if r["architecture_name"]==architecture and r["kind"]==kind),key=lambda r:r["train_layouts"])
            for ax,(metric,label,scale) in zip(axes,metrics):
                ax.plot([r["train_layouts"] for r in group],[r[metric]*scale for r in group],style,color=color,
                        label=f"{kind.upper()} {architecture}")
                ax.set_xlabel("Số bố trí train");ax.set_ylabel(label);ax.grid(alpha=.25)
    axes[0].set_ylim(0,105);axes[0].legend(fontsize=8)
    fig.suptitle("Cùng bố trí validation; checkpoint chọn bằng validation; chưa mở test cuối")
    fig.tight_layout();fig.savefig(target/"learning_curve.png",dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    latest=max(summary["completed_sizes"])
    for architecture,marker in (("reference","o"),("small","s")):
        for kind,color in (("gnn","tab:blue"),("mlp","tab:orange")):
            vals=[r for r in rows if r["architecture_name"]==architecture and r["kind"]==kind and r["train_layouts"]==latest]
            for r in vals:
                axes[0].scatter(r["parameters"],r["gated_rate_mean"],marker=marker,color=color,label=f"{kind.upper()} {architecture}")
                axes[1].scatter(r["training_seconds_mean"],r["gated_rate_mean"],marker=marker,color=color)
    axes[0].set_xlabel("Số tham số");axes[1].set_xlabel("Thời gian train trung bình (s)")
    for ax in axes:ax.set_ylabel("Rate Bob validation, lượt trượt tính 0");ax.grid(alpha=.25)
    axes[0].legend(fontsize=8);fig.suptitle(f"Train {latest} bố trí; chi phí tạo nhãn PSO được báo riêng")
    fig.tight_layout();fig.savefig(target/"model_size_and_training_cost.png",dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    for architecture,style in (("reference","-o"),("small","--s")):
        for kind,color in (("gnn","tab:blue"),("mlp","tab:orange")):
            for ax,metric in zip(axes,("case_mse_mean","pass_fraction_mean")):
                for phase,alpha in (("validation",1.),("train_probe",.35)):
                    group=sorted([r for r in summary["learning_curve"] if r["selection_rule"]=="physical"
                        and r["architecture_name"]==architecture and r["kind"]==kind and r["phase"]==phase],key=lambda r:r["train_layouts"])
                    ax.plot([r["train_layouts"] for r in group],[r[metric]*(100 if metric.startswith("pass") else 1) for r in group],
                        style,color=color,alpha=alpha,label=f"{kind.upper()} {architecture} {phase}")
                ax.set_xlabel("Số bố trí train");ax.grid(alpha=.25)
    axes[0].set_ylabel("MSE");axes[1].set_ylabel("Tỷ lệ đạt vật lý (%)");axes[1].set_ylim(0,105)
    axes[0].legend(fontsize=6);fig.suptitle("Train probe là tập con cố định; validation được dùng chọn checkpoint")
    fig.tight_layout();fig.savefig(target/"train_probe_vs_validation.png",dpi=160);plt.close(fig)
    return {"figures":["learning_curve.png","model_size_and_training_cost.png","train_probe_vs_validation.png"]}


def refresh(output,plots=True):
    output=Path(output)
    summary=summarize(output)
    write_json(output/"summary.json",summary)
    table(output/"learning_curve.csv",summary["learning_curve"])
    table(output/"paired_layout_contrasts.csv",summary["paired_layout_contrasts"])
    write_json(output/"recommendation.json",summary["recommendation"])
    if plots and summary["completed_sizes"]:
        plot(output,summary)
    return summary


def baseline_diagnostics(training):
    """Analyze already-opened 7f results, not any reserved 7g/7h test layout."""
    training=Path(training)
    physics=read_json(training/"source_physics_plan.json")["base_config"]
    k2=physics["kappa"]**2
    q=float(physics["requirements"]["qos_public"][0])
    rows=[r for p in sorted((training/"test_audit").glob("*.json")) if not p.name.endswith(".receipt.json")
          for r in read_json(p)["rows"]]
    report={"step":"7g","source":"previously observed frozen 7f test; diagnostic development data only",
            "source_selection_is_not_modified":True,"models":{}}
    streams=[]
    for kind in ("gnn","mlp"):
        part=[r for r in rows if r["kind"]==kind and r["selection_rule"]=="physical"]
        counts=Counter()
        failed=[r for r in part if not r["passed"]]
        for row in part:
            k=row["n_public"]
            mb=row["allocation"][0]
            public=np.maximum(np.asarray(row["allocation"][1:-1],float),0.)
            public=public/public.sum()*(1-mb)
            mp=public[row["order"]]
            A=mb+np.cumsum(mp[::-1])[::-1]-mp+k2
            caps=np.log2(1+mp/A)
            cap=(1+k2)*2**(-k*q)-k2
            if not row["passed"]:
                counts["mb_above_global_qos_cap"]+=int(mb>cap)
                counts["some_public_ceiling_below_qos"]+=int(np.any(caps<q))
                counts["some_public_mean_below_qos"]+=int(np.any(np.asarray(row["public_mean_rate_sic"])<q))
                counts["some_public_lower_below_qos"]+=int(np.any(np.asarray(row["public_rate_lower_sic"])<q))
            if k==6:
                for j in range(k):
                    streams.append({"kind":kind,"seed":row["seed"],"case_id":row["case_id"],"layout_id":row["layout_id"],
                        "n_public":k,"xi":row["xi"],"power_dbm":row["power_dbm"],"passed":row["passed"],"mb":mb,
                        "global_qos_mb_cap":cap,"sic_rank":j+1,"public_id_zero_based":row["order"][j],
                        "public_fraction":float(mp[j]),"qos_ceiling_margin":float(mp[j]-(2**q-1)*A[j]),
                        "public_rate_ceiling":float(caps[j]),"public_mean_rate":row["public_mean_rate_sic"][j],
                        "public_rate_lower":row["public_rate_lower_sic"][j]})
        report["models"][kind]={"outputs":len(part),"passed":sum(r["passed"] for r in part),
            "k6_outputs":sum(r["n_public"]==6 for r in part),
            "k6_passed":sum(r["passed"] for r in part if r["n_public"]==6),
            "failed_outputs":len(failed),"failure_counts_overlapping":dict(counts),
            "failed_unique_cases":len({r["case_id"] for r in failed})}
    return report,streams
