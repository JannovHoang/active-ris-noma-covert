"""Scientific plots from frozen 7c CSV/JSON only; no Torch in plot process."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot(output):
    output=Path(output)
    plan=json.loads((output/"plan.json").read_text(encoding="utf-8"))
    result=json.loads((output/"summary.json").read_text(encoding="utf-8"))
    selection=json.loads((output/"selection_frozen.json").read_text(encoding="utf-8"))
    if not result["complete"]:raise ValueError("Complete the frozen test audit before plotting")
    folder=output/"figures";folder.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10})
    seeds=plan["config"]["seeds"]
    fig,axes=plt.subplots(2,len(seeds),figsize=(max(6,4*len(seeds)),6.5),squeeze=False)
    for ri,kind in enumerate(("gnn","mlp")):
        for ci,seed in enumerate(seeds):
            with (output/"fits"/f"{kind}_s{seed}"/"history.csv").open(encoding="utf-8-sig") as f:history=list(csv.DictReader(f))
            ax=axes[ri,ci]
            for key,label in (("train_weighted_loss","Train"),("validation_weighted_loss","Validation")):
                ax.plot([int(r["epoch"]) for r in history],np.maximum([float(r[key]) for r in history],1e-12),label=label)
            for rule,label,color in (("weighted_loss","Chọn bằng loss","#777777"),("physical","Chọn bằng vật lý","#228833")):
                view=selection["views"][f"{kind}_s{seed}_{rule}"]
                ax.axvline(view["epoch"],ls="--",color=color,label=label)
            ax.set(title=f"{kind.upper()} — seed {seed}",xlabel="Epoch",ylabel="Weighted loss",yscale="log")
            ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.suptitle("Huấn luyện dùng chung nhiều K — chọn checkpoint chỉ bằng validation")
    fig.tight_layout(rect=(0,0,1,.95));fig.savefig(folder/"training_curves.png",dpi=170);plt.close(fig)
    groups=result["metrics"]["across_seeds"]
    labels=[("GNN" if g["kind"]=="gnn" else "MLP gộp")+ (" — chọn loss" if g["selection_rule"]=="weighted_loss" else " — chọn vật lý") for g in groups]
    reference=next(r for r in result["metrics"]["by_view"] if r["kind"]=="pso_label_reference")
    colors=["#4477aa","#228833","#cc6677","#aa4499","#999999"]
    labels.append("Nhãn PSO tham khảo")
    fig,axes=plt.subplots(1,2,figsize=(11.5,4.8))
    passes=[100*g["pass_fraction_mean"] for g in groups]+[100*reference["pass_fraction"]]
    pass_error=[100*g["pass_fraction_std"] for g in groups]+[0.]
    rates=[g["gated_rate_mean"] for g in groups]+[reference["gated_rate"]]
    rate_error=[g["gated_rate_std"] for g in groups]+[0.]
    for ax,values,errors,title,xlabel in ((axes[0],passes,pass_error,"Qua kiểm định mới","Tỷ lệ (%)"),
                                         (axes[1],rates,rate_error,"Rate Bob có tính lượt trượt","bit/s/Hz")):
        ax.barh(range(len(labels)),values,xerr=errors,color=colors,capsize=3)
        ax.set(yticks=range(len(labels)),yticklabels=labels,title=title,xlabel=xlabel)
        ax.invert_yaxis();ax.grid(axis="x",alpha=.2)
    axes[0].set_xlim(0,105)
    fig.text(.5,.015,"Trung bình ± độ lệch chuẩn giữa các seed train; không phải mô hình ensemble. PSO là nhãn offline, chưa so thời gian/ngân sách online.",ha="center",fontsize=8,color="#555555")
    fig.tight_layout(rect=(0,.075,1,1));fig.savefig(folder/"test_comparison.png",dpi=170);plt.close(fig)
    by_k=result["metrics"]["by_k"]
    ks=sorted({r["n_public"] for r in by_k})
    fig,axes=plt.subplots(1,2,figsize=(11.5,4.8))
    for index,group in enumerate(groups):
        label=labels[index]
        selected=[r for r in by_k if r["kind"]==group["kind"] and r["selection_rule"]==group["selection_rule"]]
        for ax,field,scale in ((axes[0],"pass_fraction",100),(axes[1],"gated_rate",1)):
            means=[scale*np.mean([r[field] for r in selected if r["n_public"]==k]) for k in ks]
            ax.plot(ks,means,"o-",color=colors[index],label=label)
    for ax in axes:
        for k in plan["unseen_public_counts"]:ax.axvspan(k-.35,k+.35,color="#ddaa33",alpha=.15)
        ax.set(xlabel="Số public user K",xticks=ks);ax.grid(alpha=.2);ax.legend(fontsize=7)
    axes[0].set(title="Tỷ lệ đạt theo K",ylabel="Tỷ lệ (%)",ylim=(-3,105))
    axes[1].set(title="Rate Bob theo K",ylabel="bit/s/Hz")
    fig.text(.5,.015,"Cùng các bố trí test, công suất và DEP; vùng tô là K chưa có trong train. Rate tính lượt trượt bằng 0.",ha="center",fontsize=9,color="#555555")
    fig.tight_layout(rect=(0,.075,1,1));fig.savefig(folder/"test_by_k.png",dpi=170);plt.close(fig)
    files=["training_curves.png","test_comparison.png","test_by_k.png"]
    (folder/"figure_manifest.json").write_text(json.dumps({"step":"7c","figures":files},indent=2)+"\n",encoding="utf-8")
    return {"step":"7c","figures":files}
