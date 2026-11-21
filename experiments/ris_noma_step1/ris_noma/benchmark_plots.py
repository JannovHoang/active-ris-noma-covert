"""Standalone figures for step 5c. This module never imports PyTorch.

Run plotting in a separate process from optimization on Windows to avoid
loading Torch and Matplotlib OpenMP runtimes together in a mixed environment.
"""
from pathlib import Path
import json
import hashlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

METHODS=("pso_full","pso_short","gnn_pso_short","mlp_pso_short")
LABELS={"pso_full":"PSO đầy đủ","pso_short":"PSO ngắn ngẫu nhiên","gnn_pso_short":"GNN + PSO ngắn","mlp_pso_short":"MLP + PSO ngắn"}
COLORS={"pso_full":"#24557c","pso_short":"#ca7b27","gnn_pso_short":"#24805d","mlp_pso_short":"#925593"}
MARKERS={"pso_full":"o","pso_short":"s","gnn_pso_short":"^","mlp_pso_short":"D"}


def read_json(path):return json.loads(Path(path).read_text(encoding="utf-8"))


def read_results(output):
    output=Path(output)
    if not read_json(output/"benchmark_status.json")["benchmark_complete"]:
        raise ValueError("Benchmark is incomplete")
    frozen=read_json(output/"search_frozen.json")
    searches,audits=[],[]
    for key,digest in frozen["search_records_sha256"].items():
        p=output/"search"/key
        if hashlib.sha256(p.read_bytes()).hexdigest()!=digest:raise ValueError("Frozen search changed")
        searches.append(read_json(p))
        p=output/"audit"/key
        if hashlib.sha256(p.read_bytes()).hexdigest()!=read_json(str(p)+".receipt.json")["sha256"]:
            raise ValueError("Committed audit changed")
        audits.append(read_json(p))
    return searches,audits


def across_layouts(rows,key):
    lids=sorted({r["layout_id"] for r in rows})
    values=[np.mean([r[key] for r in rows if r["layout_id"]==lid]) for lid in lids]
    return float(np.mean(values)),float(np.std(values,ddof=1)) if len(values)>1 else 0.


def save_figure(fig,path,caption):
    fig.text(.5,.012,caption,ha="center",fontsize=8,color="#555555")
    fig.tight_layout(rect=(0,.055,1,1));fig.savefig(path,dpi=180,facecolor="white");plt.close(fig)


def plot_benchmark(output):
    output=Path(output); searches,rows=read_results(output)
    figures=output/"figures";figures.mkdir(exist_ok=True)
    xis=sorted({r["xi"] for r in rows});n=len(rows)//4;nl=len({r["layout_id"] for r in rows})
    note=f"Pilot đã được xem trước: {nl} bố trí; {n} lượt/phương pháp. Dải: SD giữa các trung bình theo bố trí, không phải khoảng tin cậy."
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    fig,axes=plt.subplots(1,2,figsize=(12,4.7))
    for m in METHODS:
        groups=[[r for r in rows if r["method"]==m and r["xi"]==x] for x in xis]
        for ax,key,scale in [(axes[0],"bob_rate",1.),(axes[1],"bound_pass",100.)]:
            stats=[across_layouts(g,key) for g in groups]
            mean=np.array([a for a,b in stats])*scale;sd=np.array([b for a,b in stats])*scale
            ax.plot(xis,mean,marker=MARKERS[m],color=COLORS[m],label=LABELS[m],linewidth=1.8)
            ax.fill_between(xis,mean-sd,mean+sd,color=COLORS[m],alpha=.10)
    axes[0].set(title="Rate Bob của mọi phương án trả về",xlabel="DEP yêu cầu",ylabel="Rate trung bình (bit/s/Hz)")
    axes[1].set(title="Đạt đầy đủ kiểm tra trên bank mới",xlabel="DEP yêu cầu",ylabel="Tỷ lệ đạt (%)",ylim=(-5,105))
    for ax in axes:ax.set_xticks(xis);ax.grid(alpha=.2);ax.legend(fontsize=8)
    save_figure(fig,figures/"rate_and_feasibility.png",note+" Đọc rate cùng tỷ lệ đạt.")
    fig,axes=plt.subplots(1,2,figsize=(12,4.7))
    for m in METHODS:
        times=[across_layouts([r for r in rows if r["method"]==m and r["xi"]==x],"total_decision_seconds") for x in xis]
        means=np.array([t[0] for t in times]);sd=np.array([t[1] for t in times])
        axes[0].plot(xis,means,marker=MARKERS[m],color=COLORS[m],label=LABELS[m]);axes[0].fill_between(xis,means-sd,means+sd,color=COLORS[m],alpha=.10)
        full=np.array([across_layouts([r for r in rows if r["method"]=="pso_full" and r["xi"]==x],"total_decision_seconds")[0] for x in xis])
        axes[1].plot(xis,full/means,marker=MARKERS[m],color=COLORS[m],label=LABELS[m])
    axes[0].set(title="Tìm nghiệm và kiểm tra sau cùng",xlabel="DEP yêu cầu",ylabel="Tổng thời gian trung bình (giây)")
    axes[1].set(title="Tỷ số thời gian so với PSO đầy đủ",xlabel="DEP yêu cầu",ylabel="Tỷ số các thời gian trung bình")
    axes[1].axhline(1,color="gray",ls=":")
    for ax in axes:ax.set_xticks(xis);ax.grid(alpha=.2);ax.legend(fontsize=8)
    save_figure(fig,figures/"time_and_speedup.png",note+" Gồm dựng bank, feature, suy luận, PSO và audit; không gồm train/nạp model một lần.")
    fig,axes=plt.subplots(1,2,figsize=(12,4.7))
    for m in METHODS:
        histories=[r["history"] for r in searches if r["method"]==m]
        x=np.array([r["evaluations"] for r in histories[0]])
        scores=np.array([[h["gbest_score"] if h["gbest_feasible"] else np.nan for h in run] for run in histories])
        feasible=np.array([[h["gbest_feasible"] for h in run] for run in histories])
        count=np.isfinite(scores).sum(axis=0)
        avg=np.divide(np.nansum(scores,axis=0),count,out=np.full(len(x),np.nan),where=count>0)
        axes[0].plot(x,avg,color=COLORS[m],label=LABELS[m]);axes[1].plot(x,feasible.mean(axis=0)*100,color=COLORS[m],label=LABELS[m])
    axes[0].set(title="Rate tốt nhất trong các lượt đã có nghiệm khả thi",xlabel="Số lần đánh giá phương án PSO",ylabel="Rate Bob trên bank tìm kiếm (bit/s/Hz)")
    axes[1].set(title="Có nghiệm đáp ứng điều kiện bank tìm kiếm",xlabel="Số lần đánh giá phương án PSO",ylabel="Tỷ lệ lượt chạy (%)",ylim=(-5,105))
    for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize=8)
    save_figure(fig,figures/"convergence.png","Khả thi ở đây theo Gamma/sample trên bank tìm kiếm với biên cố định; kiểm định bank độc lập được báo riêng.")
    fig,axes=plt.subplots(1,2,figsize=(12,4.7))
    for m in METHODS:
        groups=[[r for r in rows if r["method"]==m and r["xi"]==x] for x in xis]
        axes[0].plot(xis,[min(r["dep_lower_margin"] for r in g) for g in groups],marker=MARKERS[m],color=COLORS[m],label=LABELS[m])
        axes[1].plot(xis,[across_layouts(g,"audit_gated_bob_rate")[0] for g in groups],marker=MARKERS[m],color=COLORS[m],label=LABELS[m])
    axes[0].axhline(0,color="gray",ls=":")
    axes[0].set(title="Phần dư cận dưới DEP nhỏ nhất",xlabel="DEP yêu cầu",ylabel="Cận dưới DEP trừ yêu cầu")
    axes[1].set(title="Chỉ tiêu rate có tính đến thất bại kiểm định",xlabel="DEP yêu cầu",ylabel="Rate trung bình, quy ước 0 nếu trượt (bit/s/Hz)")
    for ax in axes:ax.set_xticks(xis);ax.grid(alpha=.2);ax.legend(fontsize=8)
    save_figure(fig,figures/"margins_and_gated_rate.png","Chỉ tiêu bên phải đặt rate bằng 0 nếu trượt bất kỳ kiểm tra nào; đây là quy ước tính điểm, không phải goodput vật lý.")
    summary=read_json(output/"benchmark_summary.json");by={r["method"]:r for r in summary["summary"]}
    fig,axes=plt.subplots(2,2,figsize=(12,8.2));positions=np.arange(4);labels=[LABELS[m] for m in METHODS];colors=[COLORS[m] for m in METHODS]
    for ax,key,title,unit,scale in [(axes[0,0],"bound_pass_fraction","Đạt đầy đủ kiểm tra","Tỷ lệ (%)",100),
                                  (axes[0,1],"bob_rate_all_mean","Rate của mọi phương án","bit/s/Hz",1),
                                  (axes[1,0],"total_seconds_mean","Tìm nghiệm và audit","Giây/phương án",1),
                                  (axes[1,1],"audit_gated_bob_rate_mean","Rate với thất bại kiểm định tính bằng 0","bit/s/Hz",1)]:
        values=[by[m][key]*scale for m in METHODS]
        bars=ax.barh(positions,values,color=colors,height=.6);ax.set_yticks(positions,labels);ax.invert_yaxis();ax.set(title=title,xlabel=unit);ax.grid(axis="x",alpha=.2);ax.set_axisbelow(True)
        maxv=max(values) or 1.;ax.set_xlim(0,maxv*1.24)
        for bar,value in zip(bars,values):ax.text(value+maxv*.02,bar.get_y()+bar.get_height()/2,f"{value:.2f}",va="center",fontsize=9)
    save_figure(fig,figures/"overview.png",f"Thử nghiệm phát triển: {nl} bố trí, {n} lượt/phương pháp. Cần đọc chất lượng, tỷ lệ đạt và thời gian cùng nhau.")
    result={"figures":[str(p.name) for p in sorted(figures.glob('*.png'))],"layouts":nl,"runs_per_method":n,"scope":"development pilot"}
    (figures/"figure_manifest.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    return result
