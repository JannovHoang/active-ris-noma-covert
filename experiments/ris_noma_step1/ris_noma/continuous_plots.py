"""Plots of continuous scenarios and PSO teacher labels, not network accuracy."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .learning_data import read_json
from .continuous_dataset import read_lines
from .multik_dataset import write_json

def plot(output):
    output=Path(output);summary=read_json(output/'summary.json');rows=read_lines(output/'ledger.jsonl')
    root=output/'figures';root.mkdir(exist_ok=True);figures=[]
    def save(fig,name):
        fig.tight_layout();fig.savefig(root/name,dpi=160,bbox_inches='tight');plt.close(fig);figures.append(name)
    colors={'train':'tab:blue','validation':'tab:orange','test':'tab:green','test_unseen_k':'tab:red','query':'tab:purple'}
    fig,ax=plt.subplots(figsize=(9,5))
    for group in sorted({r['role'] for r in rows}):
        part=[r for r in rows if r['role']==group]
        ax.scatter([r['power_dbm'] for r in part],[r['xi'] for r in part],s=22,alpha=.7,color=colors[group],label=group)
    ax.set(xlabel='Tổng công suất (dBm)',ylabel='DEP yêu cầu',title='Các cặp đầu vào liên tục đã khai báo')
    ax.legend();ax.grid(alpha=.2);save(fig,'continuous_parameters.png')
    fig,ax=plt.subplots(figsize=(9,5));by=summary['by_role'];x=np.arange(len(by));width=.25
    for offset,key,label in ((-width,'planned','Tình huống đã khai báo'),(0,'labels','Có nhãn sau selection'),(width,'audit_passed','Qua audit độc lập')):
        bars=ax.bar(x+offset,[r[key] for r in by],width,label=label)
        ax.bar_label(bars,padding=3,fontsize=9)
    ax.set(xticks=x,xticklabels=[r['role'] for r in by],ylabel='Số tình huống',title='Độ phủ nhãn PSO; không phải độ chính xác GNN')
    ax.legend();ax.grid(axis='y',alpha=.2);ax.margins(y=.15);save(fig,'label_coverage.png')
    labeled=[r for r in rows if r['audit'] is not None]
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for k in sorted({r['n_public'] for r in labeled}):
        part=[r for r in labeled if r['n_public']==k]
        axes[0].scatter([r['xi'] for r in part],[r['audit']['dep_lower']-r['xi'] for r in part],s=16,alpha=.65,label=f'K={k}')
        axes[1].scatter([r['xi'] for r in part],[r['audit']['qos_lower_margin'] for r in part],s=16,alpha=.65,label=f'K={k}')
    for ax,title in zip(axes,('Độ dư cận DEP trên bank audit','Độ dư cận QoS public trên bank audit')):
        ax.axhline(0,color='black',linestyle='--',linewidth=1);ax.set(xlabel='DEP yêu cầu',ylabel='Cận dưới trừ yêu cầu',title=title);ax.grid(alpha=.2)
    if labeled:axes[0].legend()
    save(fig,'audit_margins.png')
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for k in sorted({r['n_public'] for r in labeled}):
        part=[r for r in labeled if r['n_public']==k]
        axes[0].scatter([r['xi'] for r in part],[r['audit']['gated_bob_rate'] for r in part],s=16,alpha=.65,label=f'K={k}')
        axes[1].scatter([r['power_dbm'] for r in part],[r['audit']['gated_bob_rate'] for r in part],s=16,alpha=.65,label=f'K={k}')
    for ax,xlabel in zip(axes,('DEP yêu cầu','Tổng công suất (dBm)')):
        ax.set(xlabel=xlabel,ylabel='Rate Bob, trượt audit tính 0 (bit/s/Hz)');ax.grid(alpha=.2)
    axes[0].set_title('Nhãn PSO theo DEP');axes[1].set_title('Nhãn PSO theo công suất')
    if labeled:axes[0].legend()
    save(fig,'teacher_rate.png')
    result={'figures':figures,'scope':summary['scope'],'mode':summary['mode'],
        'note':'PSO teacher and dataset diagnostics only; GNN/MLP not trained in 7e.'}
    write_json(root/'figure_manifest.json',result);return result
