"""7f figures separate regression error, physical feasibility and training randomness."""
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .learning_data import read_json
from .multik_dataset import write_json

COLORS={'gnn':'tab:blue','mlp':'tab:orange'}
LABELS={'weighted_loss':'chọn loss','physical':'chọn vật lý'}


def plot(output):
    output=Path(output);summary=read_json(output/'summary.json');plan=read_json(output/'plan.json')
    figures=output/'figures';figures.mkdir(exist_ok=True);saved=[]
    def save(fig,name):
        fig.tight_layout();fig.savefig(figures/name,dpi=160,bbox_inches='tight');plt.close(fig);saved.append(name)
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for column,kind in enumerate(('gnn','mlp')):
        for seed in plan['config']['seeds']:
            with (output/'fits'/f'{kind}_s{seed}'/'history.csv').open(encoding='utf-8-sig',newline='') as handle:
                history=list(csv.DictReader(handle))
            epoch=[int(r['epoch']) for r in history]
            axes[0,column].plot(epoch,[float(r['train_weighted_loss']) for r in history],label=f'train s{seed}')
            axes[0,column].plot(epoch,[float(r['validation_weighted_loss']) for r in history],linestyle='--',label=f'val s{seed}')
            axes[1,column].plot(epoch,[float(r['validation_case_mse']) for r in history],label=f'seed {seed}')
        axes[0,column].set(title=kind.upper()+' — loss có trọng số',xlabel='Epoch',ylabel='Loss')
        axes[1,column].set(title=kind.upper()+' — MSE validation',xlabel='Epoch',ylabel='MSE')
        for row in range(2):axes[row,column].grid(alpha=.2);axes[row,column].legend(fontsize=8)
    save(fig,'learning_curves.png')
    pooled=[r for r in summary['metrics']['across_seeds'] if r['regime']=='all']
    fig,axes=plt.subplots(1,3,figsize=(15,4.5));x=np.arange(len(pooled))
    names=[r['kind'].upper()+'\n'+LABELS[r['selection_rule']] for r in pooled]
    for ax,field,scale,title,ylabel in zip(axes,('pass_fraction','gated_rate','case_mse'),(100,1,1),
        ('Qua kiểm định vật lý','Rate Bob, lượt trượt tính 0','MSE phân bổ so với nhãn PSO'),('Tỷ lệ (%)','bit/s/Hz','MSE')):
        mean=[r[field+'_mean']*scale for r in pooled];std=[r[field+'_std']*scale for r in pooled]
        bars=ax.bar(x,mean,yerr=std,capsize=4,color=[COLORS[r['kind']] for r in pooled],alpha=.8)
        ax.set(xticks=x,xticklabels=names,title=title,ylabel=ylabel);ax.grid(axis='y',alpha=.2)
        if field=='pass_fraction':ax.set_ylim(0,108)
        else:ax.margins(y=.2)
        ax.bar_label(bars,fmt='%.3g',padding=4,fontsize=9)
    reference=next(r for r in summary['metrics']['by_view'] if r['kind']=='pso_label_reference')
    axes[0].axhline(100*reference['pass_fraction'],color='black',linestyle=':',linewidth=1.2)
    axes[1].axhline(reference['gated_rate'],color='black',linestyle=':',linewidth=1.2,label='Nhãn PSO offline')
    axes[1].legend(fontsize=8)
    fig.suptitle('Trung bình ± độ lệch chuẩn giữa seed train; không phải khoảng tin cậy theo bố trí',fontsize=11)
    save(fig,'test_comparison.png')
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for kind in ('gnn','mlp'):
        for rule in ('weighted_loss','physical'):
            rows=[r for r in summary['metrics']['by_k'] if r['kind']==kind and r['selection_rule']==rule]
            ks=sorted({r['n_public'] for r in rows})
            for ax,field,scale in zip(axes,('pass_fraction','gated_rate'),(100,1)):
                values=[np.mean([r[field] for r in rows if r['n_public']==k])*scale for k in ks]
                ax.plot(ks,values,marker='o',linestyle='-' if rule=='physical' else '--',color=COLORS[kind],label=kind.upper()+' '+LABELS[rule])
    for ax,title,ylabel in zip(axes,('Qua kiểm định theo K','Rate Bob theo K, trượt tính 0'),('Tỷ lệ (%)','bit/s/Hz')):
        ax.set(xlabel='Số public user K',ylabel=ylabel,title=title);ax.grid(alpha=.2);ax.legend(fontsize=8)
        for k in plan['unseen_public_counts']:ax.axvspan(k-.5,k+.5,color='gray',alpha=.1)
    reference=[r for r in summary['metrics']['by_k'] if r['kind']=='pso_label_reference']
    for ax,field,scale in zip(axes,('pass_fraction','gated_rate'),(100,1)):
        ax.plot([r['n_public'] for r in reference],[r[field]*scale for r in reference],color='black',linestyle=':',label='Nhãn PSO offline')
        ax.legend(fontsize=8)
    axes[0].set_ylim(0,108);save(fig,'test_by_k.png')
    for key,filename,xlabel in (('by_dep_bin','test_by_dep_bin.png','Khoảng DEP yêu cầu'),('by_power_bin','test_by_power_bin.png','Khoảng công suất (dBm)')):
        fig,axes=plt.subplots(1,2,figsize=(12,4.5))
        for kind in ('gnn','mlp'):
            for rule in ('weighted_loss','physical'):
                rows=[r for r in summary['metrics'][key] if r['kind']==kind and r['selection_rule']==rule]
                bins=sorted({(r['bin_low'],r['bin_high']) for r in rows})
                for ax,field,scale in zip(axes,('pass_fraction','gated_rate'),(100,1)):
                    values=[np.mean([r[field] for r in rows if (r['bin_low'],r['bin_high'])==interval])*scale for interval in bins]
                    ax.plot(range(len(bins)),values,marker='o',linestyle='-' if rule=='physical' else '--',color=COLORS[kind],label=kind.upper()+' '+LABELS[rule])
                    ax.set(xticks=range(len(bins)),xticklabels=[f'{lo:g}–{hi:g}' for lo,hi in bins])
        for ax,title,ylabel in zip(axes,('Qua kiểm định vật lý','Rate Bob, lượt trượt tính 0'),('Tỷ lệ (%)','bit/s/Hz')):
            ax.set(xlabel=xlabel,ylabel=ylabel,title=title);ax.grid(alpha=.2);ax.legend(fontsize=8)
        axes[0].set_ylim(0,108)
        fig.suptitle('Tổng hợp theo khoảng; các bố trí, K và tham số còn lại có thể khác nhau',fontsize=10)
        save(fig,filename)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5));selection=read_json(output/'selection_frozen.json')
    for kind in ('gnn','mlp'):
        for rule,marker in (('weighted_loss','o'),('physical','s')):
            rows=[v for v in selection['views'].values() if v['kind']==kind and v['selection_rule']==rule]
            axes[0].scatter([r['validation_errors']['weighted_loss'] for r in rows],[100*r['validation_physics']['pass_fraction'] for r in rows],
                marker=marker,color=COLORS[kind],label=kind.upper()+' '+LABELS[rule],s=55)
            test=[r for r in summary['metrics']['by_view'] if r['kind']==kind and r['selection_rule']==rule]
            axes[1].scatter([r['case_mse'] for r in test],[100*r['pass_fraction'] for r in test],marker=marker,color=COLORS[kind],s=55)
    axes[0].set(title='Validation dùng chọn checkpoint',xlabel='Loss validation có trọng số',ylabel='Qua vật lý (%)')
    axes[1].set(title='Test chỉ dùng đánh giá',xlabel='MSE test',ylabel='Qua vật lý (%)')
    for ax in axes:ax.set_ylim(-2,105);ax.grid(alpha=.2)
    axes[0].legend(fontsize=8);save(fig,'error_vs_feasibility.png')
    result={'step':'7f','scope':summary['scope'],'figures':saved,
            'note':'Development pilot; direct predictions only. Seed standard deviation is not a layout confidence interval.'}
    write_json(figures/'figure_manifest.json',result);return result
