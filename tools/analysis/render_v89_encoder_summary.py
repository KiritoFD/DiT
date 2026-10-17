"""Scientific overview of a completed encoder run, including failed generalization."""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run',required=True)
    a=p.parse_args()
    run=Path(a.run)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=[json.loads(s) for s in (run/'metrics.jsonl').read_text(encoding='utf-8').splitlines()]
    training=[s for s in rows if 'ce' in s]
    validation=[s for s in rows if 'validation' in s]
    acceptance=json.loads((run/'acceptance.json').read_text(encoding='utf-8'))
    selected=acceptance['selected_step']
    fig,axs=plt.subplots(2,2,figsize=(13,8),constrained_layout=True)
    x=[r['step']/1000 for r in training]
    for key,weight,label in [('ce',1,'CE'),('supcon',.3,'0.3 SupCon'),('std',.1,'0.1 STD')]:
        axs[0,0].plot(x,[r[key]*weight for r in training],label=label,linewidth=1.5)
    axs[0,0].set_yscale('symlog',linthresh=1e-5)
    axs[0,0].set_title('Training losses: low CE does not establish generalization')
    axs[0,0].legend()
    vx=[r['step']/1000 for r in validation]
    for key,label in [('prototype_top1','Prototype Top1 (micro)'),('prototype_macro_top1','Prototype Top1 (macro)'),('slot_ce_top1','CE head Top1')]:
        axs[0,1].plot(vx,[r['validation'][key] for r in validation],marker='.',label=label)
    axs[0,1].axhline(validation[0]['validation']['script_conditioned_chance'],ls=':',color='gray',label='Same-script chance')
    axs[0,1].set_title('Character-held-out validation (fixed capped queries)')
    axs[0,1].legend(fontsize=8)
    axs[1,0].plot(vx,[r['validation']['cross_char_map'] for r in validation],marker='.',color='tab:green')
    axs[1,0].set_title('Cross-character retrieval mAP on validation')
    axs[1,0].set_ylim(0,1)
    for ax in (axs[0,0],axs[0,1],axs[1,0]):
        ax.axvline(selected/1000,ls='--',color='black',alpha=.5)
        ax.set_xlabel('Training step (thousands)'); ax.grid(alpha=.2)
    values=acceptance['full_holdout_exported_prototypes']
    labels=['Train CE','Val char CE','Test char CE','Test source CE']
    heights=[acceptance['train_slot_ce_top1']]+[values[k]['slot_ce_top1'] for k in ('val_char','test_char','test_source')]
    bars=axs[1,1].bar(labels,heights,color=['#587e4b','#d99c3b','#b87345','#b45454'])
    axs[1,1].set_ylim(0,1.13); axs[1,1].set_title('Selected checkpoint: train vs full held-out CE accuracy')
    for bar,value in zip(bars,heights):
        axs[1,1].text(bar.get_x()+bar.get_width()/2,value+.02,f'{value:.1%}',ha='center')
    fig.suptitle(f'V89 style encoder | {run.name} | best validation step {selected:,}\n'
                 '20k steps / batch512 / 256D / CE + 0.3 SupCon + 0.1 STD',fontsize=14)
    fig.savefig(run/'summary.png',dpi=160)
    plt.close(fig)
    print('V89_SUMMARY_RENDERED',run/'summary.png')


if __name__=='__main__':
    main()
