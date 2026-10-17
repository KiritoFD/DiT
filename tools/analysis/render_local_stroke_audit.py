"""Render fixed-protocol local statistics and content-confounding diagnostic."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();out=Path(a.out)
    report=json.loads((out/'results.json').read_text(encoding='utf-8'))
    keys=['ink_edge','geometry','bend_width','combined']
    fig,axes=plt.subplots(1,2,figsize=(13,5),sharey=True)
    table=[]
    for ax,protocol in zip(axes,['character_holdout','source_holdout']):
        entry=report['protocols'][protocol]
        for j,subset in enumerate(['main','image_disjoint']):
            values=[entry['families'][k][subset]['by_char']['accuracy'] for k in keys]
            ci=[entry['families'][k][subset]['by_char']['ci95'] for k in keys]
            error=np.array([[v-low,high-v] for v,(low,high) in zip(values,ci)]).T
            ax.errorbar(np.arange(len(keys))+(j-.5)*.12,values,yerr=error,fmt='o',capsize=3,label=subset)
            for k,v,c in zip(keys,values,ci):table.append([protocol,subset,k,v,*c,entry['families'][k][subset]['by_char']['n']])
        ax.axhline(.5,color='gray',ls='--');ax.axhline(.6,color='green',ls=':',label='engineering gate .60')
        ax.set_xticks(range(len(keys)),keys,rotation=20);ax.set_ylim(0,1)
        ax.set_title(f'{protocol}: nominal {entry["nominal_n"]}, common {entry["common_valid_n"]}, disjoint {entry["image_disjoint_n"]}')
        ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2)
    axes[0].set_ylabel('2AFC / character-cluster bootstrap 95% CI')
    fig.suptitle('Width/angle/curvature-conditioned local patches; same shared groups in both distances')
    fig.tight_layout();fig.savefig(out/'local_ranking.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,5),sharey=True)
    for ax,protocol in zip(axes,['character_holdout','source_holdout']):
        diag=report['protocols'][protocol]['global_diagnostic'];labels=[];same=[];different=[]
        for key in ('quill','hinge'):
            for metric in ('old_std_mse','small_floor_std_mse','hellinger_mse'):
                result=diag[key][metric];labels.append(key+'\n'+metric.replace('_mse',''))
                same.append(result['content_control_samechar_negative']['by_char']['accuracy'])
                different.append(result['content_control_diffchar_negative']['by_char']['accuracy'])
        x=np.arange(len(labels));ax.bar(x-.18,same,.36,label='negative: same character')
        ax.bar(x+.18,different,.36,label='negative: different character')
        ax.axhline(.5,color='gray',ls='--');ax.set_xticks(x,labels,rotation=35,ha='right')
        ax.set_title(protocol+' / diagnostic only, not gating');ax.set_ylim(0,1);ax.legend(fontsize=8)
    axes[0].set_ylabel('2AFC; positives and anchor subset held fixed')
    fig.tight_layout();fig.savefig(out/'distance_diagnostic.png',dpi=150);plt.close(fig)
    with (out/'ranking.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.writer(stream);writer.writerow(['protocol','subset','family','accuracy','ci_low','ci_high','n']);writer.writerows(table)
    print('LOCAL_FIGURES_COMPLETE',report['decision'])


if __name__=='__main__':main()
