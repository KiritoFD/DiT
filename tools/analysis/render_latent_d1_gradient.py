"""Render the independent-transfer audit, not a model-generated illustration."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();out=Path(a.out)
    r=json.loads((out/'results.json').read_text(encoding='utf-8'));v=r['variants']
    names=['baseline','style_0.001','style_0.003','style_0.01','style_0.03','random_0.01']
    labels=['Baseline','Style 0.1%','Style 0.3%','Style 1%','Style 3%','Random 1%'];x=np.arange(len(names))
    fig,axes=plt.subplots(1,2,figsize=(12.5,4.5))
    for j,(key,label,color) in enumerate((('latent_D1','Latent D1 (optimized)','#577590'),('pixel_D1','Pixel D1 (frozen judge)','#06a77d'),('independent_CE','Independent CE (frozen judge)','#f8961e'))):
        axes[0].bar(x+(j-1)*.25,[100*v[n]['means'][key+'_top1'] for n in names],width=.25,label=label,color=color)
    axes[0].set_xticks(x,labels,rotation=25,ha='right');axes[0].set_ylim(0,105);axes[0].set_ylabel('Same-script prototype Top-1 (%)');axes[0].legend(fontsize=8)
    for j,(key,label,color) in enumerate((('pixel_D1_margin','Pixel D1','#06a77d'),('independent_CE_margin','Independent CE','#f8961e'))):
        entries=[v[n]['paired_delta_vs_baseline'][key]['by_character'] for n in names[1:]]
        means=np.array([e['accuracy'] for e in entries]);ci=np.array([e['ci95'] for e in entries])
        axes[1].errorbar(np.arange(5)+j*.1,means*1000,yerr=np.stack((means-ci[:,0],ci[:,1]-means))*1000,fmt='o-',capsize=3,color=color,label=label)
    axes[1].axhline(0,ls=':',color='black');axes[1].set_xticks(np.arange(5),labels[1:],rotation=25,ha='right')
    axes[1].set_ylabel('Prototype margin change x 1000');axes[1].legend(fontsize=8)
    for ax in axes:ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('Latent D1: large self-score gain, weak independent transfer')
    fig.text(.5,.015,'80 supported writer/script outputs, 8 fixed characters, one seed. One-step latent updates only.\nMargins use train-only prototypes excluding query character; error bars cluster by character (n=8).',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.11,1,.94));fig.savefig(out/'gradient_summary.png',dpi=180);plt.close(fig)


if __name__=='__main__':main()
