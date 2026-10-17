"""Render small local audit artifacts; no dataset or checkpoint required."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();out=Path(a.out)
    r=json.loads((out/'results.json').read_text(encoding='utf-8'))
    heat=np.load(out/'heatmap_data.npz')
    for key,title,file in [('offset','Quill-like writer offset from script mean','writer_features.png'),('residual','Quill-like GT-STD residual minus script mean','std_residual.png')]:
        fig,ax=plt.subplots(figsize=(15,8));v=heat[key];limit=max(float(np.quantile(np.abs(v),.98)),.1)
        im=ax.imshow(v,aspect='auto',cmap='coolwarm',vmin=-limit,vmax=limit);fig.colorbar(im,ax=ax,label='Train-standardized offset (clipped color scale)')
        ax.set_xticks(np.arange(0,96,8),[f'{i*15}deg' for i in range(12)])
        ax.set_yticks(range(len(v)),[f'S{i:02d}' for i in range(len(v))]);ax.set_title(title+' / descriptive, content-source confounded')
        fig.tight_layout();fig.savefig(out/file,dpi=150);plt.close(fig)
    keys=['ink','width','quill','hinge','profile','fft_gray','fft_binary','layout','latent','geometry_combo','all']
    fig,axes=plt.subplots(1,2,figsize=(16,6),sharey=True)
    for ax,mode in zip(axes,['raw','delta_std']):
        for j,protocol in enumerate(['critic_native_source','character_holdout','source_holdout']):
            vals=[r['results'][protocol][k][mode]['matched']['by_char']['accuracy'] for k in keys]
            ci=[r['results'][protocol][k][mode]['matched']['by_char']['ci95'] for k in keys]
            errors=np.array([[v-l,h-v] for v,(l,h) in zip(vals,ci)]).T
            ax.errorbar(np.arange(len(keys))+(j-1)*.12,vals,yerr=errors,fmt='o',capsize=2,label=protocol)
        ax.axhline(.5,color='gray',linestyle='--');ax.set_ylim(0,1);ax.set_xticks(range(len(keys)),keys,rotation=55,ha='right');ax.set_title(mode+' / character-cluster bootstrap 95% CI');ax.grid(axis='y',alpha=.2)
    axes[0].set_ylabel('Matched cross-character 2AFC');axes[1].legend(fontsize=8);fig.tight_layout();fig.savefig(out/'crosschar_ranking.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(14,5));names=['v84A_200k','v88_170k'];keys=list(r['models'][names[0]])
    x=np.arange(len(keys))
    for j,name in enumerate(names):
        axes[0].bar(x+(j-.5)*.35,[r['models'][name][k]['paired_standardized_abs_error'] for k in keys],.35,label=name)
    axes[0].set_xticks(x,keys,rotation=55,ha='right');axes[0].set_title('Same-glyph GT gap / lower is better');axes[0].legend()
    histkeys=[k for k in keys if 'paired_js_distance' in r['models'][names[0]][k]];x=np.arange(len(histkeys))
    for j,name in enumerate(names):axes[1].bar(x+(j-.5)*.35,[r['models'][name][k]['paired_js_distance'] for k in histkeys],.35,label=name)
    axes[1].set_xticks(x,histkeys,rotation=55,ha='right');axes[1].set_title('Same-glyph histogram JS distance / lower is better');axes[1].legend();fig.tight_layout();fig.savefig(out/'model_gaps.png',dpi=150);plt.close(fig)
    labels=[list(x) for x in heat['slots']]
    (out/'slot_labels.json').write_text(json.dumps(labels,ensure_ascii=False,indent=2),encoding='utf-8')
    print('FINGERPRINT_FIGURES_COMPLETE')


if __name__=='__main__':main()
