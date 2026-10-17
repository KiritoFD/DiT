"""Render aggregate R with coverage, and original hard single-glyph 2AFC."""
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
    families=['pixel_D1_global','gray_CLS','gray_allograph','gray_combined','binary_allograph','binary_combined']
    fig,axes=plt.subplots(2,2,figsize=(15,10));sizes=[1,4,16,32]
    for row,protocol in enumerate(['character_holdout','source_holdout']):
        for col,cohort in enumerate(['all_eligible','same_32_cohort']):
            ax=axes[row,col]
            for family in families:
                records=[r['aggregation'][protocol][str(k)][cohort][family] for k in sizes]
                y=[v['R'] if v['R'] is not None else np.nan for v in records]
                line,=ax.plot(sizes,y,'o-',label=family,linewidth=1.4)
                if family in ('gray_allograph','gray_combined') and all(v['R'] is not None for v in records):
                    ci=[v['pair_cluster_bootstrap']['R_ci95'] for v in records]
                    ax.fill_between(sizes,[c[0] for c in ci],[c[1] for c in ci],alpha=.10,color=line.get_color())
            ax.axhline(1,color='gray',ls='--');ax.axhline(1.2,color='green',ls=':')
            ax.set_xscale('log',base=2);ax.set_xticks(sizes,[str(k) for k in sizes]);ax.set_xlabel('Distinct glyphs per writer/set')
            pairs=[r['aggregation'][protocol][str(k)]['n_eligible_pairs'] for k in sizes]
            common=r['aggregation'][protocol]['32']['n_pairs_32_cohort']
            ax.set_title(f'{protocol} / {cohort}\npairs by K={pairs}; fixed cohort={common}')
            ax.set_ylabel('R = mean between / mean within distance');ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8);fig.suptitle('Matched character distributions; A/B characters disjoint. Source directories are only proxies.')
    fig.tight_layout();fig.savefig(out/'aggregation_R.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(14,5),sharey=True)
    keys=r['features'];x=np.arange(len(keys))
    for ax,protocol in zip(axes,['character_holdout','source_holdout']):
        records=[r['single_image_matched'][protocol][k]['by_char'] for k in keys]
        y=[v['accuracy'] for v in records];ci=[v['ci95'] for v in records]
        err=np.array([[v-low,high-v] for v,(low,high) in zip(y,ci)]).T
        ax.errorbar(x,y,yerr=err,fmt='o',capsize=3);ax.set_xticks(x,keys,rotation=45,ha='right')
        ax.axhline(.5,color='gray',ls='--');ax.set_ylim(0,1);ax.set_title(f'{protocol}, n={records[0]["n"]}');ax.grid(axis='y',alpha=.2)
    axes[0].set_ylabel('Original ink/width-matched single-glyph 2AFC')
    fig.tight_layout();fig.savefig(out/'single_glyph_2afc.png',dpi=150);plt.close(fig)
    print('ALLOGRAPH_FIGURES_COMPLETE')


if __name__=='__main__':main()
