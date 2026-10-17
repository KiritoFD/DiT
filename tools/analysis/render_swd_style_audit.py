"""Compact scientific figure from frozen SWD results; no model/data access."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();out=Path(a.out)
    report=json.loads((out/'results.json').read_text(encoding='utf-8'))
    methods={'A_latent32':'Latent SWD','B_patch_multiscale':'Patch SWD','C_geometry_patch':'Geometry + Patch',
             'Pixel_D1':'Pixel D1','Latent_D1':'Latent D1','ink_fraction_only':'Ink only'}
    fig,axes=plt.subplots(2,2,figsize=(12,7.5),sharex=True)
    colors=['#577590','#f8961e','#ef476f','#06a77d','#6a4c93','#888888']
    for col,key in enumerate(('test_char','test_source')):
        for (name,label),color in zip(methods.items(),colors):
            # Identical eligible writer-pair cohort at every set size.
            rows=[report['methods'][name]['sets'][key][str(k)]['common32_cohort'] for k in (1,4,16,32)]
            axes[0,col].plot(range(4),[r['R'] for r in rows],'o-',label=label,color=color,lw=1.7)
            axes[1,col].plot(range(4),[100*r['two_afc'] for r in rows],'o-',color=color,lw=1.7)
        axes[0,col].set_title(('Character holdout','Source-proxy holdout')[col])
        axes[0,col].axhline(1,color='black',ls=':',lw=1)
        axes[0,col].set_yscale('log');axes[0,col].set_ylabel('Between / within distance R (log)')
        axes[1,col].axhline(50,color='black',ls=':',lw=1)
        axes[1,col].set_ylim(0,103);axes[1,col].set_ylabel('Set-level 2AFC (%)');axes[1,col].set_xlabel('Distinct characters per set')
        for row in range(2):
            axes[row,col].set_xticks(range(4),[1,4,16,32]);axes[row,col].grid(alpha=.2)
    axes[0,0].legend(fontsize=8,ncol=2)
    fig.suptitle('SWD audit: aggregation helps learned D1, not reliable frozen SWD',fontsize=14)
    fig.text(.5,.015,'Fixed 32-character-eligible cohort: 43 writer pairs (character), 7 (source); 5 trials/pair.\nSource labels are proxies; reused images and shared writers limit independence. SWD uses squared W2.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.07,1,.95));fig.savefig(out/'swd_summary.png',dpi=180);plt.close(fig)


if __name__=='__main__':main()
