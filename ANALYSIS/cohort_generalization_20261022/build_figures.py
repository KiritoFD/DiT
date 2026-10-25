"""Render the completed cohort audit and print compact factual report tables."""
from collections import defaultdict
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'assets/audits/cohort_all_ckpt_20261022'
OUT=Path(__file__).resolve().parent


def read(path):
    with path.open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))


def main():
    status=json.loads((DATA/'status.json').read_text())
    assert status['state']=='done' and status['completed']==112
    rows=read(DATA/'results.csv')
    assert len(rows)==len({(r['model'],r['step'],r['tier'],r['split']) for r in rows})==112
    for r in rows:
        assert int(r['n'])==int(r['tier'])*(20 if r['split']=='test' else 10)
    index={(int(r['model']),int(r['step']),int(r['tier']),r['split']):r for r in rows}
    colors={2:'#9467bd',5:'#2ca02c',10:'#ff7f0e',20:'#1f77b4'}
    figures=OUT/'figures';figures.mkdir(exist_ok=True)
    stats={}
    for tier in (2,5,10,20):
        fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
        plt.rcParams['font.family']='DejaVu Sans'
        print('\nTIER',tier)
        models=[m for m in (2,5,10,20) if m>=tier]
        stats[str(tier)]={}
        for model in models:
            steps=sorted({int(r['step']) for r in rows if int(r['model'])==model and int(r['tier'])==tier})
            for split in ('test','train_half'):
                rs=[index[(model,s,tier,split)] for s in steps]
                for ax,metric,title in zip(axes,('ssim_mean','lpips_mean'),('SSIM (higher better)','VGG-LPIPS (lower better)')):
                    ax.plot(np.array(steps)/1000,[float(r[metric]) for r in rs],
                        color=colors[model],linestyle='-' if split=='test' else '--',
                        marker='o' if split=='test' else '.',label=f'M{model} {split}')
                    ax.set(title=title,xlabel='Training step (k)');ax.grid(alpha=.2)
            latest=steps[-1]
            tr=index[(model,latest,tier,'train_half')];te=index[(model,latest,tier,'test')]
            first_tr=index[(model,steps[0],tier,'train_half')];first_te=index[(model,steps[0],tier,'test')]
            best=min(steps,key=lambda s:float(index[(model,s,tier,'test')]['lpips_mean']))
            bestrow=index[(model,best,tier,'test')]
            val=dict(latest_step=latest,train=tr,test=te,best_lpips_step=best,best_lpips=bestrow['lpips_mean'],
                first_train=first_tr,first_test=first_te,
                delta_train_ssim=float(tr['ssim_mean'])-float(first_tr['ssim_mean']),
                delta_test_ssim=float(te['ssim_mean'])-float(first_te['ssim_mean']),
                delta_test_lpips=float(te['lpips_mean'])-float(first_te['lpips_mean']))
            stats[str(tier)][str(model)]=val
            print(f"M{model} latest{latest//1000}k SSIM {float(tr['ssim_mean']):.4f}/{float(te['ssim_mean']):.4f} LPIPS {float(tr['lpips_mean']):.4f}/{float(te['lpips_mean']):.4f} IoU {float(tr['ink_iou_mean']):.4f}/{float(te['ink_iou_mean']):.4f} Frag {float(tr['frag_ratio']):.3f}/{float(te['frag_ratio']):.3f} bestLP {float(bestrow['lpips_mean']):.4f}@{best//1000}k delta5k-latest trainSSIM {val['delta_train_ssim']:+.4f} testSSIM {val['delta_test_ssim']:+.4f} testLP {val['delta_test_lpips']:+.4f}")
            if tier==2:
                t=index[(model,10000,2,'test')];f=index[(model,10000,2,'train_half')]
                print('AT10K',model,'SSIM',f['ssim_mean'],t['ssim_mean'],'LPIPS',f['lpips_mean'],t['lpips_mean'],'IOU',t['ink_iou_mean'])
        axes[1].legend(fontsize=8,ncol=2)
        fig.suptitle(f'Top{tier}: test n={tier*20}, train-half n={tier*10}; EMA / CFG1 / Heun50')
        fig.savefig(figures/f'top{tier}.png',dpi=160)
        plt.close(fig)
    comparisons=[]
    for a,b,step,tier in ((20,10,30000,10),(20,5,30000,5),(20,2,10000,2),(5,2,10000,2),(20,5,30000,2)):
        paths=[]
        for model in (a,b):
            p=DATA/f'model{model}_step{step:07d}_top{tier}_test_per_image.csv'
            paths.append({r['image_path']:r for r in read(p)})
        assert paths[0].keys()==paths[1].keys()
        groups=defaultdict(list)
        for key,row in paths[0].items():groups[row['char']].append(key)
        clusters=list(groups.values());rng=np.random.default_rng(90)
        ci={}
        for metric in ('ssim','lpips','ink_iou'):
            diffs={k:float(paths[0][k][metric])-float(paths[1][k][metric]) for k in paths[0]}
            values=[]
            for _ in range(2000):
                chosen=rng.integers(0,len(clusters),len(clusters))
                vals=[diffs[k] for i in chosen for k in clusters[i]]
                values.append(np.mean(vals))
            ci[metric]=dict(delta=float(np.mean(list(diffs.values()))),ci95=np.quantile(values,[.025,.975]).tolist())
        record=dict(a=a,b=b,step=step,tier=tier,n=len(paths[0]),character_clusters=len(clusters),metrics=ci)
        comparisons.append(record);print('PAIRED',json.dumps(record))
    (OUT/'analysis.json').write_text(json.dumps(dict(status=status,tiers=stats,paired=comparisons),ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__':main()
