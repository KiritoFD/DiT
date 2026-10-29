"""Fixed-writer, character-cluster intervals for paired per-query verification."""
import argparse
import json
from pathlib import Path
import numpy as np


def compare(a,b,seed=90):
    def index(value):return {(r['writer'],r['script'],r['character']):r for r in value['rows']}
    aa,bb=index(a),index(b)
    if set(aa)!=set(bb):raise ValueError('Paired query identities differ')
    result={}
    for metric in ['auc','eer','ap','tar_at_eval_far01']:
        groups={}
        for key in sorted(aa):
            writer,script,char=key
            groups.setdefault(writer,{}).setdefault(char,[]).append(aa[key][metric]-bb[key][metric])
        rng=np.random.default_rng(seed);boot=np.zeros(2000);point=[]
        for writer,chars in groups.items():
            sums=np.array([sum(v) for v in chars.values()]);counts=np.array([len(v) for v in chars.values()])
            samples=rng.integers(len(sums),size=(len(boot),len(sums)))
            boot+=(sums[samples].sum(1)/counts[samples].sum(1))/len(groups)
            point.append(sums.sum()/counts.sum())
        result[metric]=dict(delta_pp=100*float(np.mean(point)),lo_pp=100*float(np.quantile(boot,.025)),
            hi_pp=100*float(np.quantile(boot,.975)),queries=len(aa),writers=len(groups))
    return dict(metrics=result,interpretation='Macro of per-query ROC statistics; not pooled ROC intervals. Fixed writers, resample characters; single seed, exploratory, multiple comparisons uncorrected.')


def main():
    p=argparse.ArgumentParser();p.add_argument('--evidence',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    root=Path(a.evidence);scores={arm:json.loads((root/f'{arm}_verification.json').read_text(encoding='utf-8')) for arm in ['S0','S4','S5']}
    result={f'{x}-{y}':compare(scores[x],scores[y]) for x,y in [('S4','S0'),('S5','S0'),('S5','S4')]}
    Path(a.out).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
