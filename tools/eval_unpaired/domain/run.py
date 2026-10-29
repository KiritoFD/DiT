"""Re-score all completed frozen feature caches; no image/model transfer or training."""
import argparse
from pathlib import Path
import numpy as np
from tools.eval_unpaired.protocol import read,save,csv_rows
from .statistics import verification,distributions


def main():
    p=argparse.ArgumentParser();p.add_argument('--roots',nargs='+',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    out=Path(a.out);results=[]
    for value in a.roots:
        root=Path(value);refs=read(root/'reference_rows.json');bank=dict(np.load(root/'reference_features.npz'))
        for item in read(root/'summary.json'):
            job=item['job'];name=f"{job['arm']}_{job['step']}_{job['protocol']}";dest=out/name/'result.json'
            if dest.exists():results.append(read(dest));continue
            row=read(job['csv']) if job['csv'].endswith('.json') else csv_rows(job['csv']);result=dict(job=job,source_features=str(root))
            for prefix in ['g','gt']:
                features=dict(np.load(root/name/f'{prefix}_features.npz'));scores={}
                for key,x in features.items():
                    if not key.startswith('E-'):continue
                    v=verification(x,bank[key],row,refs);save(out/name/f'{prefix}_{key}_verification.json',v)
                    scores[key]=dict(verification={k:v for k,v in v.items() if k!='rows'},distribution=distributions(x,bank[key],row,refs))
                result[prefix]=scores
            save(dest,result);results.append(result);save(out/'summary.json',results);print('DOMAIN_COMPLETE',name,flush=True)
    save(out/'complete.json',dict(jobs=len(results)))


if __name__=='__main__':main()
