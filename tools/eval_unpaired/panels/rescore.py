"""Versioned statistics-only pass over frozen embeddings; never reruns a network."""
import argparse
from pathlib import Path
import numpy as np
from tools.eval_unpaired.protocol import read,save,csv_rows
from tools.eval_unpaired.metrics import retrieval,distribution,grouped_ci


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    root=Path(a.root);out=Path(a.out);refs=read(root/'reference_rows.json');bank=dict(np.load(root/'reference_features.npz'))
    results=[]
    for item in read(root/'summary.json'):
        job=item['job'];name=f"{job['arm']}_{job['step']}_{job['protocol']}";rows=csv_rows(job['csv']);result=dict(job=job)
        for prefix in ['g','gt']:
            features=dict(np.load(root/name/f'{prefix}_features.npz'));scores={}
            for key,values in features.items():
                if not key.startswith('E-'):scores[key]=grouped_ci(values,rows);continue
                for k in [32,64,128]:
                    decision=retrieval(values,bank[key],rows,refs,k);save(out/name/f'{prefix}_{key}_K{k}_rows.json',decision)
                    scores[f'{key}_retrieval_K{k}']={k:v for k,v in decision.items() if k!='rows'}
                scores[f'{key}_style_distribution']=distribution(values,bank[key],rows,refs)
                decision=retrieval(values,bank[key],rows,refs,64,True)
                scores[f'{key}_cross_dataset_K64']={k:v for k,v in decision.items() if k!='rows'}
            result[prefix]=scores
        result['identity']=dict(feature_source=str(root),statistics_version='writer-macro MMD + coverage + cluster CI v1')
        save(out/name/'result.json',result);results.append(result)
    save(out/'summary.json',results);save(out/'complete.json',dict(jobs=len(results)))


if __name__=='__main__':main()
