"""Exploratory pooled Clean-FID/KID, explicitly small-sample; never per-writer FID."""
import argparse
import fcntl
from contextlib import nullcontext
from pathlib import Path
import numpy as np
import torch
from tools.eval_unpaired.protocol import read,save,csv_rows


def fid_low_rank(a,b):
    """Exact empirical covariance Bures trace via low-rank SVD, without PCA."""
    a=a.astype(np.float64);b=b.astype(np.float64)
    mu=a.mean(0)-b.mean(0);ac=(a-a.mean(0))/np.sqrt(len(a)-1);bc=(b-b.mean(0))/np.sqrt(len(b)-1)
    cross_trace=np.linalg.svd(ac@bc.T,compute_uv=False).sum()
    return float(mu@mu+(ac*ac).sum()+(bc*bc).sum()-2*cross_trace)


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--out',required=True)
    p.add_argument('--device',default='cuda');p.add_argument('--resident-pid',type=int,default=0);a=p.parse_args()
    root=Path(a.out);root.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4)
    from cleanfid.features import build_feature_extractor
    from cleanfid.fid import get_files_features,kernel_distance
    with (open('/tmp/dit_gpu0.lock','a+') if a.device=='cuda' else nullcontext()) as lock:
        if lock is not None:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        from tools.style_matrix.runtime.serial import gpu_idle
        if lock is not None:gpu_idle(a.resident_pid)
        model=build_feature_extractor('clean',device=torch.device(a.device),use_dataparallel=False)
        results=[]
        for job in read(a.manifest)['jobs']:
            name=f"{job['arm']}_{job['step']}_{job['protocol']}";cache=root/f'{name}.npz'
            if cache.exists():features=dict(np.load(cache))
            else:
                features={key:get_files_features([str(Path(job['samples'])/f'{key}{i}.png') for i in range(job['n'])],
                    model=model,num_workers=2,batch_size=32,device=torch.device(a.device),mode='clean',verbose=False)
                    for key in ['g','gt']}
                np.savez(cache,**features)
            x,y=features['g'],features['gt'];np.random.seed(90);n=len(y)//2
            groups={}
            for i,row in enumerate(csv_rows(job['csv'])):groups.setdefault((row['calligrapher'],row['script']),[]).append(i)
            left=[];right=[];rng=np.random.default_rng(90)
            for ids in groups.values():
                ids=rng.permutation(ids);half=len(ids)//2
                left.extend(ids[:half]);right.extend(ids[half:2*half])
            item=dict(job=job,clean_fid=fid_low_rank(x,y),kid=kernel_distance(x,y,num_subsets=100,max_subset_size=100),
                real_split_fid=fid_low_rank(y[left],y[right]),
                real_split_kid=kernel_distance(y[left],y[right],num_subsets=100,max_subset_size=100),
                matched_n=len(left),matched_gen_fid=fid_low_rank(x[left],y[right]),
                matched_gen_kid=kernel_distance(x[left],y[right],num_subsets=100,max_subset_size=100),
                status='small-sample pooled diagnostic, not formal panel-B ranking; real-split comparisons use matched writer-script counts and N')
            results.append(item);save(root/'summary.json',results);print('CLEAN_COMPLETE',name,flush=True)
        save(root/'complete.json',dict(jobs=len(results)))


if __name__=='__main__':main()
