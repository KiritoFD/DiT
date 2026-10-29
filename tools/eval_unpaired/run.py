"""Restartable frozen-evaluator screening of existing artifacts; no generator training."""
import argparse
from contextlib import nullcontext
import fcntl
import json
import time
from pathlib import Path
import numpy as np
from .protocol import read,save,csv_rows,discover,reference_bank
from .networks import Evaluators
from .networks import identity
from .metrics import retrieval,distribution,grouped_ci


def compact(value):return {k:v for k,v in value.items() if k!='rows'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--machine',required=True,choices=['4090','48'])
    p.add_argument('--out',required=True);p.add_argument('--reference-csv',required=True)
    p.add_argument('--local-checkpoint');p.add_argument('--no-iqa',action='store_true');p.add_argument('--device',default='cuda')
    p.add_argument('--iqa-metrics',nargs='+',default=['musiq-ava','musiq','topiq_nr'])
    p.add_argument('--jobs',help='Verified extra/historical jobs JSON; replaces auto-discovery')
    a=p.parse_args();root=Path(a.out);root.mkdir(parents=True,exist_ok=True)
    with (open('/tmp/dit_gpu0.lock','a+') if a.device=='cuda' else nullcontext()) as owner:
        if owner is not None:fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
        # Exclusive ownership is also checked against all external compute contexts.
        from tools.style_matrix.runtime.serial import gpu_idle
        if owner is not None:gpu_idle(2107710 if a.machine=='4090' else 0)
        import torch
        torch.set_num_threads(4)
        jobs=read(a.jobs) if a.jobs else discover(a.machine);manifest=root/'manifest.json'
        contract=dict(code={f.name:identity(f) for f in Path(__file__).parent.glob('*.py')},
            reference=identity(a.reference_csv),global_checkpoint=identity('assets/style_disc_ce_holdout.pt'),
            local_checkpoint=identity(a.local_checkpoint) if a.local_checkpoint else None,
            iqa=[] if a.no_iqa else a.iqa_metrics)
        inputs=[]
        for job in jobs:
            import hashlib
            if hashlib.sha256(Path(job['csv']).read_bytes()).hexdigest()!=job['csv_sha256']:
                raise ValueError('Evaluation CSV changed')
            if len(csv_rows(job['csv']))!=job['n']:raise ValueError('Evaluation count changed')
            inputs.append(dict(job=[job['arm'],job['step'],job['protocol']],
                images=[identity(Path(job['samples'])/f'{prefix}{i}.png') for prefix in ['g','gt'] for i in range(job['n'])]))
        contract['sample_inputs']=inputs
        if manifest.exists():
            old=read(manifest)
            if old['jobs']!=jobs:raise ValueError('Artifact set changed; use a new output version')
            if old.get('contract')!=contract:raise ValueError('Evaluator/reference contract changed; use a new output version')
        else:save(manifest,dict(version='eval_unpaired_v1_screen',machine=a.machine,jobs=jobs,
            contract=contract,
            reference_csv=a.reference_csv,training_reference=True,created=time.time(),
            missing=['OCR: no verified checkpoint','E-L: no checkpoint' if not a.local_checkpoint else '',
                     'CleanFID/KID formal panel B pending','human preference calibration pending']))
        refs=reference_bank(a.reference_csv);save(root/'reference_rows.json',refs)
        nets=Evaluators(a.device,not a.no_iqa,a.local_checkpoint,a.iqa_metrics);save(root/'networks.json',nets.provenance)
        cache=root/'reference_features.npz'
        if cache.exists():features=dict(np.load(cache))
        else:
            # Real reference feature extraction does not spend IQA compute on 3k training images.
            iqa=nets.iqa;nets.iqa={}
            features=nets.extract([r['image_path'] for r in refs],32);nets.iqa=iqa
            np.savez(cache,**features)
        if not (root/'quality_calibration.json').exists():
            paths=[r['image_path'] for r in refs[::max(1,len(refs)//24)]][:24]
            save(root/'quality_calibration.json',nets.calibrate(paths))
        summaries=[]
        for job in jobs:
            name=f"{job['arm']}_{job['step']}_{job['protocol']}";out=root/name;out.mkdir(exist_ok=True)
            if (out/'result.json').exists():summaries.append(read(out/'result.json'));continue
            rows=csv_rows(job['csv']);samples=Path(job['samples']);result=dict(job=job)
            for prefix in ['g','gt']:
                cache=out/f'{prefix}_features.npz'
                if cache.exists():query=dict(np.load(cache))
                else:
                    query=nets.extract([samples/f'{prefix}{i}.png' for i in range(len(rows))]);np.savez(cache,**query)
                scores={}
                for metric,values in query.items():
                    if metric.startswith('E-'):
                        for k in [32,64,128]:
                            retr=retrieval(values,features[metric],rows,refs,k)
                            save(out/f'{prefix}_{metric}_K{k}_rows.json',retr)
                            scores[f'{metric}_retrieval_K{k}']=compact(retr)
                        scores[f'{metric}_cross_dataset_K64']=compact(retrieval(values,features[metric],rows,refs,64,True))
                        scores[f'{metric}_style_distribution']=distribution(values,features[metric],rows,refs)
                    else:scores[metric]=grouped_ci(values,rows)
                result[prefix]=scores
            result['interpretation']='Uncalibrated IQA research scores; train-reference style metrics; not an aesthetic ranking or heldout distribution claim'
            save(out/'result.json',result);summaries.append(result);save(root/'summary.json',summaries)
            print('COMPLETED',name,flush=True)
        save(root/'summary.json',summaries);save(root/'complete.json',dict(jobs=len(jobs),time=time.time()))


if __name__=='__main__':main()
