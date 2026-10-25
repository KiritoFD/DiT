"""All saved cohort checkpoints: full sampling, exact nested membership, fast summaries."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
from statistics import mean, median
import subprocess
import sys

from .datasets import DEFAULT_ROOT, image_key, read_rows, resolve_collection, sha, write_rows
from tools.data_cohorts.train_serial import save, stamp

TRAIN = Path('assets/results/clean_cohorts_20261021/launch_30k_lr7e5')
OLD = Path('assets/audits/cohort_matrix_30k_20261022_v2')


def discover():
    jobs=[]
    for model in (20,10,5,2):
        paths=sorted((TRAIN/f'top{model}_u114').glob('*/checkpoints/*.pt'))
        seen=set()
        for path in paths:
            step=int(path.stem)
            if step in seen:raise ValueError(f'Ambiguous model{model} step{step}')
            seen.add(step);jobs.append(dict(model=model,step=step,checkpoint=str(path)))
        if not seen:raise FileNotFoundError(f'No saved Top{model} checkpoints')
    latest={m:max(j['step'] for j in jobs if j['model']==m) for m in (20,10,5,2)}
    return sorted(jobs,key=lambda j:(j['step']!=latest[j['model']],-j['step'],-j['model']))


def completed_protocol(directory):
    from src.eval.protocol import EvalRequest,is_complete
    protocols=list(Path(directory).glob('eval_samples_ctrl/step*/*/protocol.json'))
    if len(protocols)!=1:return False
    raw=json.loads(protocols[0].read_text());raw.pop('eval_id',None)
    request=EvalRequest(**raw)
    return is_complete(request,str(protocols[0].parent))[0]


def reusable(job,weight_sha):
    directory=OLD/f"model{job['model']}_test{job['model']}"
    identity=directory/'identity.json'
    if not identity.exists() or not (directory/'RESULT.json').exists():return None
    value=json.loads(identity.read_text())
    source=resolve_collection(f"top{job['model']}/test")
    if (value['checkpoint_sha256']==weight_sha and value['test_sha256']==sha(source)
            and value['step']==job['step'] and value['ema'] and value['cfg']==1.
            and value['steps']==50 and value['sampler']=='heun' and completed_protocol(directory)):
        return directory
    return None


def subset_rows(full_items,metrics,requested):
    if len(full_items)!=len(metrics):raise ValueError('Per-image metrics count mismatch')
    indexed={int(r['idx']):r for r in metrics}
    if set(indexed)!=set(range(len(full_items))):raise ValueError('Missing/duplicate sample indices')
    keyed={}
    for i,item in enumerate(full_items):
        row=indexed[i]
        if row.get('img_id') and item.get('img_id') and row['img_id']!=item['img_id']:
            raise ValueError('Sample identity mismatch')
        row=dict(row,image_path=image_key(item))
        keyed[image_key(item)]=(item,row)
    result=[]
    for item in requested:
        base,row=keyed[image_key(item)]
        if str(base.get('eval_noise_seed'))!=str(item.get('eval_noise_seed')):
            raise ValueError('Nested sample noise differs')
        result.append(dict(row))
    if len({r['image_path'] for r in result})!=len(requested):raise ValueError('Repeated sample')
    return result


def summarize(items):
    mapping=dict(ssim_mean='ssim',lpips_mean='lpips',ink_iou_mean='ink_iou',
        frag_ratio='frag_ratio',soft_ink_ratio='soft_ink_ratio',mse_mean='mse',
        skel_iou_mean='skel_iou',ink_ssim_mean='ink_ssim',hole_pred='hole_pred',hole_gt='hole_gt')
    result={name:mean(float(r[field]) for r in items) for name,field in mapping.items()}
    result.update(n=len(items),ssim_med=median(float(r['ssim']) for r in items),
        near_blank_rate=mean(float(r['soft_ink_ratio'])<.1 for r in items))
    return result


def publish(output,results):
    save(output/'results.json',results)
    write_rows(output/'results.csv',results)
    paired=defaultdict(dict)
    for row in results:paired[(row['model'],row['step'],row['tier'])][row['split']]=row
    gaps=[]
    for (model,step,tier),pair in paired.items():
        if len(pair)!=2:continue
        tr,te=pair['train_half'],pair['test']
        gaps.append(dict(model=model,step=step,tier=tier,n_train=tr['n'],n_test=te['n'],
            train_minus_test_ssim=tr['ssim_mean']-te['ssim_mean'],
            test_minus_train_lpips=te['lpips_mean']-tr['lpips_mean'],
            train_minus_test_ink_iou=tr['ink_iou_mean']-te['ink_iou_mean']))
    if gaps:write_rows(output/'fit_gaps.csv',gaps)


def run(args):
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    jobs=discover()
    expected=sum(2*sum(t<=j['model'] for t in (20,10,5,2)) for j in jobs)
    save(output/'plan.json',dict(created=stamp(),jobs=jobs,expected_results=expected,
        state='EMA',sampler='Heun50',cfg=1.,batch=args.batch,vae_batch=args.vae_batch,
        precision='Native per-image CSV: SSIM rounded to 4dp; other metrics 4-5dp. All subsets use same aggregation.',
        train_probe_manifest_sha256=sha(DEFAULT_ROOT/'train_probes_v1/manifest.json')))
    status=dict(state='running',started=stamp(),expected_results=expected,completed=0)
    results=[]
    try:
        for job in jobs:
            weight_sha=sha(job['checkpoint'])
            for split in ('test','train_half'):
                status.update(model=job['model'],step=job['step'],split=split,updated=stamp())
                save(output/'status.json',status)
                directory=reusable(job,weight_sha) if split=='test' else None
                collection=f"top{job['model']}/{split}"
                source=resolve_collection(collection)
                if directory is None:
                    directory=output/f"model{job['model']}_step{job['step']:07d}_{split}"
                    with (output/(directory.name+'.log')).open('xb') as log:
                        subprocess.run([sys.executable,'-u','-m','tools.cohort_eval.run',
                            '--checkpoint',job['checkpoint'],'--collection',collection,'--out',str(directory),
                            '--batch',str(args.batch),'--vae-batch',str(args.vae_batch)],
                            stdout=log,stderr=subprocess.STDOUT,check=True)
                if not completed_protocol(directory):raise RuntimeError(f'Incomplete artifacts: {directory}')
                full=read_rows(source);raw=read_rows(directory/'eval_stdskel_batch.csv')
                for tier in (20,10,5,2):
                    if tier>job['model']:continue
                    subset=resolve_collection(f'top{tier}/{split}')
                    per_image=subset_rows(full,raw,read_rows(subset))
                    tag=f"model{job['model']}_step{job['step']:07d}_top{tier}_{split}"
                    selected=output/(tag+'_per_image.csv');write_rows(selected,per_image)
                    result=dict(model=job['model'],step=job['step'],tier=tier,split=split,
                        **summarize(per_image),checkpoint_sha256=weight_sha,
                        dataset_sha256=sha(subset),generated_from=str(directory),per_image=str(selected))
                    results.append(result)
                status['completed']=len(results);publish(output,results);save(output/'status.json',status)
        if len(results)!=expected:raise RuntimeError('Incomplete matrix')
        status.update(state='done',updated=stamp());save(output/'status.json',status)
    except BaseException as exc:
        status.update(state='failed',error=str(exc),updated=stamp());save(output/'status.json',status)
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True)
    p.add_argument('--batch',type=int,default=32);p.add_argument('--vae-batch',type=int,default=8)
    run(p.parse_args())


if __name__=='__main__':main()
