"""Wait for independent table, resume Top20@30k frozen, audit and evaluate 31k/33k."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from tools.cohort_eval.datasets import read_rows,resolve_collection,sha,write_rows
from tools.cohort_eval.batch import subset_rows,summarize,completed_protocol
from tools.data_cohorts.train_serial import save,stamp


def run(args):
    root=Path(args.root)
    with (root/'continuation.lock').open('x') as f:f.write(str(os.getpid()))
    status=dict(state='waiting_table',started=stamp(),runner_pid=os.getpid())
    def publish():status['updated']=stamp();save(root/'status.json',status)
    publish()
    try:
        provenance=root/'table_v2/provenance.json'
        while not provenance.exists():
            proc=Path(f'/proc/{args.pretrain_pid}')
            if not proc.exists() or proc.joinpath('stat').read_text().split()[2]=='Z':
                raise RuntimeError('Independent table pretraining failed; inspect pretrain_v2.log')
            time.sleep(15)
        import torch
        source=Path(args.checkpoint);source_sha=sha(source)
        ck=torch.load(source,map_location='cpu',weights_only=False)
        assert ck['train_steps']==30000
        cfg=dict(ck['config']);del ck
        cfg.update(freeze_writer_table=True,writer_table_file=str(root/'table_v2/callig_table.npy'),
            eval_before_training=True,stop_at_step=33000,max_steps=50000,lr_decay_steps=50000,
            ckpt_every=1000,eval_every=1000,output='assets/results/writer_table_repair_top20_20261022')
        target=Path(cfg['output'])
        if target.exists():raise FileExistsError(target)
        save(root/'train_config.json',cfg)
        save(root/'experiment.json',dict(source=str(source),source_sha256=source_sha,
            source_state='raw+optimizer+EMA resume',start=30000,stop=33000,
            independent_table_provenance=str(provenance),table_sha256=sha(cfg['writer_table_file']),
            change='Replace real writer rows after resume, freeze writer table during FM, keep trunk trainable',
            loss='original FM only; no auxiliary style objective',
            limitations='No equal-budget no-swap control; evaluates combined replacement+freeze+3000 extra steps'))
        status['state']='training';publish()
        with (root/'train.log').open('xb') as log:
            child=subprocess.Popen([sys.executable,'-u','-m','src.unet.train','--config',str(root/'train_config.json'),
                                    '--resume',str(source)],stdout=log,stderr=subprocess.STDOUT)
            status['child_pid']=child.pid;publish()
            if child.wait():raise RuntimeError('Generator continuation failed; inspect train.log')
        runs=list(target.glob('*/checkpoints'))
        if len(runs)!=1:raise RuntimeError('Ambiguous continuation run')
        import numpy as np
        expected=torch.from_numpy(np.load(cfg['writer_table_file']))
        audits=[];results=[]
        baseline=read_rows(Path('assets/audits/cohort_all_ckpt_20261022/results.csv'))
        baseline={(int(r['tier']),r['split']):r for r in baseline if r['model']=='20' and r['step']=='30000'}
        for step in (31000,33000):
            path=runs[0]/f'{step:07d}.pt';status.update(state='evaluating',step=step);publish()
            ck=torch.load(path,map_location='cpu',weights_only=False)
            for kind in ('delta','ema'):
                w=ck[kind]['writer.weight']
                if not torch.equal(w[:-1],expected):raise RuntimeError(f'Frozen table drift: {step}/{kind}')
            audits.append(dict(step=step,raw_and_ema_table_bit_identical=True))
            del ck
            for split in ('test','train_half'):
                directory=root/f'eval_{step}_{split}'
                with (root/f'eval_{step}_{split}.log').open('xb') as log:
                    subprocess.run([sys.executable,'-u','-m','tools.cohort_eval.run','--checkpoint',str(path),
                        '--collection',f'top20/{split}','--out',str(directory),'--batch','32','--vae-batch','8'],
                        stdout=log,stderr=subprocess.STDOUT,check=True)
                if not completed_protocol(directory):raise RuntimeError('Incomplete repaired-model evaluation')
                full=read_rows(resolve_collection(f'top20/{split}'))
                raw=read_rows(directory/'eval_stdskel_batch.csv')
                for tier in (2,5,10,20):
                    chosen=subset_rows(full,raw,read_rows(resolve_collection(f'top{tier}/{split}')))
                    metric=summarize(chosen);before=baseline[(tier,split)]
                    row=dict(step=step,tier=tier,split=split,**metric)
                    for key in ('ssim_mean','lpips_mean','ink_iou_mean','frag_ratio'):
                        row['baseline_'+key]=float(before[key]);row['delta_'+key]=metric[key]-float(before[key])
                    results.append(row)
                    write_rows(root/f'per_image_{step}_top{tier}_{split}.csv',chosen)
                write_rows(root/'comparison.csv',results);save(root/'comparison.json',results)
        if sha(source)!=source_sha:raise RuntimeError('Source checkpoint changed')
        save(root/'frozen_table_audit.json',audits)
        status.update(state='done',n_evaluations=len(results));publish()
    except BaseException as exc:
        status.update(state='failed',error=str(exc));publish();raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True);p.add_argument('--checkpoint',required=True)
    p.add_argument('--pretrain-pid',required=True,type=int)
    run(p.parse_args())


if __name__=='__main__':main()
