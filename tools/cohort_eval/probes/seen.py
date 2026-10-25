"""After X1 training exits, compare fixed seen/test probes with the old Top20."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from tools.cohort_eval.datasets import read_rows,write_rows


def main():
    out=Path('assets/audits/extreme48m_seen_20261022')
    out.mkdir(parents=True,exist_ok=False)
    launch=Path('assets/results/extreme48m_top20_x1_launch_20261022_compiled/status.json')
    state=dict(pid=os.getpid(),phase='waiting_training',completed=0,total=4,
               next_steps='Compare same-step seen/test gaps; do not infer reduced overfitting from gap alone')
    def save():
        (out/'status.json').write_text(json.dumps(state,indent=2))
    save()
    while True:
        training=json.loads(launch.read_text())
        if training['phase']=='completed':break
        if training['phase'].startswith('failed'):raise RuntimeError(training)
        time.sleep(15)
    root=Path('assets/results/extreme48m_top20_x1/20261022-172221-Unet-flow/checkpoints')
    old=read_rows('assets/audits/cohort_all_ckpt_20261022/results.csv')
    results=[]
    for step in (25000,30000):
        for split in ('train_half','test'):
            dest=out/f'step{step:07d}_{split}'
            state.update(phase='evaluating',step=step,split=split);save()
            with (out/f'{dest.name}.log').open('w') as log:
                subprocess.run([sys.executable,'-u','-m','tools.cohort_eval.run',
                    '--checkpoint',str(root/f'{step:07d}.pt'),'--collection','top20/'+split,
                    '--out',str(dest),'--batch','32','--vae-batch','8'],
                    stdout=log,stderr=subprocess.STDOUT,check=True)
            result=json.loads((dest/'RESULT.json').read_text())
            results.append(dict(model='48.33M frozen',step=step,split=split,
                **{k:float(result[k]) for k in ('ssim_mean','lpips_mean','ink_iou_mean','frag_ratio','soft_ink_ratio')}))
            reference=next(r for r in old if r['model']=='20' and int(r['step'])==step and r['tier']=='20' and r['split']==split)
            results.append(dict(model='114M historical',step=step,split=split,
                **{k:float(reference[k]) for k in ('ssim_mean','lpips_mean','ink_iou_mean','frag_ratio','soft_ink_ratio')}))
            write_rows(out/'comparison.csv',results)
            state['completed']+=1;save()
    gaps=[]
    for model in ('48.33M frozen','114M historical'):
        for step in (25000,30000):
            tr=next(r for r in results if r['model']==model and r['step']==step and r['split']=='train_half')
            te=next(r for r in results if r['model']==model and r['step']==step and r['split']=='test')
            gaps.append(dict(model=model,step=step,ssim_seen_minus_test=tr['ssim_mean']-te['ssim_mean'],
                lpips_test_minus_seen=te['lpips_mean']-tr['lpips_mean'],
                iou_seen_minus_test=tr['ink_iou_mean']-te['ink_iou_mean']))
    write_rows(out/'gaps.csv',gaps)
    state.update(phase='completed');save()
    print(json.dumps(dict(results=results,gaps=gaps),indent=2),flush=True)


if __name__=='__main__':main()
