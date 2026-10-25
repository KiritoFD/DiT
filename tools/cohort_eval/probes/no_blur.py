"""Inference-only STD blur ablation against existing 30k sigma=.65 outputs."""
import json
from pathlib import Path
import subprocess
import sys

from tools.cohort_eval.datasets import write_rows


def main():
    out=Path('assets/audits/extreme48m_no_blur_inference_20261022')
    out.mkdir(parents=True,exist_ok=False)
    checkpoint='assets/results/extreme48m_top20_x1/20261022-172221-Unet-flow/checkpoints/0030000.pt'
    state=dict(phase='evaluating',completed=0,total=2,checkpoint=checkpoint,std_sigma=0)
    results=[]
    for split in ('test','train_half'):
        state['split']=split;(out/'status.json').write_text(json.dumps(state,indent=2))
        target=out/split
        with (out/f'{split}.log').open('w') as log:
            subprocess.run([sys.executable,'-u','-m','tools.cohort_eval.run',
                '--checkpoint',checkpoint,'--collection','top20/'+split,'--out',str(target),
                '--std-sigma','0','--batch','32','--vae-batch','8'],
                stdout=log,stderr=subprocess.STDOUT,check=True)
        base=Path('assets/audits/extreme48m_seen_20261022')/f'step0030000_{split}'
        baseline=json.loads((base/'RESULT.json').read_text())
        result=json.loads((target/'RESULT.json').read_text())
        a=json.loads((base/'identity.json').read_text());b=json.loads((target/'identity.json').read_text())
        for key in ('checkpoint_sha256','effective_csv_sha256','state','steps','cfg','noise'):
            if a[key]!=b[key]:raise ValueError('Ablation identity mismatch: '+key)
        for sigma,r in ((.65,baseline),(0,result)):
            results.append(dict(split=split,sigma=sigma,n=int(r['n']),**{k:float(r[k]) for k in
                ('ssim_mean','lpips_mean','ink_iou_mean','frag_ratio','soft_ink_ratio','near_blank_rate')}))
        write_rows(out/'comparison.csv',results)
        state['completed']+=1
    state['phase']='completed';(out/'status.json').write_text(json.dumps(state,indent=2))
    print(json.dumps(results,indent=2),flush=True)


if __name__=='__main__':main()
