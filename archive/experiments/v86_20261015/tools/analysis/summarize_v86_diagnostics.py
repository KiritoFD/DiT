"""Collect all matched 5k runs and evaluate the independent CE discriminator."""
import json
import re
import subprocess
import sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]

def main():
    out=ROOT/'assets/v86_diagnostics_20261014';out.mkdir(exist_ok=True)
    summary={}; gens=[]
    for arm in ['control','writer','brush']:
        cands=list((ROOT/'assets/runs'/f'v86_diag_{arm}').glob('*'))
        complete=[p for p in cands if (p/'checkpoints/0005000.pt.done').exists()]
        assert complete,('no complete 5k',arm)
        rp=max(complete,key=lambda p:p.name)
        cfg=json.loads((rp/'resolved_config.json').read_text())
        text=(ROOT/'logs'/f'v86_diag_{arm}_20261014.log').read_text(errors='replace')
        pp=[]
        for m in re.findall(r'dL_FM/dθ‖=([0-9.eE+-]+).*dL_writer/dθ‖=([0-9.eE+-]+).*dL_away/dθ‖=([0-9.eE+-]+)',text):
            fm,w,a=map(float,m)
            pp.append([100*cfg['w_critic_writer']*w/fm,100*cfg['w_critic_away']*a/fm])
        summary[arm]={'run':str(rp),'config':cfg,
          'final_eval':json.loads((rp/'checkpoints/eval_auto_5000.json').read_text()),
          'n_gradient_probes':len(pp),
          'weighted_gradient_percent_mean':np.mean(pp,axis=0).tolist() if pp else None,
          'weighted_gradient_percent_p10_p50_p90':np.percentile(pp,[10,50,90],axis=0).tolist() if pp else None}
        imgdir=rp/'eval_samples_ctrl/step0005000/eval200fix'
        assert all((imgdir/f'g{i}.png').exists() for i in range(187))
        gens+=['--gen',f'{arm}5k={imgdir}']
    baseline=ROOT/'assets/runs/v84A_std_cond/20261013-072215-v84A-std-cond/eval_samples_ctrl/step0200000/eval200fix'
    subprocess.run([sys.executable,str(ROOT/'tools/eval_gen_with_disc.py'),*gens,
        '--gen',f'v84A200k={baseline}','--out',str(out/'independent_disc.json')],cwd=ROOT,check=True)
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print('ALL_THREE_DIAGNOSTICS_COMPLETE',out)

if __name__=='__main__':main()
