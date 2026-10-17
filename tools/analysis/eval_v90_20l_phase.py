"""Finite phase report; records evidence but never promotes a run to the next phase."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser();p.add_argument('--results',required=True);p.add_argument('--step',required=True,type=int)
    p.add_argument('--cache',required=True);a=p.parse_args()
    runs=[d for d in Path(a.results).iterdir() if d.is_dir() and (d/f'checkpoints/{a.step:07d}.pt').is_file()]
    if len(runs)!=1:raise ValueError(f'Ambiguous or missing phase run: {runs}')
    run=runs[0];out=run/'phase_acceptance';out.mkdir(exist_ok=False)
    with (run/'eval_stdskel_summary.csv').open(encoding='utf-8-sig') as f:
        content=[r for r in csv.DictReader(f) if r['set']=='eval200fix' and int(r['step'])==a.step]
    if len(content)!=1 or not content[0].get('lpips_mean'):
        raise RuntimeError('Missing same-protocol content/LPIPS evaluation')
    ckpt=run/f'checkpoints/{a.step:07d}.pt'
    subprocess.run([sys.executable,'tools/analysis/probe_v90_std.py','--checkpoint',str(ckpt),
        '--cache',a.cache,'--out',str(out/'std_robustness')],check=True)
    generated=run/f'eval_samples_ctrl/step{a.step:07d}/eval200fix'
    base=Path('assets/runs/v84A_std_cond/20261013-072215-v84A-std-cond/eval_samples_ctrl/step0200000/eval200fix')
    subprocess.run([sys.executable,'tools/eval_gen_with_disc.py','--device','cuda',
        '--gen',f'v84A200k_reference={base}','--gen',f'v90_phase{a.step}={generated}',
        '--out',str(out/'independent_writer.json'),'--per-image-out',str(out/'independent_writer_per_image.json')],check=True)
    subprocess.run([sys.executable,'tools/v85_compare.py','--set',f'v84A200k_reference={base}',
        '--set',f'v90_phase{a.step}={generated}','--out',str(out/'style')],check=True)
    (out/'report.json').write_text(json.dumps({'step':a.step,'content':content[0],
        'next_phase_started':False,'decision':'Review this evidence before explicitly resuming the next phase.',
        'limitations':['v84A200k is a mature reference, not a matched-step depth ablation.',
                      'No trusted OCR accuracy claimed; style metrics are frozen independent evaluators.',
                      'Single fixed eval noise seed; multi-seed confirmation needed before a final quality claim.']},indent=2))


if __name__=='__main__':main()
