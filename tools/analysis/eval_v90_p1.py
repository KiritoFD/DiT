"""P1 independent/style evaluation after all three arms; fail on missing evidence."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--work',required=True)
    a=p.parse_args();root=Path(a.root);work=Path(a.work);os.chdir(work)
    out=work/'evaluation';out.mkdir(exist_ok=False)
    sets={'v84A200k':root/'assets/runs/v84A_std_cond/20261013-072215-v84A-std-cond/eval_samples_ctrl/step0200000/eval200fix'}
    runs={}
    for arm in ('A','B','Control'):
        candidates=[p for p in (work/f'assets/results/v90_P1_{arm}').iterdir() if p.is_dir() and (p/'checkpoints/0005000.pt').is_file()]
        assert len(candidates)==1,(arm,candidates)
        run=candidates[0];runs[arm]=run
        summary=run/'eval_stdskel_summary.csv'
        rows=list(csv.DictReader(summary.open(encoding='utf-8-sig')))
        for step in (1000,3000,5000):
            assert any(int(r['step'])==step and r['set']=='eval200fix' for r in rows),(arm,step)
            sets[f'{arm}_{step}']=run/f'eval_samples_ctrl/step{step:07d}/eval200fix'
    for name,d in sets.items():
        assert all((d/f'g{i}.png').exists() for i in range(187)),(name,d)
    final_checkpoints={'v84A200k':root/'assets/runs/v84A_std_cond/20261013-072215-v84A-std-cond/checkpoints/0200000.pt',
                       **{arm:run/'checkpoints/0005000.pt' for arm,run in runs.items()}}
    for name,checkpoint in final_checkpoints.items():
        dest=out/f'multiseed_{name}'
        subprocess.run([sys.executable,'tools/analysis/sample_v90_multiseed.py',
            '--checkpoint',str(checkpoint),'--out',str(dest)],check=True)
        for seed in (0,1,2):sets[f'{name}_final_seed{seed}']=dest/f'seed{seed}'
    gen=[x for name,d in sets.items() for x in ('--gen',f'{name}={d}')]
    subprocess.run([sys.executable,'tools/eval_gen_with_disc.py','--device','cuda',*gen,
        '--out',str(out/'independent_writer.json'),'--per-image-out',str(out/'independent_writer_per_image.json')],check=True)
    args=[x for name,d in sets.items() for x in ('--set',f'{name}={d}')]
    subprocess.run([sys.executable,'tools/v85_compare.py',*args,'--out',str(out/'style')],check=True)
    # Identical fixed STD/noise, writer sweep. This is separate from paired GT metrics.
    cfsets={'v84A200k':root/'assets/writer_counterfactual_20261014'}
    for arm,run in runs.items():
        dest=out/f'writer_switch_{arm}'
        subprocess.run([sys.executable,'tools/analysis/writer_counterfactual.py',
            '--ckpt',str(run/'checkpoints/0005000.pt'),'--out',str(dest),'--batch','10'],check=True)
        cfsets[arm]=dest
    cfargs=[x for name,d in cfsets.items() for x in ('--gen',f'{name}={d}')]
    subprocess.run([sys.executable,'tools/eval_gen_with_disc.py','--device','cuda',
        '--csv',str(out/'writer_switch_A/eval_map.csv'),*cfargs,
        '--out',str(out/'writer_switch_independent.json'),
        '--per-image-out',str(out/'writer_switch_per_image.json')],check=True)
    (out/'completion.json').write_text(json.dumps({'status':'complete','P2_started':False,
        'sets':{k:str(v) for k,v in sets.items()},
        'limitations':['Native checkpoints use fixed seed0; final checkpoints additionally have matched seeds0,1,2.',
                      'Counterfactual sweep has8characters; not an187-character paired GT benchmark.']},indent=2))
    print('P1_EVALUATION_COMPLETE',flush=True)


if __name__=='__main__':main()
