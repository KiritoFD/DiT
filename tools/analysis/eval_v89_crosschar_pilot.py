"""Independent evaluation of the two same-budget, ID-only generation arms."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--plan',required=True)
    a=p.parse_args();plan=Path(a.plan)
    sets={};csvs={}
    for name in ('control','crosschar'):
        cfg=json.loads((plan/f'{name}.json').read_text())
        matches=[x for x in Path(cfg['results_dir']).iterdir() if (x/'checkpoints/0010000.pt').exists()]
        if len(matches)!=1:
            raise ValueError(f'Ambiguous or incomplete {name} runs: {matches}')
        run=matches[0]
        for step in (5000,10000):
            key=f'{name}_{step}'
            directory=run/f'eval_samples_ctrl/step{step:07d}/eval200fix'
            if not all((directory/f'g{i}.png').exists() for i in range(187)):
                raise ValueError(f'Incomplete canonical evaluation: {directory}')
            sets[key]=str(directory);csvs[key]=str(run/'eval_stdskel_summary.csv')
    baseline=Path('assets/runs/v84A_std_cond/20261013-072215-v84A-std-cond')
    sets['v84A_200k']=str(baseline/'eval_samples_ctrl/step0200000/eval200fix')
    csvs['v84A_200k']=str(baseline/'eval_stdskel_summary.csv')
    out=plan/'evaluation';out.mkdir()
    args=[item for name,directory in sets.items() for item in ('--gen',f'{name}={directory}')]
    subprocess.run([sys.executable,'tools/eval_gen_with_disc.py','--device','cuda',*args,
                    '--out',str(out/'independent_writer.json')],check=True)
    args=[item for name,directory in sets.items() for item in ('--set',f'{name}={directory}')]
    joins=[item for name,path in csvs.items() for item in ('--csv-map',f'{name}={path}')]
    subprocess.run([sys.executable,'tools/v85_compare.py',*args,*joins,'--out',str(out/'style')],check=True)
    (out/'sets.json').write_text(json.dumps(sets,indent=2))
    print('V89_CROSSCHAR_EVALUATION_COMPLETE',flush=True)


if __name__=='__main__':
    main()
