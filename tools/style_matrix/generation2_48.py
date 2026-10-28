"""Validate a conditioned checkpoint through unified eval, then run J4--J7."""
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    root=Path('assets/style_matrix_20261028/generation2/48')
    completion=root/'smoke/J4/completion.json'
    while not completion.exists():
        # The initial smoke is externally started; fail if it exits without completion.
        cmd=subprocess.check_output(['pgrep','-af','tools.style_matrix.runtime.train'],text=True)
        if '--stop 5' not in cmd:raise RuntimeError('J4 smoke exited without a completion record')
        time.sleep(10)
    for arm in ('J5',):
        run=root/'smoke'/arm
        subprocess.run([sys.executable,'-u','-m','tools.style_matrix.runtime.train','--spec',str(root/'specs'/f'{arm}.json'),
                        '--stop','5','--run',str(run)],check=True)
        spec=json.loads((root/'specs'/f'{arm}.json').read_text())
        subprocess.run([sys.executable,'-u','-m','tools.style_matrix.runtime.evaluate','--checkpoint',
            str(run/'checkpoints/0000005.pt'),'--out',str(run/'evaluation'),'--csv',spec['config']['eval_csv'],
            '--writer-space','raw'],check=True)
    subprocess.run([sys.executable,'-u','-m','tools.style_matrix.runtime.serial',
        '--plan',str(root/'plan.json'),'--execute'],check=True)


if __name__=='__main__':main()
