"""Finish old S8 evaluation, deploy staged sources, probe, then run new S9."""
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import time


def main():
    old=Path('assets/style_matrix_20261027/matrix64/4090/execution/status.json')
    while True:
        status=json.loads(old.read_text())
        if status['state']=='stage_complete':break
        if status['state'] in ('failed','stopped'):raise RuntimeError(status)
        time.sleep(15)
    archive=Path('assets/style_matrix_20261028/source_generation2.tar')
    with tarfile.open(archive) as tar:
        members=tar.getmembers()
        root=Path.cwd().resolve()
        for item in members:
            if not item.isfile() or not (root/item.name).resolve().is_relative_to(root):raise ValueError(item.name)
        tar.extractall(root,members=members)
    subprocess.run([sys.executable,'-m','tools.style_matrix.prepare.generation2','--machine','4090'],check=True)
    root='assets/style_matrix_20261028/generation2/4090'
    subprocess.run([sys.executable,'-m','tools.style_matrix.probes.generation2','--spec',root+'/specs/S9.json',
                    '--micro','256','--steps','5','--out',root+'/probe.json'],check=True)
    subprocess.run([sys.executable,'-m','tools.style_matrix.runtime.serial','--plan',root+'/plan.json','--execute',
                    '--resident-pid','1967368'],check=True)


if __name__=='__main__':main()
