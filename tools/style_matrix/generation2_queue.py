"""Complete S8 eval, deploy staged code, then execute the 64h S controller."""
import json
import os
from pathlib import Path
import subprocess
import signal
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
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            args=path.read_bytes().split(b'\0')
            if len(args)>1 and args[1]==b'assets/style_matrix_20261027/matrix64/4090/run_stage.sh':
                os.kill(int(path.parent.name),signal.SIGTERM)
        except (FileNotFoundError,PermissionError,ProcessLookupError):pass
    archive=Path('assets/style_matrix_20261028/source_generation2full.tar')
    with tarfile.open(archive) as tar:
        members=tar.getmembers()
        root=Path.cwd().resolve()
        for item in members:
            if not item.isfile() or not (root/item.name).resolve().is_relative_to(root):raise ValueError(item.name)
        tar.extractall(root,members=members)
    root='assets/style_matrix_20261028/generation2full/4090'
    os.environ['TORCHINDUCTOR_CACHE_DIR']='/root/.cache/torch/inductor'
    os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
    subprocess.run([sys.executable,'-m','tools.style_matrix.generation2.runner','--machine','4090',
                    '--root',root,'--resident-pid','2107710','--prepare-only'],check=True)
    subprocess.run([sys.executable,'-m','tools.style_matrix.probes.generation2','--spec',root+'/specs/S9.json',
                    '--micro','256','--steps','3','--pair','--out',root+'/preflight_s9_pair.json'],check=True)
    os.execv(sys.executable,[sys.executable,'-u','-m','tools.style_matrix.generation2.runner',
             '--machine','4090','--root',root,'--resident-pid','2107710'])


if __name__=='__main__':main()
