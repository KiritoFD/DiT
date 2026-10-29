"""Finish historical and S11 distribution evaluation after the one-shot S11 screen."""
from pathlib import Path
import subprocess
import sys
import time
from tools.eval_unpaired.protocol import read,save


def main():
    root=Path('assets/eval_unpaired_20261029');deadline=time.time()+3*3600
    while not (root/'S11_v1/complete.json').exists():
        if time.time()>deadline:raise TimeoutError('S11 evaluation pending; architecture queues remain paused')
        time.sleep(20)
    jobs=read(root/'history48/jobs.json')+read(root/'history_u100/jobs.json');save(root/'history48_all_jobs.json',jobs)
    base=[sys.executable,'-u','-m','tools.eval_unpaired.run','--machine','48','--reference-csv','assets/u100_p1p2_20261025/p2_train.csv']
    subprocess.run(base+['--jobs',str(root/'history48_all_jobs.json'),'--out',str(root/'history48_full')],check=True)
    for name in ['S11_v1','history48_full']:
        subprocess.run([sys.executable,'-u','-m','tools.eval_unpaired.panels.clean',
            '--manifest',str(root/name/'manifest.json'),'--out',str(root/(name+'_clean'))],check=True)
    save(root/'evaluation_finish.json',dict(completed=time.time(),new_generator_training=False))


if __name__=='__main__':main()
