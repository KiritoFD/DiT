"""One-shot evaluation after the already-running S11 finishes. Never resumes training."""
import argparse
import fcntl
from pathlib import Path
import subprocess
import sys
import time
from tools.eval_unpaired.protocol import discover,save


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='assets/eval_unpaired_20261029');p.add_argument('--wait-hours',type=float,default=3)
    a=p.parse_args();root=Path(a.root);deadline=time.time()+a.wait_hours*3600
    marker=Path('assets/style_matrix_20261028/generation2full/48/runs/S11/evaluations/0050000/shared_train_panel/complete.json')
    while not marker.exists():
        if time.time()>=deadline:raise TimeoutError('S11 did not finish; no new training launched')
        time.sleep(20)
    # complete.json precedes CUDA-context teardown. Wait for the serial owner's lock.
    from tools.style_matrix.runtime.serial import gpu_idle
    while True:
        if time.time()>=deadline:raise TimeoutError('GPU boundary not free')
        try:
            with open('/tmp/dit_gpu0.lock','a+') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);gpu_idle(0)
            break
        except (BlockingIOError,RuntimeError):time.sleep(2)
    jobs=[v for v in discover('48') if v['arm']=='S11'];save(root/'s11_jobs.json',jobs)
    if not jobs:raise ValueError('S11 marker exists without audited generation samples')
    subprocess.run([sys.executable,'-u','-m','tools.eval_unpaired.run','--machine','48',
        '--reference-csv','assets/u100_p1p2_20261025/p2_train.csv','--jobs',str(root/'s11_jobs.json'),
        '--out',str(root/'S11_v1')],check=True)
    print('S11_REEVALUATION_COMPLETE; architecture controllers remain paused',flush=True)


if __name__=='__main__':main()
