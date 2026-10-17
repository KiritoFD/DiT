"""Print A/B/C commands by default; --run opts into serial training on an idle GPU."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[3]


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--state',choices=['raw','ema'],default='raw')
    p.add_argument('--arms',nargs='+',choices=['A_control','B_skip32','C_decoder'],default=['A_control','B_skip32','C_decoder'])
    p.add_argument('--run',action='store_true',help='Actually run; default only prints commands')
    p.add_argument('--log-dir',default='assets/ablation_logs/unet_connectivity')
    args=p.parse_args()
    commands=[]
    for arm in args.arms:
        config=ROOT/f'src/unet/configs/connectivity/{arm}.json'
        cmd=[sys.executable,'-u','-m','src.unet.train','--config',str(config),
             '--warm-start',str(Path(args.checkpoint).resolve()),'--warm-start-state',args.state]
        commands.append((arm,cmd))
        print(json.dumps(cmd),flush=True)
    if not args.run:return
    os.chdir(ROOT)
    out=Path(args.log_dir);out.mkdir(parents=True,exist_ok=True)
    for arm,cmd in commands:
        busy=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
        if busy:raise RuntimeError('GPU already has a compute process; refusing to stop/share it: '+busy)
        with (out/f'{arm}.log').open('x') as log:
            subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)


if __name__=='__main__':main()
