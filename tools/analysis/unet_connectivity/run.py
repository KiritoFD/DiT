"""P0 diagnostics only. Does not stop training or launch an ablation training job."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from tools.analysis.unet_connectivity.components import scan


def main():
    p=argparse.ArgumentParser()
    sub=p.add_subparsers(dest='mode',required=True)
    a=sub.add_parser('thresholds');a.add_argument('--set',action='append',required=True)
    a.add_argument('--out',required=True);a.add_argument('--n',type=int,default=187)
    a.add_argument('--thresholds',type=float,nargs='+',default=[.3,.4,.5,.6,.7])
    a.add_argument('--min-areas',type=int,nargs='+',default=[0,2,4,8])
    a=sub.add_parser('sample');a.add_argument('--checkpoint',required=True);a.add_argument('--out',required=True)
    a.add_argument('--states',nargs='+',choices=['raw','ema'],default=['raw','ema'])
    a.add_argument('--cases',nargs='+',choices=['baseline','heun100','skip08','std08'],default=['baseline','heun100','skip08','std08'])
    a.add_argument('--device',default='cuda')
    a.add_argument('--writer-checkpoint',default='assets/style_disc_ce_holdout.pt')
    args=p.parse_args()
    if args.mode=='thresholds':
        result={name:scan(directory,args.n,args.thresholds,args.min_areas) for name,directory in (x.split('=',1) for x in args.set)}
        out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
        out.write_text(json.dumps(result,separators=(',',':')));return
    from tools.analysis.unet_connectivity.sample import sample_cases
    if not Path(args.writer_checkpoint).is_file():raise FileNotFoundError(args.writer_checkpoint)
    import torch
    torch.set_num_threads(4)
    cases=sample_cases(args.checkpoint,args.out,args.states,args.cases,args.device)
    command=[sys.executable,str(ROOT/'tools/eval_gen_with_disc.py'),'--ckpt',args.writer_checkpoint,
             '--device',args.device,'--root','.','--root','data','--out',str(Path(args.out)/'writer.json'),
             '--per-image-out',str(Path(args.out)/'writer_per_image.json')]
    for name,row in cases.items():command+=['--gen',name+'='+row['directory']]
    subprocess.run(command,check=True,env=dict(os.environ,OMP_NUM_THREADS='4'))
    results={name:scan(row['directory']) for name,row in cases.items()}
    (Path(args.out)/'thresholds.json').write_text(json.dumps(results,separators=(',',':')))


if __name__=='__main__':main()
