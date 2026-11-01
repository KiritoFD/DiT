"""Writer swap compatibility entry: native FM sampler, no private ODE."""
import argparse
import subprocess
import sys


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True)
    p.add_argument('--val100',default='assets/u100_pretrain_20261024/validation_val100.csv')
    p.add_argument('--out',default='assets/p2_audit_20261025/writer_swap')
    p.add_argument('--scale',type=float,default=1.)
    a=p.parse_args()
    subprocess.run([sys.executable,'-u','-m','tools.p2_audit.generate','--checkpoint',a.checkpoint,
                    '--csv',a.val100,'--out',a.out,'--scale',str(a.scale),'--swap'],check=True)
    subprocess.run([sys.executable,'-u','-m','tools.p2_audit.score','--run',a.out],check=True)


if __name__=='__main__':main()
