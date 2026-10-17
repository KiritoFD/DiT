"""Profile existing U-Net config only. Does not launch training or choose a budget."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace',required=True)
    parser.add_argument('--config',default='src/unet/configs/fm_48.json')
    parser.add_argument('--batches',type=int,nargs='+',default=[192,384,768])
    parser.add_argument('--benchmark-steps',type=int,default=12)
    args=parser.parse_args()
    root=Path(args.workspace).resolve();os.chdir(root)
    results=[];(root/'logs').mkdir(exist_ok=True)
    for batch in args.batches:
        output=root/'logs'/f'benchmark_b{batch}.log'
        command=[sys.executable,'-u','-m','src.unet.train','--config',args.config,
                 '--global-batch-size',str(batch),'--benchmark',str(args.benchmark_steps)]
        with output.open('w') as f:
            status=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,timeout=300)
        text=output.read_text()
        if status.returncode:
            if 'out of memory' not in text.lower():raise RuntimeError(text[-5000:])
            print('OOM',batch,flush=True);break
        record=json.loads(next(line[len('BENCHMARK '):] for line in text.splitlines()
                               if line.startswith('BENCHMARK ')))
        results.append(record);print('PROFILE',record,flush=True)
    (root/'batch_profiles.json').write_text(json.dumps(results,indent=2))
    print('Profiles saved. Training horizon, batch and launch require an explicit command.')


if __name__=='__main__':main()
