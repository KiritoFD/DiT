"""Serial encode -> generic pretrain -> strict model-only Writer adaptation."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from tools.cohort_eval.datasets import sha,read_rows


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',required=True)
    p.add_argument('--execute',action='store_true',help='Without this flag only print commands')
    a=p.parse_args();path=Path(a.plan);plan=json.loads(path.read_text());folder=path.parent
    data=Path(plan['data']);manifest=json.loads((data/'manifest.json').read_text())
    if sha(data/'manifest.json')!=plan['data_manifest_sha256']:raise ValueError('Plan data manifest changed')
    for name,digest in manifest['files'].items():
        if sha(data/name)!=digest:raise ValueError('Dataset changed: '+name)
    pre=json.loads((folder/'pretrain.json').read_text());fine=json.loads((folder/'finetune.json').read_text())
    if fine['initialize_from']:raise ValueError('Use a fresh generated plan, not a previously launched stage')
    commands=[
        [sys.executable,'tools/encode_aug_latents.py','--csv',plan['encode'][0],'--out',plan['encode'][1],
         '--vae',plan['encode'][2],'--img-root','.','--encode-mode','mode','--scaling-factor','0.18215'],
        [sys.executable,'-u','-m','src.unet.train','--config',str(folder/'pretrain.json')],
        [sys.executable,'-u','-m','src.unet.train','--config',str(folder/'finetune.json')]]
    if not a.execute:
        print(json.dumps(commands,indent=2));return
    if list(Path(pre['output']).glob('*/checkpoints/*.pt')):raise FileExistsError('Existing pretrain run; do not silently restart')
    status=dict(phase='encode',completed=0,total=3,plan_sha256=sha(path))
    def save():(folder/'status.json').write_text(json.dumps(status,indent=2))
    save()
    for i,(name,command) in enumerate(zip(('encode','pretrain','finetune'),commands)):
        if name=='pretrain':
            prov=json.loads((data/'latents/PROVENANCE.json').read_text())
            if (prov['n_rows']!=len(read_rows(data/'encode.csv')) or prov['encode_mode']!='mode' or
                prov['scaling_factor']!=.18215 or prov['shift_factor']!=0 or prov['vae_dir']!=pre['vae_path']):
                raise ValueError('Encoded target latent contract mismatch/incomplete rows')
        if name=='finetune':
            checkpoints=list(Path(pre['output']).glob(f'*/checkpoints/{plan["pretrain_steps"]:07d}.pt'))
            if len(checkpoints)!=1:raise ValueError('Missing/ambiguous final pretrain checkpoint')
            fine['initialize_from']=str(checkpoints[0])
            (folder/'finetune.json').write_text(json.dumps(fine,indent=2))
            status['pretrained_sha256']=sha(checkpoints[0])
        status['phase']=name;save()
        with (folder/f'{name}.log').open('w') as log:
            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
        status['completed']=i+1;save()
    status['phase']='completed';save()


if __name__=='__main__':main()
