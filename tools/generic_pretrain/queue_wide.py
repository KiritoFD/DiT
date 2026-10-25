"""Wait for the48M curriculum, then run the same data/budget with the114M backbone."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch
from src.unet.io import Config,build
from .contracts import sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',default='assets/generic48_kxl_top10_20261022')
    parser.add_argument('--out',default='assets/generic114_kxl_top10_20261023')
    args=parser.parse_args();source=Path(args.source);out=Path(args.out)
    out.mkdir(parents=True,exist_ok=False)
    pre=json.loads((source/'pretrain_50k_config.json').read_text())
    fine=json.loads((source/'top10_50k_config.json').read_text())
    for config,name in ((pre,'pretrain'),(fine,'top10')):
        config.update(channels=[192,384,576],num_res_blocks=4,decoder_res_blocks=None,
            initialize_from='',output=str(out/(name+'_runs')))
        Config(**config).validate()
        (out/(name+'_config.json')).write_text(json.dumps(config,indent=2))
    with torch.device('meta'):model=build(Config(**pre))
    parameters=sum(p.numel() for p in model.parameters());del model
    identity=dict(source=str(source),source_manifest_sha256=sha(source/'manifest.json'),
        pretrain_parameters=parameters,changes={'channels':[192,384,576],'num_res_blocks':4},
        exposure='Same B512 and50k/10k steps as48M; no silent batch fallback',
        configs={name:sha(out/(name+'_config.json')) for name in ('pretrain','top10')})
    (out/'queue_manifest.json').write_text(json.dumps(identity,indent=2))
    status=dict(phase='waiting_for_48m_complete',runner_pid=os.getpid(),started=time.strftime('%F %T'),
        upstream=str(source/'status.json'),pretrain_target=50000,finetune_target=10000,
        batch=pre['batch'],save_every=5000,parameters=parameters)
    def save():(out/'status.json').write_text(json.dumps(status,indent=2))
    def run(command,phase,logname):
        with (out/logname).open('w') as log:
            process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
            status.update(phase=phase,child_pid=process.pid);save()
            if process.wait():raise RuntimeError(phase+' failed; inspect '+str(out/logname))
    save()
    try:
        while True:
            try:upstream=json.loads((source/'status.json').read_text())
            except json.JSONDecodeError:time.sleep(5);continue
            if upstream['phase']=='failed':raise RuntimeError('Upstream48M failed; do not start GPU work')
            if upstream['phase']=='completed':break
            time.sleep(60)
        if sha(source/'manifest.json')!=identity['source_manifest_sha256']:
            raise ValueError('Upstream data manifest changed while waiting')
        manifest=json.loads((source/'manifest.json').read_text())
        for name,digest in manifest['files'].items():
            if sha(source/name)!=digest:raise ValueError('Source changed: '+name)
        provenance=json.loads((Path(pre['latent_shards_dir'])/'PROVENANCE.json').read_text())
        if provenance['encode_mode']!='mode' or provenance['scaling_factor']!=pre['vae_scaling_factor']:
            raise ValueError('VAE cache convention mismatch')
        base=[sys.executable,'-u','-m','src.unet.train']
        run(base+['--config',str(out/'pretrain_config.json'),'--benchmark','3'],
            'gpu_capacity_precheck','capacity.log')
        run(base+['--config',str(out/'pretrain_config.json')],'pretraining','pretrain.log')
        checkpoints=list(Path(pre['output']).glob('*/checkpoints/0050000.pt'))
        if len(checkpoints)!=1:raise ValueError('Missing/ambiguous114M50k checkpoint')
        fine['initialize_from']=str(checkpoints[0]);Config(**fine).validate()
        (out/'top10_config.json').write_text(json.dumps(fine,indent=2))
        status['source_checkpoint']=str(checkpoints[0]);save()
        run(base+['--config',str(out/'top10_config.json')],'top10_finetuning','top10.log')
        status.update(phase='completed',ended=time.strftime('%F %T'));save()
    except BaseException as error:
        status.update(phase='failed',error=repr(error));save();raise


if __name__=='__main__':main()
