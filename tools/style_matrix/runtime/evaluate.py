"""Unified evaluation with the checkpoint's matching raw/EMA decoder."""
import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import torch
from src.unet.io import load,evaluate
from .common import sha,save


def paired_vae(checkpoint,ck,config,out,use_ema):
    if 'joint_arm' not in ck:return config.vae_path
    from diffusers import AutoencoderKL
    from src.unet.joint.objective import frozen_fingerprint
    vae=AutoencoderKL.from_pretrained(config.vae_path,local_files_only=True)
    if frozen_fingerprint(vae)!=ck['vae_frozen_sha256']:raise ValueError('Original frozen VAE changed')
    if (config.vae_scaling_factor,config.vae_shift_factor)!=(ck['scaling'],ck['shift']):
        raise ValueError('Checkpoint latent scale/shift changed')
    state='decoder_ema' if use_ema else 'decoder'
    vae.decoder.load_state_dict(ck[state],strict=True)
    # Unique immutable directory makes both the evaluator cache and VAE hash correct.
    identity=dict(checkpoint_sha256=sha(checkpoint),decoder_state=state,
                  frozen_vae_sha256=ck['vae_frozen_sha256'],scaling=ck['scaling'],shift=ck['shift'])
    folder=Path(out)/'paired_vae'/identity['checkpoint_sha256']/state
    manifest=folder/'identity.json'
    if manifest.exists():
        if json.loads(manifest.read_text())!=identity:raise ValueError('Paired VAE identity changed')
        files=json.loads((folder/'files.json').read_text())
        if any(sha(folder/k)!=v for k,v in files.items()):raise ValueError('Paired VAE files changed')
    else:
        if folder.exists():raise FileExistsError('Incomplete decoder export; preserve and select a new output')
        vae.save_pretrained(folder,safe_serialization=True)
        save(folder/'files.json',{p.name:sha(p) for p in folder.iterdir() if p.is_file()})
        save(manifest,identity)
    return str(folder)


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--out',required=True)
    p.add_argument('--csv',required=True);p.add_argument('--raw',action='store_true')
    p.add_argument('--writer-space',choices=['raw','top10-row'],required=True)
    a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    model,cfg,ck=load(a.checkpoint,'cuda',use_ema=not a.raw)
    path=paired_vae(a.checkpoint,ck,cfg,out,not a.raw)
    callig=cfg.callig_remap_json
    if a.writer_space=='top10-row':
        # Explicit protocol, never guess semantic IDs from their magnitude.
        callig=str(out/'top10_rows.json');save(callig,{str(i):i for i in range(10)})
    cfg=replace(cfg,eval_csv=a.csv,callig_remap_json=callig,vae_path=path,output=str(out))
    identity=dict(checkpoint_sha256=sha(a.checkpoint),csv_sha256=sha(a.csv),ema=not a.raw,
                  writer_space=a.writer_space,decoder_path=path,step=int(ck['train_steps']))
    if (out/'request.json').exists() and json.loads((out/'request.json').read_text())!=identity:
        raise ValueError('Evaluation directory identity changed')
    save(out/'request.json',identity)
    with torch.no_grad():result=evaluate(model,cfg,ck['train_steps'],str(out),use_ema=not a.raw)
    import csv
    expected_n=sum(1 for _ in csv.DictReader(open(a.csv,encoding='utf-8-sig')))
    metrics=result.get('eval200fix',result)
    if not metrics or 'ssim' not in metrics:
        path=out/'eval_samples_ctrl'/f"step{int(ck['train_steps']):07d}"/'eval200fix'/'metrics.json'
        metrics=json.loads(path.read_text())
    for key in ('ssim','lpips','ink_iou','n'):
        if key not in metrics or not math.isfinite(float(metrics[key])):raise ValueError('Incomplete metric: '+key)
    if int(metrics['n'])!=expected_n:raise ValueError('Evaluation dropped or changed samples')
    save(out/'result.json',metrics)
    save(out/'complete.json',identity)


if __name__=='__main__':main()
