"""Cross-decoder diagnostic (plan section 7): same U-Net latent decoded by
original / J2 / J3 decoders, and each J U-Net paired with every decoder.

Strict attribution needs identical latents across decoders; running the same
U-Net with a fixed protocol and seed produces the same latents per (unet,seed),
so each (unet, decoder) cell shares latents within the unet. Outputs per-cell
metrics (SSIM/LPIPS/ink) via the standard io.evaluate protocol; E-G identity
AUC per cell is computed later from the saved PNGs.

Runs on the 48 machine after the 85k eval serial finishes (GPU).
"""
import argparse
import json
from pathlib import Path
from dataclasses import replace
import torch
from src.unet import io as unet_io
from tools.style_matrix.runtime.evaluate import paired_vae


ARMS={'J0':'J0','J2':'J2','J3':'J3'}
RUNS=Path('assets/style_matrix_20261027/matrix64/48/execution/runs')


def ckpt_path(arm):
    return RUNS/arm/'checkpoints'/'0005000.pt'


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',default='assets/eval_style_v2_cache/48/cross_decoder')
    p.add_argument('--steps',type=int,default=50)
    a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    decoders={}
    export_root=out/'paired_vae'
    for arm in ('J2','J3'):
        path=ckpt_path(arm)
        model,cfg,ck=unet_io.load(str(path),'cuda',use_ema=True)
        folder=paired_vae(str(path),ck,cfg,export_root/arm,use_ema=True)
        decoders[arm]=folder
        torch.cuda.empty_cache()
    summary={}
    for unet_arm in ARMS:
        path=ckpt_path(unet_arm)
        model,cfg,ck=unet_io.load(str(path),'cuda',use_ema=True)
        for dec_arm,vae_path in [('original',cfg.vae_path),('J2',decoders['J2']),('J3',decoders['J3'])]:
            tag=f'{unet_arm}_x_{dec_arm}'
            cell=out/tag
            if (cell/'result.json').is_file():
                summary[tag]=json.loads((cell/'result.json').read_text());continue
            cfg_v=replace(cfg,vae_path=vae_path,output=str(cell))
            with torch.no_grad():
                result=unet_io.evaluate(model,cfg_v,int(ck['train_steps']),str(cell),use_ema=True)
            metrics=result.get('eval200fix',{})
            if not metrics or 'ssim' not in metrics:
                mfile=cell/'eval_samples_ctrl'/f"step{int(ck['train_steps']):07d}"/'eval200fix'/'metrics.json'
                metrics=json.loads(mfile.read_text())
            summary[tag]={k:metrics.get(k) for k in ('ssim','lpips','ink_iou','n')}
            (cell/'result.json').write_text(json.dumps(summary[tag],indent=2))
            print('CELL',tag,json.dumps(summary[tag]),flush=True)
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    print('CROSS-DECODER DONE',flush=True)


if __name__=='__main__':main()
