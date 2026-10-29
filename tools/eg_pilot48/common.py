"""eg_pilot48 common: S5 loading, differentiable E-G, reference bank, checks.

Pilot rules (docs/1030/eg_supervision_pilot48/PROTOCOL.md):
- unified start: S5@85k EMA weights from assets/style_focus_85k_20261029/48_v3
- E-G is evaluation-only in the arms' primary metrics; any training use is
  gated by the gradient pilot (exp1)
- every artifact is SHA-traceable; no new checkpoint retention required
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F

S5_CKPT='assets/style_focus_85k_20261029/48_v3/runs/S5/checkpoints/0085000.pt'
EG_CKPT='assets/style_disc_ce_holdout.pt'
REAL_CSV='assets/u100_p1p2_20261025/p2_train.csv'
PILOT_ROOT=Path('docs/1030/eg_supervision_pilot48')
TMP_ROOT=Path('assets/eg_pilot48_tmp')


def sha256_file(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def save_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    temp.replace(path)


class EGEncoder:
    """Frozen E-G with a differentiable forward matching eval preprocessing.

    Eval path (tools/eval_unpaired/networks.batch): RGB[0,1] 256 -> bilinear160
    -> ImageNet norm -> net.emb. Here x may carry gradients; nothing detaches.
    """

    def __init__(self,device='cuda'):
        from tools.eval_unpaired.networks import Evaluators
        nets=Evaluators(device=device,iqa=False)
        self.net=nets.global_net;self.size=nets.size;self.device=device
        self.net.eval().requires_grad_(False)
        self.mean=torch.tensor([.485,.456,.406],device=device)[None,:,None,None]
        self.std=torch.tensor([.229,.224,.225],device=device)[None,:,None,None]

    def features(self,x01):
        """x01: RGB [0,1] tensor (B,3,256,256), may require grad. -> (B,512)."""
        scaled=F.interpolate(x01.to(self.device),(self.size,self.size),mode='bilinear',align_corners=False)
        normalized=(scaled-self.mean)/self.std
        return self.net.emb(normalized)

    @torch.no_grad()
    def features_numpy(self,x01):
        return self.features(x01).float().cpu().numpy()


def load_s5(device='cuda'):
    """S5@85k: (model, cfg, ck) with EMA weights loaded as raw (unified start)."""
    from src.unet import io as unet_io
    model,cfg,ck=unet_io.load(S5_CKPT,device,use_ema=True)
    return model,cfg,ck


def load_vae_decoder(device='cuda'):
    """Frozen original VAE decoder for differentiable decode."""
    from diffusers import AutoencoderKL
    vae=AutoencoderKL.from_pretrained('pretrained_models/sd-vae-ft-ema',local_files_only=True).to(device).eval()
    for p in vae.parameters():p.requires_grad_(False)
    return vae


def decode(vae,latents):
    """latents (B,4,32,32) -> RGB [0,1] (B,3,256,256), differentiable."""
    sf=0.18215
    return (vae.decode(latents.float()/sf).sample+1.)/2.


def real_bank_feats(encoder,bank_rows,cache='assets/eg_pilot48_tmp/bank_eg.npy'):
    """E-G features for the reference bank rows (reuse eval_unpaired cache)."""
    cache=Path(cache)
    if cache.is_file():
        return np.load(cache)
    feats=encoder.features_numpy(bank_tensors([r['image_path'] for r in bank_rows]))
    cache.parent.mkdir(parents=True,exist_ok=True)
    np.save(cache,feats)
    return feats


def bank_tensors(paths,batch=32):
    from PIL import Image
    outs=[]
    for start in range(0,len(paths),batch):
        chunk=[np.asarray(Image.open(p).convert('RGB').resize((256,256),Image.Resampling.BICUBIC)).copy() for p in paths[start:start+batch]]
        outs.append(torch.from_numpy(np.stack(chunk)).permute(0,3,1,2).float()/255)
    return torch.cat(outs)


def eg2_bank_feats(encoder2,bank_rows,cache='assets/eg_pilot48_tmp/bank_eg2.npy'):
    cache=Path(cache)
    if cache.is_file():return np.load(cache)
    feats=encoder2.features_numpy(bank_tensors([r['image_path'] for r in bank_rows]))
    cache.parent.mkdir(parents=True,exist_ok=True)
    np.save(cache,feats)
    return feats


def fm_loss(process,model,z,g,w,s,micro):
    """Standard FM loss on one microbatch slice set (matches runtime.step)."""
    losses=[]
    for begin in range(0,len(z),micro):
        part=slice(begin,begin+micro)
        t=process.sample_t(len(z[part]),z.device)
        noise=torch.randn_like(z[part])
        with torch.autocast('cuda',dtype=torch.bfloat16):
            loss=process.training_losses(model,z[part],t[part],
                dict(g=g[part],y_callig=w[part],y_script=s[part],condition_dropped=True),
                noise=noise[part])['loss'].sum()/len(z)
        losses.append(loss)
    return torch.stack(losses).sum()


class EG2Encoder:
    """Independent writer encoder (exp1.1). Same interface as EGEncoder:
    features(RGB[0,1] tensor) -> normalized embedding; input converted to gray."""

    def __init__(self,weights='assets/eg_pilot48_tmp/eg2.pt',device='cuda'):
        from .exp1_eg2 import build_model
        ck=torch.load(weights,map_location='cpu')
        self.net=build_model(device)
        self.net.load_state_dict(ck['net'])
        self.net.eval().requires_grad_(False)
        self.device=device

    def features(self,x01):
        gray=x01.mean(1,keepdim=True)
        return F.normalize(self.net(gray),dim=1)

    @torch.no_grad()
    def features_numpy(self,x01):
        return self.features(x01).float().cpu().numpy()


def quantize_ste(x01):
    """Straight-through uint8 quantization: aligns train-time decodes with the
    PNG domain used by the bank features and every evaluation."""
    return x01+(torch.round(x01*255.)/255.-x01).detach()
