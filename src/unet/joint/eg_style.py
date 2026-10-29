"""P3-B: E-G style ranking objective over decoded clean-latent estimates.

Frozen components: VAE (encoder+decoder params), E-G ResNet18. Gradients FLOW
through the frozen decoder and E-G back to the U-Net — the decoder is never
updated. Every `every` steps, a small image batch is decoded from mid-noise
clean-latent estimates and scored against a frozen reference bank of real
E-G embeddings (same-script, per-character exclusion at build time).

The first active step auto-calibrates the loss weight so the E-G branch
gradient norm equals `target_ratio` of the FM gradient norm measured on the
same batch; the calibration is recorded and reused unchanged afterwards.
"""
import hashlib
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F


def eg_fingerprint(net):
    digest=hashlib.sha256()
    for name,value in net.state_dict().items():
        digest.update(name.encode());digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class EGStyleObjective:
    def __init__(self,cfg,refs_npz,every=4,image_micro=4,target_ratio=0.02,
                 t_low=0.2,t_high=0.5,temperature=0.07,seed=90):
        from tools.eval_gen_with_disc import load_evaluator_model
        from diffusers import AutoencoderKL
        self.device='cuda'
        self.net,self.kind,_=load_evaluator_model('assets/style_disc_ce_holdout.pt',self.device)
        if self.kind!='resnet18':raise ValueError('Expected audited global ResNet checkpoint')
        self.net.eval().requires_grad_(False)
        self.eg_sha=eg_fingerprint(self.net)
        self.vae=AutoencoderKL.from_pretrained(cfg.vae_path,local_files_only=True).to(self.device).eval().requires_grad_(False)
        self.sf=cfg.vae_scaling_factor;self.shift=cfg.vae_shift_factor
        refs=np.load(refs_npz,allow_pickle=True)
        self.refs={int(k):{'embs':torch.from_numpy(np.asarray(v['embs'],dtype=np.float32)).to(self.device),
                           'chars':list(v['chars'])} for k,v in refs.item().items()}
        self.every=every;self.micro=image_micro;self.target_ratio=target_ratio
        self.t_low=t_low;self.t_high=t_high;self.temperature=temperature;self.seed=seed
        self.weight=None;self.calibration=None;self.pair_checked=False

    def decode(self,z):
        return self.vae.decode(z.float()/self.sf+self.shift).sample

    def eg_embed(self,images):
        x=F.interpolate(images,size=(160,),mode='bilinear',align_corners=False)
        x=(x-EG_MEAN.to(x.device)[None,:,None,None])/EG_STD.to(x.device)[None,:,None,None]
        pooled=self.net.emb(x)
        return pooled/pooled.norm(dim=-1,keepdim=True).clamp_min(1e-6)

    def scores(self,emb,slot):
        ref=self.refs[slot]['embs']
        sims=emb@ref.T
        k=min(5,sims.shape[1])
        return sims.topk(k,dim=1).values.mean(1)/self.temperature

    def backward(self,model,z,g,conditions,global_step,targets=None,img_ids=None,reader=None):
        if global_step%self.every!=0:
            return dict(eg_samples=0,eg_skipped=True)
        w,s=conditions
        cand=[i for i in range(len(z)) if f"{int(w[i])}_{int(s[i])}" in self.refs]
        if not cand:
            return dict(eg_samples=0,eg_no_target=True)
        n=min(self.micro,len(cand))
        rng=np.random.default_rng(self.seed+global_step*7919)
        idx=torch.tensor([cand[j] for j in rng.choice(len(cand),n,replace=False)],device=z.device)
        wt=w[idx]
        t=(torch.rand(n,device=z.device)*(self.t_high-self.t_low)+self.t_low)
        eps=torch.randn(len(idx),*z.shape[1:],device=z.device)
        zt=z[idx]*(1-t[:,None,None,None])+eps*t[:,None,None,None]
        with torch.autocast('cuda',dtype=torch.bfloat16):
            velocity=model(zt,t*1000.,w[idx],g=g[idx],y_script=s[idx],condition_dropped=True)
        zhat=zt-t[:,None,None,None]*velocity.float()
        images=self.decode(zhat)
        emb=self.eg_embed(images)
        slots=[f"{int(wi)}_{int(si)}" for wi,si in zip(wt,s[idx])]
        logits=torch.stack([self.scores(emb,sl) for sl in slots])
        ref_ids=sorted(self.refs)
        target=torch.tensor([ref_ids.index(sl) for sl in slots],device=z.device)
        loss=F.cross_entropy(logits,target)
        stats=dict(eg_samples=n,eg_loss=float(loss.detach()),
                   eg_weight=self.weight)
        params=[p for p in model.parameters() if p.requires_grad]
        if self.weight is None:
            # FM gradients already sit in p.grad (train.py backwarded the FM loss);
            # snapshot them, add the unit-weight E-G gradient, rescale exactly.
            snap={id(p):p.grad.detach().clone() for p in params if p.grad is not None}
            fm_norm=float(torch.sqrt(sum((g**2).sum() for g in snap.values())))
            loss.backward()
            eg_sq=0.;delta={}
            for p in params:
                g=p.grad if p.grad is not None else torch.zeros_like(p)
                d=g-snap.get(id(p),torch.zeros_like(p))
                delta[id(p)]=d;eg_sq+=float((d**2).sum())
            eg_norm=float(np.sqrt(eg_sq))
            self.weight=self.target_ratio*fm_norm/max(eg_norm,1e-12)
            with torch.no_grad():
                for p in params:
                    d=delta.get(id(p))
                    p.grad=(snap.get(id(p),torch.zeros_like(p))+self.weight*d) if d is not None else snap.get(id(p))
            self.calibration=dict(fm_norm=fm_norm,eg_norm_unit=eg_norm,
                weight=self.weight,global_step=global_step)
            stats['eg_calibrated']=True
            return stats
        (self.weight*loss).backward()
        return stats

    def clip_decoder(self):
        return 0.

    def update_ema(self):
        return None

    def checkpoint(self):
        assert eg_fingerprint(self.net)==self.eg_sha
        return dict(joint_arm='EG',eg_weight=self.weight,eg_calibration=self.calibration,
                    eg_sha256=self.eg_sha,every=self.every,image_micro=self.micro,
                    target_ratio=self.target_ratio)

    def restore(self,ck):
        if ck.get('joint_arm')!='EG':raise ValueError('Checkpoint is not an EG-style run')
        self.weight=ck.get('eg_weight');self.calibration=ck.get('eg_calibration')


EG_MEAN=torch.tensor([0.485,0.456,0.406])
EG_STD=torch.tensor([0.229,0.224,0.225])
