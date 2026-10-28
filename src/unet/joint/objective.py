"""J0--J4: sequential image backward, preserving the frozen latent interface."""
import copy
import hashlib
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from .losses import edge_loss


def frozen_fingerprint(vae):
    digest=hashlib.sha256()
    for name,value in vae.state_dict().items():
        if not name.startswith('decoder.'):
            digest.update(name.encode());digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class GTImages:
    def __init__(self,rows):
        self.rows={int(r['img_id']):r for r in rows}
    def __call__(self,ids):
        images=[]
        for value in ids:
            row=self.rows[int(value)]
            if (row.get('aug') or '').strip():raise ValueError('Joint GT needs an unaugmented cached target')
            im=Image.open(row['image_path']).convert('RGB')
            if im.size!=(256,256):raise ValueError('GT crop does not match the cached 256x256 contract')
            images.append(torch.from_numpy(np.asarray(im).copy()).permute(2,0,1).float()/127.5-1)
        return torch.stack(images).cuda(non_blocking=True)


class JointDecoderObjective:
    def __init__(self,arm,config,image_micro=16,image_batch=16,img_weight=.1,rec_weight=1.):
        from diffusers import AutoencoderKL
        import lpips
        if arm not in ('J0','J1','J2','J3','J4','G2J4','G2J5','G2J6','G2J7'):raise ValueError(arm)
        self.arm=arm;self.image=arm in ('J1','J3','J4');self.reconstruction=arm in ('J2','J3')
        # J4 is the decomposition control for J3: decoder trains on generated
        # latents through the image branch only, with no GT reconstruction anchor.
        self.train_decoder=arm in ('J3','J4')
        self.detach_image=arm in ('G2J4','G2J5','G2J6')
        if arm.startswith('G2'):
            self.image=True;self.reconstruction=False;self.train_decoder=True
        self.sf=config.vae_scaling_factor;self.shift=config.vae_shift_factor
        self.seed=config.seed;self.micro=image_micro;self.count=image_batch
        self.img_weight=float(img_weight);self.rec_weight=float(rec_weight)
        self.vae=AutoencoderKL.from_pretrained(config.vae_path,local_files_only=True).cuda().eval().requires_grad_(False)
        self.vae.decoder.requires_grad_(self.train_decoder)
        if arm in ('G2J5','G2J6','G2J7'):
            from .conditioned import ConditionedDecoder
            self.vae.decoder=ConditionedDecoder(self.vae.decoder,config.num_writers,config.num_scripts,
                                                spatial=arm in ('G2J6','G2J7')).cuda()
        if self.train_decoder:self.vae.decoder.train()
        self.perceptual=lpips.LPIPS(net='alex',verbose=False).cuda().eval().requires_grad_(False)
        self.decoder_ema=copy.deepcopy(self.vae.decoder).eval().requires_grad_(False) if self.train_decoder else None
        assert not any(getattr(m,'gradient_checkpointing',False) for m in self.vae.modules())
        assert all(n.startswith('decoder.') for n,p in self.vae.named_parameters() if p.requires_grad)
        self.frozen_sha=frozen_fingerprint(self.vae)
        self.pair_checked=False

    def decode(self,z,w=None,s=None,g=None):
        if self.arm in ('G2J5','G2J6','G2J7'):
            if w is None:raise ValueError('Conditional decoder requires explicit Writer, Script and STD')
            return self.vae.decoder(self.vae.post_quant_conv(z.float()/self.sf+self.shift),w,s,g)
        return self.vae.decode(z.float()/self.sf+self.shift).sample

    def image_loss(self,pred,target):
        l1=F.l1_loss(pred,target);perceptual=self.perceptual(pred,target).mean();edge=edge_loss(pred,target)
        return l1+.1*perceptual+.1*edge,dict(l1=float(l1.detach()),lpips=float(perceptual.detach()),edge=float(edge.detach()))

    def backward(self,model,z,g,conditions,global_step,targets=None,img_ids=None,reader=None):
        if self.arm=='J0':return dict(image_samples=0,reconstruction_samples=0)
        n=min(self.count,len(z));rng=np.random.default_rng(self.seed+global_step*100003)
        ids=torch.tensor(rng.choice(len(z),n,replace=False),device=z.device)
        if targets is not None:truth=targets[ids] if len(targets)==len(z) else targets
        else:
            if reader is None or img_ids is None:raise ValueError('Image supervision requires paired img_id GT')
            truth=reader([img_ids[i] for i in ids.cpu().tolist()])
            if not self.pair_checked:
                with torch.no_grad():
                    mean=(self.vae.encode(truth[:2]).latent_dist.mode()-self.shift)*self.sf
                    latent_error=float((mean-z[ids[:2]]).abs().mean())
                    reconstruction_error=float((self.decode(z[ids[:2]],conditions[0][ids[:2]],conditions[1][ids[:2]],g[ids[:2]])-truth[:2]).abs().mean())
                if latent_error>.15 or reconstruction_error>.10:
                    raise ValueError(f'GT/latent crop or VAE mismatch: latent={latent_error}, reconstruction={reconstruction_error}')
                self.pair_checked=True
        w,s=conditions;stats=dict(image_samples=n if self.image else 0,reconstruction_samples=n if self.reconstruction else 0)
        # The auxiliary branch must not change the NEXT FM batch's random draws.
        with torch.random.fork_rng(devices=[z.device.index or 0]):
            torch.manual_seed(self.seed+global_step*100003+7)
            tt=torch.rand(n,device=z.device)*.3+.2;eps=torch.randn_like(z[ids])
            for begin in range(0,n,self.micro):
                part=slice(begin,begin+self.micro);selected=ids[part];weight=len(selected)/n
                if self.image:
                    t=tt[part];zt=z[selected]*(1-t[:,None,None,None])+eps[part]*t[:,None,None,None]
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        velocity=model(zt,t*1000.,w[selected],g=g[selected],y_script=s[selected],condition_dropped=True)
                    zhat=zt-t[:,None,None,None]*velocity.float()
                    if self.detach_image:zhat=zhat.detach()
                    pred=self.decode(zhat,w[selected],s[selected],g[selected])
                    loss,detail=self.image_loss(pred,truth[part])
                    (self.img_weight*weight*loss).backward()
                    stats['image']=detail
                    del pred,loss,zhat,velocity
                if self.reconstruction:
                    pred=self.decode(z[selected].detach())
                    loss,detail=self.image_loss(pred,truth[part])
                    (self.rec_weight*weight*loss).backward()
                    stats['reconstruction']=detail
                    del pred,loss
        return stats

    def clip_decoder(self):
        if not self.train_decoder:return 0.
        return float(torch.nn.utils.clip_grad_norm_(self.vae.decoder.parameters(),1.,error_if_nonfinite=True))

    def update_ema(self):
        if self.decoder_ema is not None:
            from src.train.ema import update_ema
            update_ema(self.decoder_ema,self.vae.decoder,.999)

    def checkpoint(self):
        from src.train.ckpt import state_to_cpu
        assert frozen_fingerprint(self.vae)==self.frozen_sha, 'Frozen latent coordinates changed'
        return dict(joint_arm=self.arm,decoder=state_to_cpu(self.vae.decoder.state_dict()),
                    decoder_ema=state_to_cpu((self.decoder_ema or self.vae.decoder).state_dict()),
                    vae_frozen_sha256=self.frozen_sha,image_weight=self.img_weight,rec_weight=self.rec_weight,
                    image_batch=self.count,image_micro=self.micro,scaling=self.sf,shift=self.shift)

    def restore(self,ck):
        assert ck['vae_frozen_sha256']==self.frozen_sha
        assert ck['scaling']==self.sf and ck['shift']==self.shift
        self.vae.decoder.load_state_dict(ck['decoder'],strict=True)
        if self.decoder_ema is not None:self.decoder_ema.load_state_dict(ck['decoder_ema'],strict=True)
        self.img_weight=float(ck['image_weight']);self.rec_weight=float(ck['rec_weight'])
