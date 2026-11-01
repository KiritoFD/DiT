"""Strict decoder-only forks of the VAE actually recorded in a P2 checkpoint."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

from .objectives import reconstruction


def frozen_hash(vae):
    h = hashlib.sha256()
    for name, tensor in vae.state_dict().items():
        if not name.startswith('decoder.'):
            h.update(name.encode()); h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class Images(Dataset):
    def __init__(self, csv_path):
        self.rows = list(csv.DictReader(open(csv_path, encoding='utf-8-sig')))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        # Fail on corrupt/missing GT; never silently train on a white substitute.
        im = Image.open(self.rows[i]['image_path']).convert('RGB').resize((256,256), Image.Resampling.BICUBIC)
        return torch.from_numpy(np.asarray(im).copy()).permute(2,0,1).float() / 127.5 - 1


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--p2', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--arm', choices=['V0','V1'], required=True)
    p.add_argument('--steps', type=int, default=2500)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--accum', type=int, default=4)
    p.add_argument('--save-every', type=int, default=1250)
    p.add_argument('--lr', type=float, default=1e-5)
    a = p.parse_args(argv)
    from diffusers import AutoencoderKL
    import lpips
    from src.unet.io import checkpoint_config
    cfg = checkpoint_config(torch.load(a.p2,map_location='cpu',weights_only=False))
    out = Path(a.output); out.mkdir(parents=True,exist_ok=True)
    if (out/'identity.json').exists():
        raise FileExistsError('Independent fork output already initialized')
    torch.manual_seed(90); np.random.seed(90)
    vae = AutoencoderKL.from_pretrained(cfg.vae_path,local_files_only=True).cuda().eval()
    vae.requires_grad_(False); vae.decoder.requires_grad_(True)
    vae.decoder.train()
    params = list(vae.decoder.parameters())
    assert all(n.startswith('decoder.') for n,v in vae.named_parameters() if v.requires_grad)
    dataset = Images(cfg.data_csv)
    loader = DataLoader(dataset,batch_size=a.batch,shuffle=True,num_workers=4,
                        pin_memory=True,drop_last=True)
    fixed = torch.stack([dataset[i] for i in range(4)]).cuda()
    with torch.no_grad():
        encoded_before = vae.encode(fixed).latent_dist.parameters.detach().clone()
        post_input = vae.post_quant_conv(encoded_before[:,:4]).detach().clone()
    original_hash = frozen_hash(vae)
    identity = dict(arm=a.arm,source_p2=a.p2,vae_path=cfg.vae_path,
                    scaling=cfg.vae_scaling_factor,shift=cfg.vae_shift_factor,
                    frozen_sha256=original_hash,data_csv=cfg.data_csv,n=len(dataset),
                    batch=a.batch,accum=a.accum,steps=a.steps,lr=a.lr,seed=90,
                    posterior='sample, frozen encoder; identical seeded arms',
                    loss='L1 + LPIPS(VGG)' + (' + multiscale Sobel, decoder gradient cap .2' if a.arm=='V1' else ''))
    (out/'identity.json').write_text(json.dumps(identity,indent=2))
    print('DECODER_ONLY_CONTRACT',json.dumps(identity),flush=True)
    perceptual = lpips.LPIPS(net='vgg',verbose=False).cuda().eval().requires_grad_(False)
    opt = torch.optim.AdamW(params,lr=a.lr,weight_decay=0.)
    stream = iter(loader); start=time.monotonic()
    log = open(out/'metrics.jsonl','a',buffering=1)

    def verify():
        assert frozen_hash(vae) == original_hash, 'Frozen coordinates changed'
        with torch.no_grad():
            after = vae.encode(fixed).latent_dist.parameters
            post = vae.post_quant_conv(after[:,:4])
        assert torch.equal(after,encoded_before), 'Posterior mean/logvar changed'
        assert torch.equal(post,post_input), 'post_quant_conv changed'
        return dict(frozen_hash_equal=True,posterior_bitwise_equal=True,post_quant_bitwise_equal=True)

    for step in range(1,a.steps+1):
        opt.zero_grad(set_to_none=True)
        probe_stats={}
        for micro in range(a.accum):
            try: x=next(stream)
            except StopIteration: stream=iter(loader); x=next(stream)
            x=x.cuda(non_blocking=True)
            with torch.no_grad(): z=vae.encode(x).latent_dist.sample()
            pred=vae.decode(z).sample
            loss,stats=reconstruction(pred,x,perceptual,params,edge=a.arm=='V1',
                                       probe=micro==0 and (step==1 or step%250==0))
            if micro==0: probe_stats={k:v for k,v in stats.items() if 'grad' in k}
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite reconstruction')
            (loss/a.accum).backward()
        grad=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
        opt.step()
        if step==1 or step%25==0:
            stats.update(probe_stats)
            rec=dict(step=step,seconds=time.monotonic()-start,grad_norm=float(grad),
                      peak_gb=torch.cuda.max_memory_allocated()/1e9,**stats)
            print(json.dumps(rec),flush=True); log.write(json.dumps(rec)+'\n')
        if step==1 or step%a.save_every==0 or step==a.steps:
            checks=verify()
            (out/'contract_check.json').write_text(json.dumps(dict(step=step,**checks),indent=2))
        if step%a.save_every==0 or step==a.steps:
            dest=out/f'{step:07d}'; vae.save_pretrained(dest,safe_serialization=True)
            torch.save(dict(step=step,optimizer=opt.state_dict(),rng=torch.get_rng_state(),
                            cuda_rng=torch.cuda.get_rng_state()),dest/'training_state.pt')
            print('SAVED',dest,checks,flush=True)
    log.close()


if __name__=='__main__': main()
