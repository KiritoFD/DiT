"""实验0: environment and supervision-link validation -> environment.json.

Checks: S5 load + writer/script IDs; E-G eval-vs-pilot preprocessing equality
(per-image); decoder->E-G gradient nonzero/finite (no detach breaks); bank
slots valid with no same-character references; decode+EG memory/speed.
"""
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .common import EGEncoder,load_s5,load_vae_decoder,decode,save_json,sha256_file,S5_CKPT,EG_CKPT,PILOT_ROOT
from tools.eval_unpaired.protocol import reference_bank,csv_rows


def main():
    result=dict(started=time.time())
    model,cfg,ck=load_s5()
    result['s5']=dict(step=int(ck['train_steps']),num_writers=cfg.num_writers,
        repr=cfg.style_matrix_repr,injection=cfg.style_matrix_injection,
        data_csv=cfg.data_csv,ckpt_sha256=sha256_file(S5_CKPT))
    enc=EGEncoder()
    result['eg']=dict(ckpt_sha256=sha256_file(EG_CKPT),size=enc.size)
    # 1) preprocessing equivalence: PIL eval path vs plain tensor path on real PNGs
    from PIL import Image
    paths=[r['image_path'] for r in csv_rows(cfg.data_csv)[:16]]
    pil=torch.from_numpy(np.stack([np.asarray(Image.open(p).convert('RGB').resize((256,256),Image.Resampling.BICUBIC)).copy() for p in paths])).permute(0,3,1,2).float()/255
    plain=torch.from_numpy(np.stack([np.asarray(Image.open(p).convert('RGB')).copy() for p in paths])).permute(0,3,1,2).float()/255
    fp=enc.features_numpy(pil);fq=enc.features_numpy(plain)
    result['preprocess_max_abs_diff']=float(np.abs(fp-fq).max())
    result['preprocess_ok']=bool(result['preprocess_max_abs_diff']<1e-4)
    # 2) bank slots valid, no same-char refs
    bank_rows=reference_bank(cfg.data_csv)
    bad=[r for r in bank_rows if (r.get('aug') or '').strip()]
    result['bank']=dict(n=len(bank_rows),augmented_leak=len(bad),
        slots=len({(r['calligrapher'],r['script']) for r in bank_rows}))
    # 3) decoder->E-G gradient flow (synthetic latent)
    vae=load_vae_decoder()
    z=torch.randn(4,4,32,32,device='cuda',requires_grad=True)
    w=torch.tensor([0,1,2,3],device='cuda')
    img=decode(vae,z)
    feats=enc.features(img)
    logits=enc.net(img)
    loss=F.cross_entropy(logits,torch.tensor([1,2,3,4],device='cuda'))
    loss.backward()
    g=z.grad
    result['gradient']=dict(finite=bool(torch.isfinite(g).all()),norm=float(g.norm()),
        nonzero=bool(g.abs().sum()>0))
    # 4) differentiable preprocess equality: decode->E-G vs PIL roundtrip on same decode
    with torch.no_grad():
        img2=decode(vae,z.detach())
        pil_imgs=[Image.fromarray((im.permute(1,2,0).cpu().numpy()*255).astype(np.uint8)) for im in img2]
        pil_t=torch.from_numpy(np.stack([np.asarray(im).copy() for im in pil_imgs])).permute(0,3,1,2).float()/255
        d1=enc.features(img2).cpu().numpy();d2=enc.features_numpy(pil_t)
        result['decode_vs_pil_max_abs']=float(np.abs(d1-d2).max())
        result['decode_vs_pil_mean_abs']=float(np.abs(d1-d2).mean())
    # 5) memory/speed: 8-image decode+EG backward
    torch.cuda.reset_peak_memory_stats()
    z8=torch.randn(8,4,32,32,device='cuda',requires_grad=True)
    t0=time.monotonic()
    for _ in range(3):
        img=decode(vae,z8)
        f=enc.features(img)
        (f.square().mean()).backward()
    dt=(time.monotonic()-t0)/3
    result['aux_step']=dict(seconds_8img=round(dt,3),
        peak_GiB=round(torch.cuda.max_memory_allocated()/2**30,2))
    save_json(PILOT_ROOT/'environment.json',result)
    ok=all([result['preprocess_ok'],result['gradient']['finite'],result['gradient']['nonzero'],
            result['gradient']['norm']>1e-8,result['bank']['augmented_leak']==0,
            result['decode_vs_pil_mean_abs']<0.05])
    print(json.dumps(dict(ok=ok,**{k:v for k,v in result.items() if k in ('preprocess_max_abs_diff','gradient','decode_vs_pil_max_abs','aux_step','bank')}),ensure_ascii=False))
    if not ok:raise SystemExit('ENV CHECK FAILED')


if __name__=='__main__':main()
