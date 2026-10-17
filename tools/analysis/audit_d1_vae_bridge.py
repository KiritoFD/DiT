"""P0 frozen pixel-D1 / canonical VAE bridge. No model training or DiT writes."""
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from diffusers import AutoencoderKL
from src.model.writer_style_pixel import WriterStylePixel
from tools.train_writer_style_pixel import evaluate_features,sha
from tools.analysis.accept_writer_style_pixel import grouped


def pixel_input(rgb):
    # Differentiable approximation to PIL RGB->L / bicubic used in training.
    gray=(rgb*rgb.new_tensor([.299,.587,.114])[None,:,None,None]).sum(1,keepdim=True)
    return F.interpolate(gray,size=(128,128),mode='bicubic',align_corners=False,antialias=True).clamp(0,1)


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args()
    torch.set_num_threads(4);out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    base=Path('/root/Workspace/xy/DiT');data=torch.load('assets/data/data.pt',map_location='cpu',weights_only=False)
    ck=torch.load('assets/D1/best.pt',map_location='cpu',weights_only=False);assert ck['step']==9500
    model=WriterStylePixel(len(data['slots'])).cuda().eval().requires_grad_(False);model.load_state_dict(ck['model'],strict=True)
    canonical=base/'assets/runs/v84A_std_cond/20261013-072215-v84A-std-cond/checkpoints/0200000.pt'
    cp=torch.load(canonical,map_location='cpu',weights_only=False)
    args=cp.get('args',{});args=vars(args) if not isinstance(args,dict) else args
    print('V84_VAE_CONFIG',{k:v for k,v in args.items() if any(x in k for x in ('vae','scale','shift'))},flush=True)
    vaepath=Path('data/pretrained/sd-vae-ft-ema');vae=AutoencoderKL.from_pretrained(vaepath).cuda().eval().requires_grad_(False)
    # v84A/cache both use posterior.mode * .18215 with zero white shift.
    assert not args.get('white_latent',False)
    d=np.load('scratch/disc_lat_full.npz',allow_pickle=True);z=d['z']
    rows=[r for r in csv.DictReader(open('exp-std/csv/train.csv',encoding='utf-8-sig')) if not (r.get('aug') or '').strip()]
    replay=[]
    with torch.no_grad():
        for i in (0,len(z)//2,len(z)-1):
            image=torch.from_numpy(np.array(Image.open(rows[i]['image_path']).convert('RGB'))).permute(2,0,1)[None].cuda().float()/127.5-1
            encoded=vae.encode(image).latent_dist.mode()*.18215;cached=torch.from_numpy(z[i:i+1].copy()).cuda().float()
            relative=float((encoded-cached).square().mean().sqrt()/encoded.square().mean().sqrt());assert relative<.002,relative
            replay.append({'row':i,'relative_rms':relative})
    raw=np.load('assets/D1/clean_features.npz')['embedding'];tr=data['splits']['train']
    rawpred=np.empty(len(z),int);recon=np.empty_like(raw);pred=np.empty(len(z),int);gray=torch.from_numpy(data['gray'].numpy()).cuda()
    with torch.no_grad():
        for start in range(0,len(z),16):
            ids=np.arange(start,min(start+16,len(z)))
            latent=torch.from_numpy(z[ids].copy()).cuda().float()
            with torch.autocast('cuda',dtype=torch.bfloat16):
                rgb=(vae.decode(latent/.18215).sample.float().clamp(-1,1)+1)/2
                o=model(pixel_input(rgb));r=model(gray[ids].float()/255)
            recon[ids]=o['embedding'].float().cpu().numpy();pred[ids]=o['slot_logits'].argmax(1).cpu().numpy()
            rawpred[ids]=r['slot_logits'].argmax(1).cpu().numpy()
            if start%2048==0:print('ROUNDTRIP',start,'/',len(z),flush=True)
    np.savez_compressed(out/'features.npz',raw=raw,reconstructed=recon,raw_pred=rawpred,reconstructed_pred=pred)
    report={'d1_step':9500,'d1_sha256':sha('assets/D1/best.pt'),'vae_path':str(vaepath.resolve()),
        'vae_config_sha256':sha(vaepath/'config.json'),'latent_contract':'posterior.mode * .18215, zero shift; cache fp16',
        'encode_replay':replay,'raw_recon_cosine_quantiles':np.quantile((raw*recon).sum(1),[.05,.5,.95]).tolist(),
        'preprocessing':'PIL RGB/L128 for raw; differentiable RGB luma .299/.587/.114 + antialias bicubic128 for reconstruction',
        'protocols':{},'limitations':['Source directories are not verified artworks.','Reuse of cached posterior modes validated on 3 rows, not full re-encode.',
        'This only evaluates GT roundtrip, not generated distribution or latent gradient quality.']}
    for name,query_features in [('raw',raw),('recon_clean_bank',recon),('recon_recon_bank',recon)]:
        features=query_features.copy();pr=rawpred if name=='raw' else pred
        if name=='recon_clean_bank':features[tr]=raw[tr]
        report['protocols'][name]={}
        for split in ('test_char','test_source'):
            q=data['splits'][split];m=evaluate_features(features,pr,data,q,tr)
            m['matched']=grouped(features,data['matched_pairs'][split],data)
            m['per_slot_ce_recall']={str(i):float((pr[q[data['labels'][q]==i]]==i).mean()) for i in range(len(data['slots'])) if (data['labels'][q]==i).any()}
            report['protocols'][name][split]=m
        print('P0_RESULT',name,json.dumps(report['protocols'][name]),flush=True)
    rawmetric=report['protocols']['raw'];metric=report['protocols']['recon_clean_bank']
    passgate=(metric['test_char']['matched']['by_char']['accuracy']>=.57 and
              metric['test_source']['matched']['by_char']['accuracy']>=.55 and
              metric['test_char']['prototype_top1']>=rawmetric['test_char']['prototype_top1']-.05)
    report['p0_preliminary_pass']=bool(passgate)
    report['next_action']='Review class drift, then P1 if appropriate; no automatic DiT training.'
    (out/'p0.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('P0_COMPLETE',passgate,flush=True)


if __name__=='__main__':main()
