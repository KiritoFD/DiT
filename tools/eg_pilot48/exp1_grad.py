"""实验1.2: gradient causal pilot (clean).

For ~100 balanced (writer,character) samples from the training loader:
ẑ0 = z_t - t*v_θ(z_t,t,g,w) at t∈{0.2,0.4,0.6}; optimize Δz under
{eg_ce, eg_rank, shuffled-writer CE}; measure ΔS_EG, ΔS_EG2 (independent),
LPIPS and Ink-IoU of the decoded change. No U-Net update anywhere.
"""
import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .common import EGEncoder,EG2Encoder,load_s5,load_vae_decoder,decode,quantize_ste,save_json,PILOT_ROOT
from tools.eval_unpaired.protocol import reference_bank


def build_samples(cfg,dataset,want_per_slot=12,cap=100):
    import csv as _csv
    rows=list(_csv.DictReader(open(cfg.data_csv,encoding='utf-8-sig')))
    info={r['img_id']:(r['script'],r['character']) for r in rows if r.get('img_id')}
    by_slot=defaultdict(list)
    for r in rows:by_slot[(int(r['calligrapher_id']),r['script'])].append(r)
    rng=random.Random(5)
    want={}
    for slot,rs in sorted(by_slot.items()):
        for r in rng.sample(rs,min(want_per_slot,len(rs))):
            want.setdefault(r['img_id'],None)
    want={k:0 for k in list(want)[:cap]}
    kept=[]
    from torch.utils.data import DataLoader
    loader_iter=iter(DataLoader(dataset,batch_size=64,num_workers=4))
    with torch.no_grad():
        for batch in loader_iter:
            ids=batch['img_id']
            for b in range(len(ids)):
                key=str(ids[b])
                if key in want and want[key]==0:
                    want[key]=1
                    kept.append({kk:batch[kk][b:b+1].cpu() for kk in batch})
            if all(v==1 for v in want.values()):break
    return kept,info


def margin(feat,bank_feats,bank_rows,tgt,script,char,k=64):
    sims=feat[0]@bank_feats.T
    tgt_ids=[j for j,r in enumerate(bank_rows) if int(r['calligrapher_id'])==tgt]
    oth_ids=[j for j,r in enumerate(bank_rows) if int(r['calligrapher_id'])!=tgt]
    t_sc=sims[tgt_ids[:k]].topk(5).values.mean()
    o_sc=sims[oth_ids[:256]].topk(5).values.mean()
    return float(t_sc-o_sc)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--samples',type=int,default=100)
    p.add_argument('--opt-steps',type=int,default=60)
    p.add_argument('--out',default=PILOT_ROOT/'eg_gradient_audit.csv')
    a=p.parse_args()
    torch.manual_seed(11);np.random.seed(11);random.seed(11)
    model,cfg,ck=load_s5()
    model.eval()
    for prm in model.parameters():prm.requires_grad_(False)
    vae=load_vae_decoder()
    enc=EGEncoder()
    eg2=None
    try:eg2=EG2Encoder()
    except Exception as e:print('EG2 unavailable:',e,flush=True)
    from src.unet.data import loader,StdBank
    from src.unet.io import training_args,diffusion
    args=training_args(cfg,PILOT_ROOT/'tmp_runs')
    dataset,sampler,_,_,_=loader(cfg,args,None,0)
    process=diffusion(cfg)
    bank=StdBank(cfg)
    kept,info=build_samples(cfg,dataset,cap=a.samples)
    print('kept samples',len(kept),flush=True)
    bank_rows=reference_bank(cfg.data_csv)
    bank_feats=enc.features_numpy(__import__('tools.eg_pilot48.common',fromlist=['bank_tensors']).bank_tensors([r['image_path'] for r in bank_rows]))
    eg2_bank=None
    if eg2 is not None:
        from .common import bank_tensors as BT
        with torch.no_grad():eg2_bank=eg2.features_numpy(BT([r['image_path'] for r in bank_rows]).to('cuda'))
    lpips_fn=None
    try:
        import lpips as lpips_mod
        lpips_fn=lpips_mod.LPIPS(net='alex',verbose=False).cuda().eval()
        for pp in lpips_fn.parameters():pp.requires_grad_(False)
    except Exception as e:print('lpips unavailable',e,flush=True)
    results=[]
    rng=random.Random(9)
    for t_val in (0.2,0.4,0.6):
        for loss_name in ('eg_ce','eg_rank','shuffled'):
            agg=defaultdict(list)
            for sample in kept:
                z_real=sample['latent'].cuda().float()
                g=sample['std_soft'].cuda().float()
                if g.dtype==torch.uint8:g=g/255.
                if g.ndim==3:g=g[:,None]
                w=int(sample['y_callig'])
                wt=torch.tensor([w],device='cuda')
                img_id=str(sample['img_id'][0]) if 'img_id' in sample else ''
                script,char=info.get(img_id,('',None))
                with torch.no_grad():
                    t=torch.full((1,),t_val,device='cuda')
                    v=model(z_real,t*1000.,wt,g=g,y_script=sample['y_script'].cuda())
                    z0=(z_real-t[:,None,None,None]*v.float()).detach()
                wrong_w=(w+rng.randint(1,9))%10
                delta=(torch.randn_like(z0)*1e-3).requires_grad_(True)
                optz=torch.optim.Adam([delta],lr=2e-3)
                for st in range(a.opt_steps):
                    img=quantize_ste(decode(vae,z0+delta))
                    f=enc.features(img)
                    if loss_name in ('eg_ce','shuffled'):
                        tgt=wrong_w if loss_name=='shuffled' else w
                        loss=F.cross_entropy(enc.net(img),torch.tensor([tgt],device='cuda'))
                    else:
                        sims=f[0]@bank_feats.T
                        tgt_ids=[j for j,r in enumerate(bank_rows) if int(r['calligrapher_id'])==w and r['script']==script and r['character']!=char]
                        if len(tgt_ids)<8:tgt_ids=[j for j,r in enumerate(bank_rows) if int(r['calligrapher_id'])==w]
                        other_w=sorted({int(r['calligrapher_id']) for r in bank_rows if r['script']==script and int(r['calligrapher_id'])!=w})
                        t_sc=sims[tgt_ids[:64]].topk(5).values.mean()
                        o_sc=[sims[[j for j,r in enumerate(bank_rows) if int(r['calligrapher_id'])==ow][:64]].topk(5).values.mean() for ow in other_w[:9]]
                        loss=F.cross_entropy(torch.stack([t_sc]+o_sc)[None]/0.05,torch.tensor([0],device='cuda'))
                    optz.zero_grad();loss.backward();optz.step()
                with torch.no_grad():
                    img0=quantize_ste(decode(vae,z0));img1=quantize_ste(decode(vae,z0+delta))
                    d_eg=margin(enc.features(img1),bank_feats,bank_rows,w,script,char)-margin(enc.features(img0),bank_feats,bank_rows,w,script,char)
                    d_eg2=0.
                    if eg2 is not None:
                        f0=F.normalize(eg2.net(img0.mean(1,keepdim=True)),dim=1)
                        f1=F.normalize(eg2.net(img1.mean(1,keepdim=True)),dim=1)
                        d_eg2=margin(f1,eg2_bank,bank_rows,w,script,char)-margin(f0,eg2_bank,bank_rows,w,script,char)
                    d_lp=float(lpips_fn(img1*2-1,img0*2-1).item()) if lpips_fn is not None else float(F.l1_loss(img1,img0).item())
                    g0=img0[0].mean(0).cpu().numpy()<0.5;g1=img1[0].mean(0).cpu().numpy()<0.5
                    iou=float((g0&g1).sum()/max((g0|g1).sum(),1))
                    agg['n'].append(1);agg['d_eg'].append(d_eg);agg['d_eg2'].append(d_eg2)
                    agg['d_lpips'].append(d_lp);agg['d_iou'].append(iou)
            row=dict(t=t_val,loss=loss_name,n=len(agg['n']),
                d_eg=round(float(np.mean(agg['d_eg'])),4),
                d_eg2=round(float(np.mean(agg['d_eg2'])),4),
                d_lpips=round(float(np.mean(agg['d_lpips'])),5),
                ink_iou_keep=round(float(np.mean(agg['d_iou'])),4),
                transfer_ratio=round(float(np.mean(agg['d_eg2']))/max(float(np.mean(agg['d_eg'])),1e-6),3) if np.mean(agg['d_eg'])>1e-4 else None)
            results.append(row);print(json.dumps(row),flush=True)
    with open(a.out,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(results[0].keys()));w.writeheader();w.writerows(results)
    save_json(PILOT_ROOT/'eg_gradient_verdict.json',dict(rows=results,
        verdict_note='eg↑+eg2↑+content-ok=接入训练; eg↑+eg2→=投机; eg↑+eg2↓=停止图像域监督'))


if __name__=='__main__':main()
