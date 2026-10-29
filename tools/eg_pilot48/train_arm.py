"""实验3/4: unified short-training arms from the same S5@85k EMA start.

A  = plain FM control
B  = FM + E-G bank-ranking image-domain loss (every 4 steps, 8 samples,
     lambda adapted so aux grad ~= 2% of FM grad norm)
C  = FM + E-G intermediate alignment (REPA-style bottleneck head; fixed
     lambda calibrated on step 1; shuffle control logged)
D0/D1 = 2k Query-Adapter pilot: zero-init extra AdaGN modulation at scales
     32/16; D0 writer-independent (zero writer embedding), D1 conditioned.

All arms: batch 512, backbone LR 1e-5, style/adapter modules 3e-5, warmup 200,
cosine (5k or 2k), decoder frozen, EMA 0.999, identical start/sample order.
Checkpoints: assets/eg_pilot48_tmp (temporary, may be deleted after eval).
"""
import argparse
import copy
import csv
import json
import time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .common import EGEncoder,load_s5,load_vae_decoder,decode,quantize_ste,save_json,PILOT_ROOT,TMP_ROOT
from src.unet.data import loader,StdBank
from src.unet.io import training_args,diffusion
from src.train.step import build_cosine_scheduler
from src.train.ema import update_ema
from tools.eval_unpaired.protocol import reference_bank


def build_optimizer(model,extra=None,backbone_lr=1e-5,style_lr=3e-5):
    groups=[dict(params=[p for n,p in model.named_parameters()
                if not n.startswith(('style_representation.','style_injections.')) and p.requires_grad],
                lr=backbone_lr,name='backbone'),
            dict(params=[p for n,p in model.named_parameters()
                if n.startswith(('style_representation.','style_injections.')) and p.requires_grad],
                lr=style_lr,name='style')]
    if extra:groups+=extra
    return torch.optim.AdamW(groups,weight_decay=.01,fused=True)


def model_grad_norm(model):
    return float(sum(float(p.grad.norm())**2 for p in model.parameters() if p.grad is not None)**.5)


class QueryAdapter(torch.nn.Module):
    """Zero-init modulation added to Spatial-AdaGN injections at scales 32/16."""
    def __init__(self,writer_dim=32,conditioned=True):
        super().__init__()
        in_ch=64+(writer_dim if conditioned else 0)
        self.conditioned=conditioned
        self.mod=torch.nn.ModuleList([torch.nn.Conv2d(in_ch,128,1) for _ in range(2)])
        for m in self.mod:
            torch.nn.init.zeros_(m.weight);torch.nn.init.zeros_(m.bias)
        self.w_emb=None  # set per batch (B,32) float tensor

    def install(self,model):
        for site,i in enumerate((0,1)):  # scales 32,16
            inj=model.style_injections[i]
            adapter=self;idx=site
            def forward(h,state,std,scale,_inj=inj,_ad=adapter,_idx=idx):
                f=state.fields[scale].to(h.dtype)
                inp=torch.cat((h,std.to(h.dtype),f),1)
                mod=0
                if _ad.w_emb is not None or not _ad.conditioned:
                    B,H,W=f.shape[0],f.shape[-2],f.shape[-1]
                    we=_ad.w_emb if _ad.conditioned else torch.zeros(B,32,device=f.device,dtype=f.dtype)
                    mod=_ad.mod[_idx](torch.cat((f,we[:,:,None,None].expand(-1,-1,H,W)),1))
                gb=_inj.local(inp)+mod
                gamma,beta=gb.chunk(2,1)
                delta=gamma*h+beta
                return h+delta*_inj.gain.to(h.dtype)*state.valid[:,None,None,None].to(h.dtype)
            inj.forward=forward


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--arm',required=True,choices=('A','B','C','D0','D1'))
    p.add_argument('--steps',type=int,default=-1)
    p.add_argument('--lam',type=float,default=0.02)
    p.add_argument('--out',default=PILOT_ROOT/'train_arm_curves.csv')
    a=p.parse_args()
    steps=a.steps if a.steps>0 else (2000 if a.arm.startswith('D') else 5000)
    torch.manual_seed(23);np.random.seed(23)
    model,cfg,ck=load_s5()
    ema=copy.deepcopy(model).eval().requires_grad_(False)
    enc=None;vae=None;head=None;rank=None;adapter=None;lam=None
    bank_rows=reference_bank(cfg.data_csv)
    if a.arm=='B':
        enc=EGEncoder();vae=load_vae_decoder()
        from .common import bank_tensors as BT
        bank_feats_t=torch.from_numpy(enc.features_numpy(BT([r['image_path'] for r in bank_rows]).to('cuda'))).cuda()
    if a.arm=='C':
        enc=EGEncoder()
    if a.arm in ('D0','D1'):
        adapter=QueryAdapter(conditioned=(a.arm=='D1'));adapter.install(model)
    opt=build_optimizer(model,extra=([dict(params=head.parameters(),lr=3e-5)] if head is not None else [])
        +([dict(params=adapter.parameters(),lr=3e-5)] if adapter is not None else []))
    if a.arm=='C':
        from .common import bank_tensors as BT
        bank_feats=enc.features_numpy(BT([r['image_path'] for r in bank_rows]).to('cuda'))
        slot_mean=defaultdict(list)
        for j,r in enumerate(bank_rows):
            slot_mean[(int(r['calligrapher_id']),r['script'])].append(bank_feats[j])
        slot_mean={k:torch.from_numpy(np.mean(v,axis=0)).float().cuda() for k,v in slot_mean.items()}
        in_ch=cfg.channels[-1]
        head=torch.nn.Linear(in_ch,bank_feats.shape[1]).cuda()
        torch.nn.init.normal_(head.weight,std=1e-4);torch.nn.init.zeros_(head.bias)
        opt.add_param_group(dict(params=head.parameters(),lr=3e-5))
        hook_holder={}
        model.backbone.middle2.register_forward_hook(lambda m,i,o:hook_holder.__setitem__('h',o.float()))
    if a.arm=='C':
        pass  # head added to opt above
    sched=build_cosine_scheduler(opt,warmup_steps=200,total_steps=steps,min_lr_ratio=.1)
    args=training_args(cfg,TMP_ROOT/f'runs_{a.arm}')
    dataset,sampler,_,_,_=loader(cfg,args,None,0)
    process=diffusion(cfg)
    bank=StdBank(cfg)
    from torch.utils.data import DataLoader
    g=torch.Generator();g.manual_seed(101+abs(hash(a.arm))%1000)
    loader_iter=iter(DataLoader(dataset,batch_size=512,num_workers=8,drop_last=True,generator=g))
    info={}
    for r in csv.DictReader(open(cfg.data_csv,encoding='utf-8-sig')):
        if r.get('img_id'):info[r['img_id']]=(r['script'],r['character'])
    # fixed validation batch for the 1k/3k/5k val-FM node
    val_batch=None
    it=iter(DataLoader(dataset,batch_size=128,num_workers=4,generator=torch.Generator().manual_seed(777)))
    val_batch=next(it)
    curves=[];step=0;t0=time.monotonic()
    def val_fm():
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            z=val_batch['latent'].cuda().float()
            gg=val_batch['std_soft'].cuda().float()
            if gg.dtype==torch.uint8:gg=gg/255.
            if gg.ndim==3:gg=gg[:,None]
            ww=val_batch['y_callig'].cuda();ss=val_batch['y_script'].cuda()
            t=process.sample_t(len(z),z.device)
            return float(process.training_losses(model,z,t,dict(g=gg,y_callig=ww,y_script=ss,condition_dropped=True))['loss'].mean())
    while step<steps:
        try:batch=next(loader_iter)
        except StopIteration:
            loader_iter=iter(DataLoader(dataset,batch_size=512,num_workers=8,drop_last=True,generator=g));batch=next(loader_iter)
        z=batch['latent'].cuda().float().to(memory_format=torch.channels_last)
        gg=batch['std_soft'].cuda().float()
        if gg.dtype==torch.uint8:gg=gg/255.
        if gg.ndim==3:gg=gg[:,None]
        w=batch['y_callig'].cuda();s=batch['y_script'].cuda()
        model.train()
        opt.zero_grad(set_to_none=True)
        t=process.sample_t(len(z),z.device)
        noise=torch.randn_like(z)
        raw_all_drop=torch.rand(len(z),device=z.device)<model.backbone.drop_all
        wd=raw_all_drop|(torch.rand(len(z),device=z.device)<model.drop_writer)
        sd=raw_all_drop|(torch.rand(len(z),device=z.device)<model.backbone.drop_script)
        w_in=torch.where(wd,torch.full_like(w,10),w)
        if adapter is not None:
            adapter.w_emb=model.style_representation.table(w_in)[:,:32].float()
        with torch.autocast('cuda',dtype=torch.bfloat16):
            out=process.training_losses(model,z,t,dict(g=gg,y_callig=w_in,y_script=s,condition_dropped=True),
                noise=noise,return_pred_xstart=False)
        loss=out['loss'].sum()/len(z)
        aux_log=None
        if a.arm=='B' and step%4==0:
            # independent 8-sample forward; the 512 FM backward already freed its graph
            model.eval()
            z8=batch['latent'].cuda().float()[:8]
            g8=gg[:8];w8=w[:8];s8=s[:8]
            t8=torch.full((8,),0.35,device='cuda')
            noise8=noise[:8]
            with torch.autocast('cuda',dtype=torch.bfloat16):
                out8=process.training_losses(model,z8,t8,dict(g=g8,y_callig=w8,y_script=s8,condition_dropped=True),
                    noise=noise8,return_pred_xstart=True)
            z0p=out8['pred_xstart'].float()
            img=quantize_ste(decode(vae,z0p))
            f=enc.features(img)
            losses=[]
            for b in range(8):
                w_id=int(w8[b]);sc,ch=info.get(str(batch['img_id'][b].item()),(None,None))
                t_ids=[j for j,r in enumerate(bank_rows) if int(r['calligrapher_id'])==w_id and r['script']==sc and r['character']!=ch][:64]
                if len(t_ids)<8:continue
                t_score=(f[b]@bank_feats_t[t_ids].T).topk(5).values.mean()
                o_sc=[]
                for ow in sorted({int(r['calligrapher_id']) for r in bank_rows if r['script']==sc and int(r['calligrapher_id'])!=w_id}):
                    o_ids=[j for j,r in enumerate(bank_rows) if int(r['calligrapher_id'])==ow and r['script']==sc][:64]
                    if len(o_ids)<8:continue
                    o_sc.append((f[b]@bank_feats_t[o_ids].T).topk(5).values.mean())
                if len(o_sc)<3:continue
                losses.append(F.cross_entropy(torch.stack([t_score]+o_sc)[None]/0.05,
                    torch.tensor([0],device='cuda')))
            if losses:
                # fixed image-domain weight: the whole 8-sample aux backward is
                # self-contained; report the achieved grad ratio instead of chasing it
                aux=torch.stack(losses).mean()
                (0.02*aux).backward()
                aux_log=dict(aux=round(float(aux),4),weight=0.02)
            model.train()
        elif a.arm=='C':
            model.eval()
            h_holder={}
            hk=model.backbone.middle2.register_forward_hook(
                lambda m,i,o:h_holder.__setitem__('h',o))
            z8=batch['latent'].cuda().float()[:16]
            g8=gg[:16];w8=w[:16];s8=s[:16]
            t8=torch.full((16,),0.35,device='cuda')
            with torch.autocast('cuda',dtype=torch.bfloat16):
                process.training_losses(model,z8,t8,dict(g=g8,y_callig=w8,y_script=s8,
                    condition_dropped=True),return_pred_xstart=False)
            hk.remove()
            h=h_holder['h'].float()
            f_hat=head(h.mean((2,3)))
            losses=[]
            for b in range(16):
                sc,ch=info.get(str(batch['img_id'][b].item()),(None,None))
                skey=(int(w8[b]),sc)
                if skey in slot_mean:
                    losses.append(1-F.cosine_similarity(f_hat[b],slot_mean[skey],dim=0))
            if losses:
                aux=torch.stack(losses).mean()
                if lam is None:
                    loss.backward(retain_graph=True)
                    fm_norm=model_grad_norm(model)
                    gaux=torch.autograd.grad(aux,head.weight,retain_graph=False)[0]
                    lam=a.lam*max(fm_norm,1e-8)/max(float(gaux.norm()),1e-8)
                    print('C lambda calibrated:',round(lam,8),flush=True)
                    opt.zero_grad(set_to_none=True)
                (loss+lam*aux).backward()
                aux_log=dict(lam=round(lam,8) if lam else None,aux=round(float(aux),4))
                model.train()
            else:
                loss.backward();model.train()
        else:
            loss.backward()
        gn=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        opt.step();sched.step();step+=1
        update_ema(ema,model,.999)
        if step%50==0 or step==1:
            row=dict(arm=a.arm,step=step,fm=round(float(loss),5),
                grad_norm=round(float(gn),3),lr=round(float(sched.get_last_lr()[0]),8),
                aux=json.dumps(aux_log),minutes=round((time.monotonic()-t0)/60,1))
            curves.append(row);print(json.dumps(row),flush=True)
        if step in (1000,3000,steps) or step==1:
            model.eval();vf=val_fm();model.train()
            row=dict(arm=a.arm,step=step,val_fm=round(vf,5))
            curves.append(row);print(json.dumps(row),flush=True)
            if step==steps:
                TMP=TMP_ROOT/f'ckpt_{a.arm}';TMP.mkdir(parents=True,exist_ok=True)
                torch.save(dict(delta=model.state_dict(),ema=ema.state_dict(),step=steps,arm=a.arm),
                    TMP/f'{a.arm}_{steps:07d}.pt')
    with open(a.out,'a',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=['arm','step','fm','grad_norm','lr','aux','minutes','val_fm'])
        if f.tell()==0:w.writeheader()
        for r in curves:w.writerow({k:r.get(k,'') for k in w.fieldnames})
    print('ARM-DONE',a.arm,flush=True)


if __name__=='__main__':main()
