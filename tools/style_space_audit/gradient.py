"""Pilot frozen ResNet-style gradient on P2 latents; no generator weight update."""
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
import torch
from torch.nn import functional as F
from .features import StyleDisc
from tools.three_day.contracts import save
from tools.three_day.audits.teachers import Teacher,payloads,decode
from tools.three_day.audits.metrics import interval,scores


def main(root):
    from tools.analysis.latent_style_gradient import bounded_step,prototypes
    from src.eval.decoder import load_eval_vae
    root=Path(root);out=root/'gradient';out.mkdir(exist_ok=True)
    rows=json.loads((root/'rows.json').read_text());real=json.loads((root/'real_rows.json').read_text())
    f=np.load(root/'features/REAL_raw.npz')['embedding'];groups=defaultdict(list)
    for i,r in enumerate(real):
        if not r['source_heldout']:groups[r['writer'],r['script']].append(i)
    keys=sorted(groups);selected=[];counts=defaultdict(int);chars=set()
    for i,r in enumerate(rows):
        key=r['writer'],r['script']
        if key not in groups or counts[r['writer']]>=2 or r['char'] in chars:continue
        if not any(real[j]['char']!=r['char'] for j in groups[key]):continue
        selected.append(i);counts[r['writer']]+=1;chars.add(r['char'])
    if len(selected)<10:raise ValueError('Too few legal train-reference pilot queries')
    rr=[rows[i] for i in selected];zz=torch.from_numpy(np.load(root/'P2_CFG1_latents.npy')[selected]).cuda().float()
    proto=[];mask=[];targets=[]
    for r in rr:
        pp=[];mm=[]
        for key in keys:
            ids=[i for i in groups[key] if real[i]['char']!=r['char']]
            v=f[ids].mean(0) if ids else np.zeros(f.shape[1]);v/=max(np.linalg.norm(v),1e-8)
            pp.append(v);mm.append(bool(ids) and key[1]==r['script'])
        proto.append(pp);mask.append(mm);targets.append(keys.index((r['writer'],r['script'])))
    proto=torch.tensor(np.array(proto),device='cuda');mask=torch.tensor(mask,device='cuda');target=torch.tensor(targets,device='cuda')
    teacher=StyleDisc().cuda();vae=load_eval_vae('cuda','pretrained_models/sd-vae-ft-ema').eval().requires_grad_(False)
    gradient=[]
    for j in range(0,len(rr),4):
        z=zz[j:j+4].detach().requires_grad_(True);e,_=teacher(decode(vae,z))
        sim=torch.einsum('bd,bkd->bk',e,proto[j:j+4]);cp=sim.gather(1,target[j:j+4,None])
        valid=mask[j:j+4].clone();valid.scatter_(1,target[j:j+4,None],False)
        loss=(F.softplus((sim-cp+.1)/.1)*valid).sum()/valid.sum()
        gradient.append(torch.autograd.grad(loss,z)[0].detach())
    gradient=torch.cat(gradient);rng=torch.Generator(device='cuda').manual_seed(90)
    variants={'zero':zz,'style':bounded_step(zz,gradient,.01),'random':bounded_step(zz,torch.randn(zz.shape,device='cuda',generator=rng),.01),'inverse':bounded_step(zz,-gradient,.01)}
    pixel,data=payloads();metadata=[dict(writer=r['writer'],script=r['script'],character=r['char']) for r in rr]
    judge_root=Path('assets/three_day_20261026');independent={name:Teacher(name,judge_root,len(data['slots'])).cuda() for name in ('pixel','D2')}
    refs={}
    for name in independent:
        ff=np.load(judge_root/'F1'/(name+'_features.npy'));pp,mm=prototypes(ff,data,metadata)
        tt=torch.tensor([data['slots'].index((r['writer'],r['script'])) for r in rr],device='cuda')
        refs[name]=(pp.cuda(),mm.cuda(),tt)
    values={};images={}
    with torch.inference_mode():
        for name,z in variants.items():
            rgb=torch.cat([decode(vae,z[j:j+4]).cpu() for j in range(0,len(z),4)]);images[name]=rgb
            e=torch.cat([teacher(rgb[j:j+8].cuda())[0] for j in range(0,len(z),8)])
            values[name]={'self':scores(e,proto,mask,target)}
            for judge,model in independent.items():
                ee=[]
                for j in range(0,len(z),8):
                    with torch.autocast('cuda',dtype=torch.bfloat16):ee.append(model(z[j:j+8] if model.domain=='latent' else rgb[j:j+8].cuda()).float())
                values[name][judge]=scores(torch.cat(ee),*refs[judge])
            values[name]={judge:{k:v.cpu().numpy() for k,v in result.items()} for judge,result in values[name].items()}
    chars=[r['char'] for r in rr];summary={}
    for judge in ('self','pixel','D2'):
        summary[judge]=dict(style_minus_zero=interval(values['style'][judge]['margin']-values['zero'][judge]['margin'],chars),
            style_minus_random=interval(values['style'][judge]['margin']-values['random'][judge]['margin'],chars),
            inverse_minus_zero=interval(values['inverse'][judge]['margin']-values['zero'][judge]['margin'],chars),
            top1_before=float(values['zero'][judge]['correct'].mean()),top1_after=float(values['style'][judge]['correct'].mean()))
    import lpips
    lp=lpips.LPIPS(net='vgg',verbose=False).cuda().eval().requires_grad_(False);perceptual=[]
    with torch.inference_mode():
        for j in range(0,len(rr),4):perceptual.extend(lp(2*images['style'][j:j+4].cuda()-1,2*images['zero'][j:j+4].cuda()-1).flatten().cpu().tolist())
    mass={k:(1-v).flatten(1).mean(1).numpy() for k,v in images.items()}
    records=[dict(rr[i],idx=selected[i],lpips_change=float(perceptual[i]),ink_change=float(mass['style'][i]-mass['zero'][i]),
        margins={j:{name:float(values[name][j]['margin'][i]) for name in variants} for j in ('self','pixel','D2')}) for i in range(len(rr))]
    save(out/'results.json',dict(n=len(rr),seed=0,relative_latent_radius=.01,summary=summary,lpips_change=float(np.mean(perceptual)),
        ink_abs_change=float(np.abs(mass['style']-mass['zero']).mean()),per_image=records,
        caveat='Single-seed 20-or-fewer-character pilot. Teacher source generalization remains unqualified; no Style Loss training approval.'))
    assert all(p.grad is None for p in teacher.parameters())
    canvas=Image.new('RGB',(4*128,24+len(rr)*128),'white');d=ImageDraw.Draw(canvas)
    for j,k in enumerate(variants):d.text((j*128+2,4),k,fill='black')
    for i in range(len(rr)):
        for j,k in enumerate(variants):
            a=(images[k][i].clamp(0,1).permute(1,2,0).numpy()*255).round().astype('uint8')
            canvas.paste(Image.fromarray(a).resize((128,128)),(j*128,24+i*128))
    canvas.save(out/'poster.png');print('STYLE_DISC_GRADIENT_PILOT_DONE',summary,flush=True)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();torch.set_num_threads(8);main(a.root)
