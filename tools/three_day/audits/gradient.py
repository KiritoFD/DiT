"""F2: matched 1% style/random/inverse updates on a native three-seed bank."""
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
import torch
from torch.nn import functional as F
from tools.three_day.contracts import save
from .teachers import Teacher,payloads,decode
from .metrics import interval,scores


def main(root):
    from tools.analysis.latent_style_gradient import prototypes,hard_reference_features,contrastive_per_image,bounded_step
    from src.eval.decoder import load_eval_vae
    root=Path(root);out=root/'F2';out.mkdir(exist_ok=True);pixel,data=payloads()
    metadata=json.loads((root/'bank/metadata.json').read_text())
    meta=[dict(writer=r['calligrapher'],script=r['script'],character=r['character'],seed=r['seed']) for r in metadata]
    train=np.asarray(data['splits']['train']);eligible=[]
    for i,r in enumerate(meta):
        pos=(data['writer'][train]==r['writer'])&(data['script'][train]==r['script'])&(data['char'][train]!=r['character'])
        neg=(data['writer'][train]!=r['writer'])&(data['script'][train]==r['script'])&(data['char'][train]!=r['character'])
        if pos.any() and neg.any() and (r['writer'],r['script']) in data['slots']:eligible.append(i)
    if len({meta[i]['character'] for i in eligible})<24:raise ValueError('Insufficient strict train-reference character coverage')
    meta=[meta[i] for i in eligible];z=torch.load(root/'bank/latents.pt',map_location='cpu',weights_only=True)[eligible].float().cuda()
    images=torch.load(root/'bank/images.pt',map_location='cpu',weights_only=True)
    gt=images['gt'][[i%48 for i in eligible]];target=torch.tensor([data['slots'].index((r['writer'],r['script'])) for r in meta],device='cuda')
    names=('DINO','pixel','D1','D2','D3','pixel_patch');models={name:Teacher(name,root,len(data['slots'])).cuda() for name in names}
    refs={}
    for name in names:
        f=np.load(root/'F1'/(name+'_features.npy'));proto,mask=prototypes(f,data,meta)
        targets_cpu=target.cpu();pos=proto[torch.arange(len(meta)),targets_cpu]
        neg=proto.clone();nmask=mask.clone();nmask[torch.arange(len(meta)),targets_cpu]=False
        refs[name]=[v.cuda() for v in (proto,mask,pos,neg,nmask)]
    identity=json.loads((root/'bank/identity.json').read_text());vae=load_eval_vae('cuda',identity['vae']).eval().requires_grad_(False)
    scale=identity['scale'];shift=identity['shift'];n=len(z)
    chars=[r['character'] for r in meta];rng=torch.Generator(device='cuda').manual_seed(90)
    random_direction=torch.randn(z.shape,generator=rng,device='cuda');report=dict(n=n,characters=len(set(chars)),seeds=[0,1,2],
        protocol='same 1% relative L2 latent radius; zero/random/inverse; train-only same-script writer prototypes excluding query character in every slot. NOT same-character negatives.',
        independent_caveat='Distinct frozen networks share data and can share shortcuts; not proof of complete genuine style.',candidates={})

    def embed(model,zz,rgb):
        result=[]
        for j in range(0,len(zz),16):
            with torch.autocast('cuda',dtype=torch.bfloat16):result.append(model(zz[j:j+16] if model.domain=='latent' else rgb[j:j+16].cuda()).float())
        return torch.cat(result)

    def assess(zz):
        with torch.no_grad():
            rgb=torch.cat([decode(vae,zz[j:j+8],scale,shift).cpu() for j in range(0,n,8)])
            judges={}
            for name,model in models.items():
                proto,mask,_,_,_=refs[name];s=scores(embed(model,zz,rgb),proto,mask,target)
                judges[name]={k:v.cpu().numpy() for k,v in s.items()}
            pred=rgb.mean(1)<.5;truth=gt.mean(1)<.5
            iou=(pred&truth).flatten(1).sum(1)/(pred|truth).flatten(1).sum(1).clamp_min(1)
            ink=(1-rgb).flatten(1).mean(1)
        return rgb,judges,iou.numpy(),ink.numpy()

    base_rgb,base_scores,base_iou,base_ink=assess(z)
    random_z=bounded_step(z,random_direction,.01);random_rgb,random_scores,random_iou,random_ink=assess(random_z)
    feature_report=json.loads((root/'style_teacher_audit.json').read_text())
    qualified=[name for name in names if all(feature_report['teachers'][name][s][k]['ci95'][0]>.5
        for s,k in [('test_char','matched_char'),('test_source','matched_source')])]
    import lpips
    perceptual=lpips.LPIPS(net='vgg',verbose=False).cuda().eval().requires_grad_(False)
    def lp(a,b):
        with torch.inference_mode():
            return torch.cat([perceptual(2*a[j:j+8].cuda()-1,2*b[j:j+8].cuda()-1).flatten().cpu() for j in range(0,len(a),8)]).numpy()
    base_lp=lp(base_rgb,gt)
    from scipy.ndimage import distance_transform_edt,maximum_filter
    def widths(rgb):
        answer=[]
        for x in rgb.mean(1).numpy():
            m=x<.5;d=distance_transform_edt(m);r=m&(d>=maximum_filter(d,size=5))
            answer.append(float(np.median(2*d[r]-1)) if r.any() else 0.)
        return np.array(answer)
    base_width=widths(base_rgb)
    for name,model in models.items():
        gradients=[];proto,mask,pos,neg,nmask=refs[name]
        for start in range(0,n,8):
            zz=z[start:start+8].detach().requires_grad_(True)
            x=zz if model.domain=='latent' else decode(vae,zz,scale,shift)
            with torch.autocast('cuda',dtype=torch.bfloat16):e=model(x)
            loss=contrastive_per_image(e.float(),pos[start:start+8],neg[start:start+8],nmask[start:start+8]).sum()
            gradients.append(torch.autograd.grad(loss,zz)[0].detach())
        gradient=torch.cat(gradients)
        if not torch.isfinite(gradient).all() or not gradient.flatten(1).norm(dim=1).gt(0).all():raise ValueError('Invalid teacher gradient '+name)
        updated=bounded_step(z,gradient,.01);inverse=bounded_step(z,-gradient,.01)
        rgb,after,iou,ink=assess(updated);_,back,_,_=assess(inverse)
        independent=[j for j in ('DINO','pixel','D2') if j!=name]
        transfer={j:interval((after[j]['margin']-base_scores[j]['margin'])-(random_scores[j]['margin']-base_scores[j]['margin']),chars) for j in independent}
        source=feature_report['teachers'][name]['test_source']['matched_source']
        valid_representation=name in qualified
        usable=[j for j in independent if j in qualified and not (name=='pixel_patch' and j=='pixel')]
        lp_gt=lp(rgb,gt);lp_change=lp(rgb,base_rgb);width_change=widths(rgb)-base_width
        passes=valid_representation and bool(usable) and all(transfer[j]['ci95'][0]>0 for j in usable) and float((iou-base_iou).mean())>=-.005 and float(np.abs(ink-base_ink).mean())<.01 and float((lp_gt-base_lp).mean())<.005 and float(np.abs(width_change).mean())<.5
        with torch.no_grad():
            rt=torch.cat([(vae.encode(2*rgb[j:j+8].cuda()-1).latent_dist.mean-shift)*scale for j in range(0,n,8)])
            rt_self=scores(embed(model,rt,rgb),proto,mask,target)['margin'].cpu().numpy()
        result=dict(pass_global=bool(passes and name!='pixel_patch'),pass_patch=bool(passes and name=='pixel_patch'),
            representation_gate=bool(valid_representation),transfer=transfer,qualified_independent_judges=usable,
            unqualified_judges_excluded_from_gate=[j for j in independent if j not in qualified],
            self_gain=interval(after[name]['margin']-base_scores[name]['margin'],chars),
            inverse_gain=interval(back[name]['margin']-base_scores[name]['margin'],chars),
            roundtrip_self_gain=interval(rt_self-base_scores[name]['margin'],chars),
            iou_delta=interval(iou-base_iou,chars),ink_abs_change=float(np.abs(ink-base_ink).mean()),
            lpips_change=float(lp_change.mean()),lpips_gt_delta=interval(lp_gt-base_lp,chars),width_abs_change=float(np.abs(width_change).mean()),
            patch_gate='Foreground 16x16 Pixel-D1 feature-map mean/std; separate local loss, not the global embedding.')
        per_image=[dict(meta[i],idx=i,iou_before=float(base_iou[i]),iou_after=float(iou[i]),
            lpips_gt_before=float(base_lp[i]),lpips_gt_after=float(lp_gt[i]),lpips_change=float(lp_change[i]),
            width_change=float(width_change[i]),ink_change=float(ink[i]-base_ink[i]),
            margins={j:dict(before=float(base_scores[j]['margin'][i]),after=float(after[j]['margin'][i]),
                random=float(random_scores[j]['margin'][i]),inverse=float(back[j]['margin'][i])) for j in names}) for i in range(n)]
        save(out/(name+'_per_image.json'),per_image)
        torch.save(dict(updated=updated.cpu(),eligible_indices=eligible),out/(name+'_latent_update.pt'))
        save(out/(name+'.json'),result);report['candidates'][name]=result
        canvas=Image.new('RGB',(5*128,24+12*128),'white');d=ImageDraw.Draw(canvas)
        for col,title in enumerate(['GT','Baseline','Style 1%','Random 1%','abs change x8']):d.text((col*128+2,5),title,fill='black')
        for i in range(min(12,n)):
            diff=((rgb[i]-base_rgb[i]).abs()*8).clamp(0,1)
            for j,im in enumerate([gt[i],base_rgb[i],rgb[i],random_rgb[i],diff]):
                a=(im.permute(1,2,0).clamp(0,1).numpy()*255).round().astype('uint8')
                canvas.paste(Image.fromarray(a).resize((128,128)),(j*128,24+i*128))
        canvas.save(out/(name+'_poster.png'));save(root/'gradient_transfer.json',report)
        print('F2_DONE',name,'pass_global',passes,flush=True)
    # Patch supervision is not allowed to inherit a global gate.
    save(root/'style_gates.json',dict(global_candidates=[k for k,v in report['candidates'].items() if v['pass_global']],
        patch_candidates=[k for k,v in report['candidates'].items() if v['pass_patch']],
        gate_caveat='At least one distinct qualified frozen evaluator must transfer beyond matched random; models still share data. Requires visual verification before generator training.',
        visual_review_required=True,content_status='see content_attribution.json'))


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();torch.set_num_threads(8);main(a.root)
