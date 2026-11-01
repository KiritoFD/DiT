"""Score completed PNGs, independently of sampling and its GPU allocations."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import numpy as np
from PIL import Image
from scipy.stats import wasserstein_distance
import torch
from .common import save,paired_ci


def match_ink(ink,target):
    """Match mass by fractional grayscale morphology, including saturated ink."""
    from scipy.ndimage import minimum_filter,maximum_filter
    operation=minimum_filter if ink.mean()>target else maximum_filter
    previous=ink
    for radius in range(1,65):
        current=operation(ink,size=2*radius+1,mode='constant',cval=0)
        a,b=float(previous.mean()),float(current.mean())
        if min(a,b)<=target<=max(a,b):
            fraction=(target-a)/(b-a) if b!=a else 0.
            return previous+fraction*(current-previous)
        previous=current
    raise ValueError('Ink target unreachable within diagnostic morphology range')


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True)
    p.add_argument('--baseline');a=p.parse_args();torch.set_num_threads(8)
    root=Path(a.run);rows=json.loads((root/'samples.json').read_text());n=len(rows)
    swap=json.loads((root/'generation.json').read_text())['swap']
    imgs=[Image.open(root/'samples'/f'g{i}.png').convert('RGB') for i in range(n)]
    from tools.u100_pretrain.eval_round1 import DinoJudge,ridge_widths
    judge=DinoJudge(None)
    probe=torch.load('assets/p2_audit_20261025/writer_probe_fixed.pt',map_location='cpu',weights_only=False)
    if probe.get('input_contract')!='vae_gray_float_0_1_no_second_div255':raise ValueError('Unverified probe preprocessing')
    if probe.get('classes')!=list(range(10)):raise ValueError('Probe class order does not match Top10 rows')
    targets=np.array([int(r['calligrapher_id']) for r in rows])
    def classify(images):
        gray=np.stack([np.asarray(im.convert('L'))[None] for im in images])
        features=judge.features([gray[i:i+32] for i in range(0,len(gray),32)])
        logits=features@np.asarray(probe['coef']).T+np.asarray(probe['intercept'])
        prob=np.exp(logits-logits.max(1,keepdims=True));prob/=prob.sum(1,keepdims=True)
        margin=prob[np.arange(n),targets]-np.where(np.eye(10)[targets].astype(bool),-np.inf,prob).max(1)
        return (prob.argmax(1)==targets),margin
    correct,margin=classify(imgs)
    records=[dict(idx=i,char=r['character'],writer=int(targets[i]),correct=bool(correct[i]),margin=float(margin[i])) for i,r in enumerate(rows)]
    summary=dict(n=n,writer_micro=float(correct.mean()),writer_macro=float(np.mean([correct[targets==w].mean() for w in np.unique(targets)])),
                 margin=float(margin.mean()),judge_caveat='Frozen DINO logistic probe: image-level holdout, not strict character/source holdout; not a sufficient teacher gate')
    if not swap:
        gts=[Image.open(root/'samples'/f'gt{i}.png').convert('RGB') for i in range(n)]
        gt_correct,gt_margin=classify(gts)
        summary.update(gt_writer_micro=float(gt_correct.mean()),gt_writer_margin=float(gt_margin.mean()))
        if a.baseline:
            # Diagnostic only: match total soft ink to the paired CFG=1 output.
            # This changes stroke geometry; it is not a pure style intervention.
            matched=[];residual=[]
            for i,im in enumerate(imgs):
                base=np.asarray(Image.open(Path(a.baseline)/'samples'/f'g{i}.png').convert('L'),dtype=np.float32)/255
                ink=1-np.asarray(im.convert('L'),dtype=np.float32)/255
                target=float((1-base).mean());value=match_ink(ink,target)
                quantized=np.rint((1-value)*255).astype('uint8')
                residual.append(float((1-quantized.astype(np.float32)/255).mean()-target))
                matched.append(Image.fromarray(quantized).convert('RGB'))
            if np.max(np.abs(residual))>1/255:raise ValueError('Ink matching did not reach its target')
            mc,mm=classify(matched)
            base_records=json.loads((Path(a.baseline)/'scores.json').read_text())['per_image']
            summary.update(ink_matched_writer_micro=float(mc.mean()),ink_matched_margin=float(mm.mean()),
                ink_matched_margin_delta=paired_ci(mm-np.array([r['margin'] for r in base_records]),[r['character'] for r in rows]),
                ink_match_max_mass_residual=float(np.max(np.abs(residual))),
                ink_match_caveat='Fractional grayscale erosion/dilation matching soft ink; changes geometry and is not proof of genuine style')
        if not a.baseline:
            from src.eval.decoder import load_eval_vae
            vae=load_eval_vae('cuda','pretrained_models/sd-vae-ft-ema');recons=[]
            with torch.inference_mode():
                for start in range(0,n,8):
                    x=torch.from_numpy(np.stack([np.asarray(im).copy() for im in gts[start:start+8]])).permute(0,3,1,2).cuda().float()/127.5-1
                    y=vae.decode(vae.encode(x).latent_dist.mode()).sample.clamp(-1,1)
                    recons.extend(Image.fromarray(im) for im in ((y+1)*127.5).byte().permute(0,2,3,1).cpu().numpy())
            rc,rm=classify(recons)
            summary.update(roundtrip_gt_micro=float(rc.mean()),roundtrip_gt_margin=float(rm.mean()))
            del vae
        from src.eval.metrics import ssim,frag_ratio
        from src.eval.metrics_ink import ink_iou
        def metrics(i):
            x=np.asarray(imgs[i],dtype=np.float32)/255;y=np.asarray(gts[i],dtype=np.float32)/255
            wx,wy=ridge_widths(imgs[i].convert('L')),ridge_widths(gts[i].convert('L'))
            if wx is None or wy is None:raise ValueError(f'Empty stroke sample {i}; do not silently change sample set')
            return dict(ssim=ssim(x,y),ink_iou=ink_iou(x,y),frag=frag_ratio(x,y),
                        soft_ink=float((1-x).sum()/max((1-y).sum(),1e-6)),
                        width_w1=float(wasserstein_distance(wx,wy)),width_p50=float(np.median(wx)),gt_width_p50=float(np.median(wy)))
        with ThreadPoolExecutor(max_workers=8) as pool:
            for r,m in zip(records,pool.map(metrics,range(n))):r.update(m)
        del judge;torch.cuda.empty_cache()
        import lpips
        lp=lpips.LPIPS(net='vgg').cuda().eval()
        with torch.inference_mode():
            for start in range(0,n,16):
                def tensor(images):return torch.from_numpy(np.stack([np.asarray(x).copy() for x in images])).permute(0,3,1,2).cuda().float()/127.5-1
                values=lp(tensor(imgs[start:start+16]),tensor(gts[start:start+16])).flatten().cpu().tolist()
                for r,v in zip(records[start:start+16],values):r['lpips']=v
        for k in ('ssim','lpips','ink_iou','frag','soft_ink','width_w1','width_p50','gt_width_p50'):summary[k]=float(np.mean([r[k] for r in records]))
        summary['frag_median']=float(np.median([r['frag'] for r in records]))
    if a.baseline:
        base=json.loads((Path(a.baseline)/'scores.json').read_text())['per_image']
        if [(r['char'],r['writer']) for r in records]!=[(r['char'],r['writer']) for r in base]:raise ValueError('Paired sample identities differ')
        summary['paired_delta']={k:paired_ci([r[k]-b[k] for r,b in zip(records,base)],[r['char'] for r in records]) for k in ('margin','correct')}
    save(root/'scores.json',dict(summary=summary,per_image=records));print('SCORED',a.run,summary,flush=True)


if __name__=='__main__':main()
