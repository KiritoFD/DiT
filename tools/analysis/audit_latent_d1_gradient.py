"""Frozen v84A latent one-step audit. No DiT forward, weights or training writes."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ.setdefault(key,'4')
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from scipy.ndimage import label
from skimage.metrics import structural_similarity
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from diffusers import AutoencoderKL
from src.model.writer_style_latent import build_writer_style_latent
from src.model.writer_style_pixel import WriterStylePixel
from src.eval.style.brush import features_from_gray
from tools.eval_gen_with_disc import Enc,MEAN,STD
from tools.analysis.latent_style_gradient import (rms,bounded_step,reference_ids,prototypes,
    hard_reference_features,contrastive_per_image,prototype_scores,validate_pretrained_archive)
from tools.analysis.run_style_fingerprint_audit import interval


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def pixel_input(rgb):
    gray=(rgb*rgb.new_tensor([.299,.587,.114])[None,:,None,None]).sum(1,keepdim=True)
    return F.interpolate(gray,size=(128,128),mode='bicubic',align_corners=False,antialias=True).clamp(0,1)


def independent_tensor(images):
    x=np.stack([np.asarray(Image.fromarray(im).resize((160,160),Image.Resampling.BICUBIC)).copy() for im in images])
    t=torch.from_numpy(x).cuda().permute(0,3,1,2).float()/255
    return (t-t.new_tensor(MEAN)[None,:,None,None])/t.new_tensor(STD)[None,:,None,None]


@torch.no_grad()
def independent_train_bank(model,data,rows,out):
    cache=out/'independent_train_features.npz'
    if cache.exists():return np.load(cache)['embedding']
    result=np.zeros((len(data['z']),256),np.float32);train=data['splits']['train']
    def read(i):
        with Image.open(rows[i]['image_path']) as image:return np.asarray(image.convert('RGB').resize((160,160),Image.Resampling.BICUBIC)).copy()
    with ThreadPoolExecutor(max_workers=8) as pool:
        for start in range(0,len(train),64):
            ids=train[start:start+64];images=list(pool.map(read,ids))
            result[ids]=model.emb(independent_tensor(images)).cpu().numpy()
            if start%2048==0:print('INDEPENDENT_TRAIN_BANK',start,'/',len(train),flush=True)
    np.savez_compressed(cache,embedding=result)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True)
    p.add_argument('--negative-protocol',choices=('hard_samechar','prototype'),default='hard_samechar')
    p.add_argument('--lpips-backbone',help='Verified alexnet-owt-7be5be79.pth in an isolated checkpoints directory')
    p.add_argument('--latent-run',help='Selected encoder run directory; defaults to original latent D1')
    p.add_argument('--generation-bank',help='Shared expanded bank containing latents.pt and manifest.json')
    p.add_argument('--step-sizes',default='0.001,0.003,0.01,0.03')
    p.add_argument('--roundtrip',action='store_true')
    p.add_argument('--reference-cache',help='Shared immutable independent-judge training feature cache')
    p.add_argument('--validation-smoke',action='store_true',help='Infrastructure-only tiny bank; never an acceptance result')
    a=p.parse_args();root=Path(a.root);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    if (out/'results.json').exists():raise FileExistsError('Use a fresh completed-audit output directory')
    torch.set_num_threads(4);torch.manual_seed(89);torch.cuda.set_per_process_memory_fraction(.16);start=time.time()
    backbone=Path(a.lpips_backbone) if a.lpips_backbone else Path(torch.hub.get_dir())/'checkpoints/alexnet-owt-7be5be79.pth'
    backbone_sha=validate_pretrained_archive(backbone,'7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02')
    data=torch.load(root/'assets/writer_latent_data_20261016/data.pt',map_location='cpu',weights_only=False)
    latent_run=Path(a.latent_run) if a.latent_run else root/'scratch/writer_latent_d0d1_20261016_01/assets/D1'
    pixel_run=root/'scratch/writer_pixel_d0d1_20261016_01/assets/D1'
    lc=torch.load(latent_run/'best.pt',map_location='cpu',weights_only=False)
    pc=torch.load(pixel_run/'best.pt',map_location='cpu',weights_only=False)
    latent=build_writer_style_latent(len(data['slots']),lc['config'].get('architecture','d1')).cuda().eval().requires_grad_(False);latent.load_state_dict(lc['model'],strict=True)
    pixel=WriterStylePixel(len(data['slots'])).cuda().eval().requires_grad_(False);pixel.load_state_dict(pc['model'],strict=True)
    source=Path(a.generation_bank) if a.generation_bank else root/'assets/writer_counterfactual_20261014';manifest=json.loads((source/'manifest.json').read_text(encoding='utf-8'))
    canonical=root/'assets/runs/v84A_std_cond/20261013-072215-v84A-std-cond/checkpoints/0200000.pt'
    assert sha(canonical)==manifest['checkpoint_sha256'],'Counterfactual checkpoint identity mismatch'
    assert manifest['sf']==.18215 and manifest['shift']==0 and manifest['sde_gamma']==0
    zs=torch.load(source/'latents.pt',map_location='cpu',weights_only=True).float() if a.generation_bank else torch.cat([torch.load(source/f'latent_{i}.pt',map_location='cpu',weights_only=True).float() for i in manifest['selected_eval_indices']])
    metadata=[];ids=[];excluded=[]
    for i,row in enumerate(manifest['per_image']):
        positive,negative=reference_ids(data,row['writer'],row['script'],row['character'])
        supported=(row['writer'],row['script']) in data['slots']
        if supported and len(positive) and (len(negative) or a.negative_protocol=='prototype'):metadata.append(row);ids.append(i)
        else:excluded.append({'original_image_index':i,'supported':supported,'positive_count':len(positive),'samechar_negative_count':len(negative)})
    print('REFERENCE_COVERAGE',json.dumps({'protocol':a.negative_protocol,'eligible_queries':len(ids),'excluded_count':len(excluded)},ensure_ascii=False),flush=True)
    (out/'reference_coverage.json').write_text(json.dumps({'protocol':a.negative_protocol,'eligible_queries':len(ids),'excluded':excluded},ensure_ascii=False,indent=2),encoding='utf-8')
    assert len(ids)>=(2 if a.validation_smoke else 16),'Too few legal train-only queries; do not silently switch protocols'
    z=zs[ids].cuda();n=len(z);target=torch.tensor([data['slots'].index((r['writer'],r['script'])) for r in metadata],device='cuda')
    features={'latent_D1':np.load(latent_run/'clean_features.npz')['embedding'],
              'pixel_D1':np.load(pixel_run/'clean_features.npz')['embedding']}
    references={}
    for name,f in features.items():
        proto,mask=prototypes(f,data,metadata)
        if a.negative_protocol=='hard_samechar':pos,neg,nmask=hard_reference_features(f,data,metadata,proto)
        else:
            target_cpu=target.cpu();pos=proto[torch.arange(n),target_cpu];neg=proto.clone();nmask=mask.clone()
            nmask[torch.arange(n),target_cpu]=False
        references[name]=tuple(t.cuda() for t in (proto,mask,pos,neg,nmask))
    _,_,pos,neg,nmask=references['latent_D1']
    zz=z.detach().clone().requires_grad_(True);loss=contrastive_per_image(latent(zz)['embedding'],pos,neg,nmask).mean()
    gradient=torch.autograd.grad(loss,zz)[0].detach();assert torch.isfinite(gradient).all() and rms(gradient).gt(0).all()
    assert all(p.grad is None for p in latent.parameters())
    generator=torch.Generator(device='cuda').manual_seed(89)
    random=torch.randn(z.shape,device='cuda',generator=generator)
    variants={'baseline':z.detach()}
    for rho in map(float,a.step_sizes.split(',')):variants[f'style_{rho:g}']=bounded_step(z,gradient,rho)
    variants['inverse_0.01']=bounded_step(z,-gradient,.01);variants['random_0.01']=bounded_step(z,random,.01)
    torch.save({'z':z.cpu(),'gradient':gradient.cpu(),'input_indices':ids,'variants':{k:v.cpu() for k,v in variants.items()}},out/'latent_steps.pt')
    print('GRADIENT_INPUTS',n,'supported samples',a.negative_protocol,'excluded',len(excluded),'unique_chars',len({r['character'] for r in metadata}),flush=True)
    vae=AutoencoderKL.from_pretrained(root/'data/pretrained/sd-vae-ft-ema').cuda().eval().requires_grad_(False)
    images={};roundtrips={}
    with torch.no_grad():
        for name,v in variants.items():
            rgb=torch.cat([((vae.decode(v[s:s+4]/.18215).sample.float().clamp(-1,1)+1)/2).cpu() for s in range(0,n,4)])
            images[name]=rgb
            if a.roundtrip:
                roundtrips[name]=torch.cat([(vae.encode(2*rgb[s:s+4].cuda()-1).latent_dist.mean*.18215).cpu() for s in range(0,n,4)])
            directory=out/name;directory.mkdir(exist_ok=True)
            for i,im in enumerate(rgb):Image.fromarray((im.permute(1,2,0).numpy()*255).round().astype(np.uint8)).save(directory/f'g{i}.png')
            print('DECODED',name,flush=True)
    if a.roundtrip:torch.save(roundtrips,out/'roundtrip_latents.pt')
    del vae;torch.cuda.empty_cache()
    # Verify cached generator latent / decoder replay before interpreting changes.
    replay=None
    if not a.generation_bank:
        stored=torch.stack([torch.from_numpy(np.asarray(Image.open(source/f'g{i}.png').convert('RGB')).copy()).permute(2,0,1).float()/255 for i in ids])
        replay=float((images['baseline']-stored).abs().mean());assert replay<.01,replay
    oldck=root/'assets/style_disc_ce_holdout.pt';ck=torch.load(oldck,map_location='cpu',weights_only=False)
    independent=Enc(d=ck['d'],n_cls=ck['n_cls']).cuda().eval().requires_grad_(False);independent.load_state_dict(ck['model'],strict=True)
    rows=[r for r in csv.DictReader(open(root/'exp-std/csv/train.csv',encoding='utf-8-sig')) if not (r.get('aug') or '').strip()]
    assert len(rows)==len(data['z'])
    assert all((r['calligrapher'],r['script'],r['character'])==(w,s,c) for r,w,s,c in zip(rows,data['writer'],data['script'],data['char']))
    reference_cache=Path(a.reference_cache) if a.reference_cache else out;reference_cache.mkdir(parents=True,exist_ok=True)
    bank=independent_train_bank(independent,data,rows,reference_cache);proto,mask=prototypes(bank,data,metadata)
    independent_refs=(proto.cuda(),mask.cuda())
    import lpips
    original_hub=torch.hub.get_dir()
    try:
        # Isolated verified cache: do not overwrite the user's corrupt global cache.
        torch.hub.set_dir(str(backbone.parent.parent))
        perceptual=lpips.LPIPS(net='alex',verbose=False).cuda().eval().requires_grad_(False)
    finally:torch.hub.set_dir(original_hub)
    report={'source':{'manifest_sha256':sha(source/'manifest.json'),'generator_sha256':manifest['checkpoint_sha256'],
            'latent_d1_step':lc['step'],'pixel_d1_step':pc['step'],'latent_d1_sha256':sha(latent_run/'best.pt'),
            'pixel_d1_sha256':sha(pixel_run/'best.pt'),'independent_ce_sha256':sha(oldck),'latent_data_manifest_sha256':sha(root/'assets/writer_latent_data_20261016/manifest.json'),
            'lpips_alexnet_sha256':backbone_sha,
            'latent_arm':lc['config']['arm'],'latent_architecture':lc['config'].get('architecture','d1'),
            'source_latent_sha256':{'shared_bank':sha(source/'latents.pt')} if a.generation_bank else {str(i):sha(source/f'latent_{i}.pt') for i in manifest['selected_eval_indices']}},
        'protocol':{'kind':'single-step frozen generated latent; NO DiT weight update or trajectory intervention','validation_smoke_only':a.validation_smoke,'n':n,'input_indices':ids,
            'characters':sorted({r['character'] for r in metadata}),'negative_protocol':a.negative_protocol,
            'negative':'strict same-character different-writer TRAIN GT only' if a.negative_protocol=='hard_samechar' else 'same-script different-writer TRAIN prototypes; query character excluded; NOT hard samechar loss',
            'positive':'train-only same-writer different-character, character/source balanced prototype',
            'prototype_metric':'same-script slots only, query character excluded from all prototypes; not the historical shuffled6000 23-way metric',
            'relative_rms_steps':list(map(float,a.step_sizes.split(','))),'decoder_replay_mae_to_stored_png':replay,'excluded':excluded,
            'metric_aliases':{'latent_D1':'Legacy field name for SELECTED latent arm, see source.latent_arm'},
            'seeds':manifest.get('seeds',[manifest.get('seed',0)]),'roundtrip':'posterior.mean, FP32, decoded image clamp [-1,1], sf .18215, zero shift' if a.roundtrip else None},
        'limitations':['Finite character/seed coverage explicitly recorded in protocol; not full187 validation.',
            'Pixel D1 and old CE are separate weights, not statistically independent of the training data or shortcuts.',
            'LPIPS/SSIM/InkIoU here compare against original generated image, NOT GT quality.',
            'Small latent steps do not establish trainable DiT controllability or FM-gradient compatibility.',
            'Source directories are proxies; no cross-source generation guarantee.'],
        'frozen_parameter_gradient_check':all(p.grad is None for p in latent.parameters()),'variants':{}}
    records={};base=images['baseline'];base_gray=pixel_input(base).numpy()[:,0];base_ink=base_gray<.55
    with torch.no_grad():
        for name,v in variants.items():
            rgb=images[name];cols={};le=latent(v)['embedding'];pemb=[];oemb=[];lp=[]
            for s in range(0,n,8):
                x=rgb[s:s+8].cuda();pemb.append(pixel(pixel_input(x))['embedding'].cpu())
                arrays=(x.cpu().permute(0,2,3,1).numpy()*255).round().astype(np.uint8)
                oemb.append(independent.emb(independent_tensor(arrays)).cpu())
                lp.extend(perceptual(2*x-1,2*base[s:s+8].cuda()-1).flatten().cpu().tolist())
            for judge,e in (('latent_D1',le),('pixel_D1',torch.cat(pemb).cuda()),('independent_CE',torch.cat(oemb).cuda())):
                if judge=='independent_CE':pr,ma=independent_refs
                else:pr,ma,po,ne,nm=references[judge];cols[f'{judge}_contrastive_loss']=contrastive_per_image(e,po,ne,nm).cpu().numpy()
                for field,values in prototype_scores(e,pr,ma,target).items():cols[f'{judge}_{field}']=values.cpu().numpy()
            gray=pixel_input(rgb).numpy()[:,0];ink=gray<.55
            cols['relative_latent_rms']=(rms(v-z)/rms(z)).cpu().numpy();cols['lpips_to_baseline']=np.array(lp)
            cols['ssim_to_baseline']=np.array([structural_similarity(b,g,data_range=1.) for b,g in zip(base_gray,gray)])
            cols['ink_iou_to_baseline']=(ink&base_ink).sum((1,2))/np.maximum((ink|base_ink).sum((1,2)),1)
            cols['ink_fraction']=ink.mean((1,2));cols['connected_components']=np.array([label(x)[1] for x in ink])
            widths=[]
            for x in (rgb.permute(0,2,3,1).numpy()*255).round().astype(np.uint8):widths.append(features_from_gray(x).width_stats()['p50'])
            cols['stroke_width_median']=np.array(widths)
            records[name]=cols
            report['variants'][name]={'means':{k:float(np.mean(vv)) for k,vv in cols.items()},'per_image':{k:vv.tolist() for k,vv in cols.items()}}
            if a.roundtrip:
                rt=roundtrips[name].cuda();pr,ma,_,_,_=references['latent_D1'];rs=prototype_scores(latent(rt)['embedding'],pr,ma,target)
                report['variants'][name]['roundtrip']={'relative_rms':(rms(rt-v)/rms(v)).cpu().tolist(),
                    'prototype_top1':rs['top1'].cpu().tolist(),'margin':rs['margin'].cpu().tolist()}
            print('GRADIENT_METRICS',name,json.dumps(report['variants'][name]['means']),flush=True)
    del independent,perceptual;torch.cuda.empty_cache()
    # Frozen external OCR head: agreement with GT's predicted class is a proxy,
    # not true OCR accuracy. Ground-truth class IDs are not inferred from writers.
    from src.eval.style.encoder import StyleEncoder
    ocr=StyleEncoder.load(str(root/'pretrained_models/eval_classifier.pt'),device='cuda')
    eval_rows=list(csv.DictReader(open(root/'exp-std/csv/eval200_fixed.csv',encoding='utf-8-sig')))
    originals=sorted({r['original_idx'] for r in metadata});gt_arrays=[]
    for i in originals:
        with Image.open(eval_rows[i]['image_path']) as im:gt_arrays.append(np.asarray(im.convert('RGB')).copy())
    with torch.no_grad():
        gt_pred=ocr.net(ocr.to_tensor(gt_arrays).cuda())[0].argmax(1).cpu().numpy()
        gt_lookup=dict(zip(originals,gt_pred));targets=np.array([gt_lookup[r['original_idx']] for r in metadata]);baseline_pred=None
        for name,rgb in images.items():
            preds=[]
            for s in range(0,n,8):
                arrays=(rgb[s:s+8].permute(0,2,3,1).numpy()*255).round().astype(np.uint8)
                preds.extend(ocr.net(ocr.to_tensor(arrays).cuda())[0].argmax(1).cpu().tolist())
            preds=np.array(preds)
            if name=='baseline':baseline_pred=preds
            records[name]['ocr_head_agrees_gt']=(preds==targets).astype(float)
            records[name]['ocr_head_agrees_baseline']=(preds==baseline_pred).astype(float)
            for key in ('ocr_head_agrees_gt','ocr_head_agrees_baseline'):
                report['variants'][name]['means'][key]=float(records[name][key].mean())
                report['variants'][name]['per_image'][key]=records[name][key].tolist()
    for name,cols in records.items():
        if name=='baseline':continue
        report['variants'][name]['paired_delta_vs_baseline']={}
        for key,values in cols.items():
            delta=values-records['baseline'][key]
            report['variants'][name]['paired_delta_vs_baseline'][key]={
                'by_character':interval(delta,np.array([r['character'] for r in metadata])),
                'by_writer':interval(delta,np.array([r['writer'] for r in metadata])),
                'note':'Interval accuracy field is mean metric delta, not a probability.'}
            if a.generation_bank:
                report['variants'][name]['paired_delta_vs_baseline'][key]['by_reference_source_proxy']=interval(delta,np.array([r['gt_source_proxy'] for r in metadata]))
        if a.roundtrip:
            rt=report['variants'][name]['roundtrip'];brt=report['variants']['baseline']['roundtrip']
            original_gain=cols['latent_D1_margin']-records['baseline']['latent_D1_margin']
            retained=np.array(rt['margin'])-brt['margin']
            rt['mean_margin_gain_before']=float(original_gain.mean());rt['mean_margin_gain_after']=float(retained.mean())
            rt['gain_retention_ratio']=float(retained.mean()/original_gain.mean()) if abs(original_gain.mean())>1e-8 else None
    report['elapsed_seconds']=time.time()-start;report['peak_cuda_allocated_GiB']=torch.cuda.max_memory_allocated()/1024**3
    report['metadata']=metadata;report['next_action']='Research review only; no automatic DiT or D2/D3 training.'
    (out/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('LATENT_GRADIENT_AUDIT_COMPLETE',n,'seconds',int(time.time()-start),'peak_GiB',report['peak_cuda_allocated_GiB'],flush=True)


if __name__=='__main__':main()
