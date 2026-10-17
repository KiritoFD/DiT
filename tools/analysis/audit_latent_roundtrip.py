"""Deterministic VAE projection diagnostic, not a true manifold distance."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from diffusers import AutoencoderKL
from src.model.writer_style_latent import build_writer_style_latent
from tools.analysis.latent_style_gradient import prototypes,prototype_scores,rms
from tools.analysis.run_style_fingerprint_audit import interval


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--source-audit',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    root=Path(a.root);source=Path(a.source_audit);out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);report=json.loads((source/'results.json').read_text(encoding='utf-8'))
    values=torch.load(source/'latent_steps.pt',map_location='cpu',weights_only=True);data=torch.load(root/'assets/writer_latent_data_20261016/data.pt',map_location='cpu',weights_only=False)
    run=root/'scratch/writer_latent_d0d1_20261016_01/assets/D1';ck=torch.load(run/'best.pt',map_location='cpu',weights_only=False)
    model=build_writer_style_latent(len(data['slots']),ck['config'].get('architecture','d1')).cuda().eval().requires_grad_(False);model.load_state_dict(ck['model'],strict=True)
    proto,mask=prototypes(np.load(run/'clean_features.npz')['embedding'],data,report['metadata']);proto=proto.cuda();mask=mask.cuda()
    target=torch.tensor([data['slots'].index((r['writer'],r['script'])) for r in report['metadata']],device='cuda')
    vae=AutoencoderKL.from_pretrained(root/'data/pretrained/sd-vae-ft-ema').cuda().eval().requires_grad_(False)
    result={'contract':'z/.18215 -> decode -> clamp[-1,1] -> encode posterior.mean * .18215; zero shift; FP32; no posterior sampling',
            'limitations':'Roundtrip discrepancy is NOT manifold distance. Compare to baseline; retained self-score is not independent style proof. Eight characters, one seed.',
            'variants':{}};saved={}
    with torch.no_grad():
        for name,z in values['variants'].items():
            rt=[]
            for s in range(0,len(z),4):
                zz=z[s:s+4].cuda();pixels=vae.decode(zz/.18215).sample.clamp(-1,1)
                rt.append((vae.encode(pixels).latent_dist.mean*.18215).cpu())
            rz=torch.cat(rt);saved[name]=rz;before=model(z.cuda())['embedding'];after=model(rz.cuda())['embedding']
            bs=prototype_scores(before,proto,mask,target);rs=prototype_scores(after,proto,mask,target)
            err=(rms(z-rz)/rms(z)).numpy()
            result['variants'][name]={'mean_relative_roundtrip_rms':float(err.mean()),'relative_roundtrip_rms':err.tolist(),
                'before_top1':float(bs['top1'].mean()),'after_top1':float(rs['top1'].mean()),
                'before_margin':bs['margin'].cpu().tolist(),'after_margin':rs['margin'].cpu().tolist()}
            print('ROUNDTRIP_VARIANT',name,result['variants'][name]['before_top1'],result['variants'][name]['after_top1'],'rms',float(err.mean()),flush=True)
    baseline=result['variants']['baseline'];chars=np.array([r['character'] for r in report['metadata']])
    for name,item in result['variants'].items():
        if name=='baseline':continue
        gain=np.array(item['before_margin'])-baseline['before_margin'];retained=np.array(item['after_margin'])-baseline['after_margin']
        item['mean_margin_gain_before']=float(gain.mean());item['mean_margin_gain_after']=float(retained.mean())
        item['gain_retention_ratio']=float(retained.mean()/gain.mean()) if abs(gain.mean())>1e-8 else None
        item['roundtrip_gain_by_character']=interval(retained,chars)
    torch.save(saved,out/'roundtripped_latents.pt');(out/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print('ROUNDTRIP_AUDIT_COMPLETE',out,flush=True)


if __name__=='__main__':main()
