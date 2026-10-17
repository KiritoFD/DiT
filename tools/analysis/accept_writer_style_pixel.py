"""Heldout evaluation AFTER checkpoint selection on val only. Never launches DiT."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from src.model.writer_style_pixel import WriterStylePixel
from tools.train_writer_style_pixel import embed,evaluate_features,sha
from tools.analysis.run_style_fingerprint_audit import interval


def grouped(features,pairs,data):
    margin=(features[pairs[:,0]]*features[pairs[:,1]]).sum(1)-(features[pairs[:,0]]*features[pairs[:,2]]).sum(1)
    correct=(margin>0).astype(float)+.5*(margin==0)
    used=set();keep=[]
    for i,tri in enumerate(pairs):
        if not used.intersection(tri):used.update(tri);keep.append(i)
    keep=np.array(keep,int)
    return {'by_char':interval(correct,data['char'][pairs[:,0]]),
            'by_source':interval(correct,data['source'][pairs[:,0]]),
            'image_disjoint':interval(correct[keep],data['char'][pairs[keep,0]]),
            'mean_margin':float(margin.mean())}


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--data-root',required=True)
    p.add_argument('--validation-only',action='store_true',help='Infrastructure smoke without test metrics')
    a=p.parse_args();torch.set_num_threads(4);run=Path(a.run)
    data=torch.load(Path(a.data_root)/'data.pt',map_location='cpu',weights_only=False)
    ck=torch.load(run/'best.pt',map_location='cpu',weights_only=False)
    assert ck['config']['data_manifest_sha256']==sha(Path(a.data_root)/'manifest.json')
    domain=ck['config'].get('domain','pixel')
    if domain=='latent':
        from src.model.writer_style_latent import build_writer_style_latent
        factory=lambda n:build_writer_style_latent(n,ck['config'].get('architecture','d1'))
    else:factory=WriterStylePixel
    model=factory(len(data['slots'])).cuda().eval().requires_grad_(False)
    model.load_state_dict(ck['model'],strict=True);gray=data['z' if domain=='latent' else 'gray'].cuda();indices=np.arange(len(gray));tr=data['splits']['train']
    report={'step':ck['step'],'checkpoint_sha256':sha(run/'best.pt'),'data_manifest_sha256':ck['config']['data_manifest_sha256'],
            'arm':ck['config']['arm'],'domain':domain,'conditions':'image/latent only; no STD/reference/writer/script input',
            'limitations':['Source directories are proxies, not verified artwork identity.',
                'Grouped bootstrap separately by query character/source; gallery references reused.',
                'Old 78.47% Critic T1 uses a different split and sampling; not an apples-to-apples benchmark.',
                'These are GT encoder metrics, not evidence of usefulness for generated images.'],
            'conditions_evaluated':{}}
    clean_features=None
    for perturb in (None,'blur','ink_gain'):
        features,pred=embed(model,gray,indices,perturb=perturb,domain=domain)
        if perturb is None:clean_features=features
        else:features[tr]=clean_features[tr]  # frozen clean real-reference bank
        result={}
        for key in (('val_char',) if a.validation_only else ('test_char','test_source')):
            q=data['splits'][key];m=evaluate_features(features,pred,data,q,tr)
            m['matched']=grouped(features,data['matched_pairs'][key],data)
            # Target classification-head score, same-character alternatives;
            # independent of the source-balanced feature prototype metric.
            weights=torch.nn.functional.normalize(model.head.weight,dim=-1).detach().cpu().numpy()
            rng=np.random.default_rng(89);outcomes=[];qidx=[]
            for i in q:
                ns=q[(data['char'][q]==data['char'][i])&(data['si'][q]==data['si'][i])&(data['labels'][q]!=data['labels'][i])]
                if len(ns):
                    j=int(rng.choice(ns));margin=float((features[i]-features[j])@weights[data['labels'][i]])
                    outcomes.append(float(margin>0)+.5*float(margin==0));qidx.append(i)
            m['samechar_target_head_2afc']=interval(outcomes,data['char'][qidx]);result[key]=m
        result['train_slot_ce_top1']=float((pred[tr]==data['labels'][tr]).mean())
        report['conditions_evaluated'][perturb or 'clean']=result
        print('PIXEL_ACCEPTANCE',perturb or 'clean',json.dumps(result),flush=True)
    # Differentiability is tested in image space. No unverified latent adapter.
    image=gray[tr[:8]].float()
    if domain=='pixel':image=image.div(255)
    image.requires_grad_(True)
    output=model(image);weights=torch.nn.functional.normalize(model.head.weight,dim=-1)
    loss=-(output['embedding']*weights[torch.from_numpy(data['labels'][tr[:8]]).cuda()]).sum(1).mean()
    grad=torch.autograd.grad(loss,image)[0]
    report['gradient_probe']={'norm':float(grad.norm()),'finite':bool(torch.isfinite(grad).all()),
                              'frozen_parameters':all(p.grad is None for p in model.parameters()),'domain':domain}
    if domain=='latent':report['latent_perturbation_note']='blur is latent-channel average pooling; ink_gain means latent gain .95, NOT a physical image ink/exposure manipulation'
    clean=report['conditions_evaluated']['clean']
    if a.validation_only:passgate=False
    else:
        cc=clean['test_char']['matched']['by_char'];ss=clean['test_source']['matched']
        passgate=cc['accuracy']>=.60 and min(ss['by_char']['ci95'][0],ss['by_source']['ci95'][0])>.50
    report['research_gate_passed']=bool(passgate)
    report['next_action']='Research review only. No automatic DiT training, including on pass.'
    (run/('acceptance_smoke_val_only.json' if a.validation_only else 'acceptance.json')).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    np.savez_compressed(run/'clean_features.npz',embedding=clean_features)
    print('PIXEL_ACCEPTANCE_COMPLETE',run,flush=True)
    if not a.validation_only and report['arm']=='D1':
        from tools.analysis.compare_writer_pixel import compare
        compare(run.parent,a.data_root)


if __name__=='__main__':main()
