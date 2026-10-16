"""Post-training acceptance on the entire held-out sets and exported prototypes.

Does not select checkpoints or fit anything to test data. Historical critic is
not a clean baseline. A completed run is not automatically an accepted encoder.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.model.style_encoder import build_style_encoder
from tools.train_prototype_style_encoder import retrieval_metrics, source_balanced_prototypes, sha


def wrong_std_lookup(std_scripts, std_chars, row_scripts, row_chars):
    """Materialized metadata only: never decompress NPZ members in a hot loop."""
    choices = {}
    for j,(s,c) in enumerate(zip(std_scripts,std_chars)):
        choices.setdefault(str(s),{}).setdefault(str(c),j)
    result = []
    for s,c in zip(row_scripts,row_chars):
        bank = choices[str(s)]
        other = next((value for char,value in bank.items() if char!=str(c)),None)
        if other is None:
            raise ValueError(f'No different-character STD in script {s}')
        result.append(other)
    return np.asarray(result,dtype=np.int64)


def same_char_2afc(features, labels, scripts, chars, prototypes, seed=89):
    """Target prototype: held-out real glyph vs same-char/script other writer."""
    rng = np.random.default_rng(seed)
    correct, margins = [], []
    for i in range(len(features)):
        negatives = np.flatnonzero((scripts==scripts[i]) & (chars==chars[i]) & (labels!=labels[i]))
        if not len(negatives):
            continue
        j = int(rng.choice(negatives))
        margin = float((features[i]-features[j]) @ prototypes[labels[i]])
        margins.append(margin)
        correct.append(float(margin>0)+.5*float(margin==0))
    return {'accuracy':float(np.mean(correct)) if correct else None,
            'n_valid':len(correct),'n_skipped':len(features)-len(correct),
            'mean_margin':float(np.mean(margins)) if margins else None,
            'chance':.5,'definition':'prototype-targeted, same-character/script alternative writer; ties score 0.5'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',required=True)
    p.add_argument('--data',default='scratch/disc_lat_full.npz')
    p.add_argument('--device',default='cuda')
    a = p.parse_args()
    torch.set_num_threads(4)
    run = Path(a.run)
    manifest = json.loads((run/'manifest.json').read_text())
    ck = torch.load(run/'best.pt',map_location='cpu',weights_only=False)
    exported = torch.load(run/'prototypes.pt',map_location='cpu',weights_only=False)
    assert ck['manifest_sha256']==sha(run/'manifest.json')==exported['manifest_sha256']
    assert ck['step']==exported['encoder_step']
    split = np.load(run/'split_indices.npz')
    assert np.array_equal(exported['reference_indices'],split['train'])
    slots, scripts = ck['slots'],ck['scripts']
    real_slots = exported['slots']
    assert slots[:len(real_slots)]==real_slots
    model = build_style_encoder(len(slots),len(scripts),ck.get('architecture','legacy')).to(a.device).eval()
    model.load_state_dict(ck['model'],strict=True)
    model.requires_grad_(False)
    d = np.load(a.data,allow_pickle=True)
    assert sha(a.data)==manifest['data_sha256']
    labels = np.array([slots.index(str(w)+'·'+str(s)) for w,s in zip(d['writer'],d['script'])])
    si = np.array([scripts.index(str(s)) for s in d['script']])
    chars, sources = d['char'].astype(str),d['work'].astype(str)
    slot_scripts = np.array([scripts.index(s.rsplit('·',1)[1]) for s in real_slots])
    z = torch.from_numpy(d['z'].copy())
    g = 1-torch.from_numpy(d['g_std'].copy()).float().unsqueeze(1)/255
    gi = d['std_idx'].astype(np.int64)
    wrong_std = wrong_std_lookup(d['std_script'].astype(str),d['std_char'].astype(str),
                                d['script'].astype(str),chars)
    assert not set(chars[split['train']]) & set(chars[split['test_char']])
    assert not set(chars[split['train']]) & set(chars[split['val_char']])
    assert not set(zip(labels[split['train']],sources[split['train']])) & set(zip(labels[split['test_source']],sources[split['test_source']]))

    @torch.no_grad()
    def embed(indices,probe='normal'):
        result,pred = [],[]
        for start in range(0,len(indices),256):
            ii = np.asarray(indices[start:start+256])
            zz = z[ii].to(a.device).float()
            gg = g[gi[ii]].to(a.device)
            if probe=='white_latent':
                zz = torch.from_numpy(d['z_probe'][0].copy()).to(a.device).float()[None].expand_as(zz)
            elif probe=='no_std':
                gg = torch.zeros_like(gg)
            elif probe=='wrong_std_same_script':
                # Deterministic unrelated STD in the same script; distinct char required.
                gg = g[wrong_std[ii]].to(a.device)
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=a.device.startswith('cuda')):
                value = model(zz,gg,torch.as_tensor(si[ii],device=a.device))
            result.append(value['embedding'].float().cpu().numpy())
            pred.append(value['slot_logits'].float().argmax(1).cpu().numpy())
        return np.concatenate(result),np.concatenate(pred)

    proto = exported['prototypes'].numpy()
    assert proto.shape==(len(real_slots),256) and np.isfinite(proto).all()
    assert np.allclose(np.linalg.norm(proto,axis=1),1,atol=1e-5)
    train_features,train_predictions = embed(split['train'])
    reconstructed = source_balanced_prototypes(train_features,labels[split['train']],sources[split['train']],len(real_slots))
    assert np.max(np.abs(reconstructed-proto))<1e-4, 'Exported prototypes failed deterministic replay'
    gallery = split['gallery']
    gf,_ = embed(gallery)
    capped_proto = source_balanced_prototypes(gf,labels[gallery],sources[gallery],len(real_slots))
    final = json.loads((run/'final_results.json').read_text())
    replay = {}
    full = {}
    feature_cache = {}
    for key in ('val_char','test_char','test_source'):
        capped = split['eval_'+key]
        qf,pred = embed(capped)
        m = retrieval_metrics(qf,gf,labels[capped],labels[gallery],si[capped],si[gallery],chars[capped],chars[gallery],capped_proto,slot_scripts)
        for metric in ('cross_char_map','prototype_top1','prototype_macro_top1'):
            assert abs(m[metric]-final['metrics'][key][metric])<1e-5, (key,metric)
        replay[key] = {'n':len(capped),'matches_training_final_report':True}
        indices = split[key]
        qf,pred = embed(indices)
        feature_cache[key]=qf
        m = retrieval_metrics(qf,gf,labels[indices],labels[gallery],si[indices],si[gallery],chars[indices],chars[gallery],proto,slot_scripts)
        m['slot_ce_top1']=float(np.mean(pred==labels[indices]))
        m['same_char_2afc']=same_char_2afc(qf,labels[indices],si[indices],chars[indices],proto)
        full[key]=m
        print('FULL_HOLDOUT',key,json.dumps(m),flush=True)
    sample = split['eval_test_char']
    normal,_ = embed(sample)
    normal_metrics = retrieval_metrics(normal,gf,labels[sample],labels[gallery],si[sample],si[gallery],
                                       chars[sample],chars[gallery],proto,slot_scripts)
    probes = {'normal':{'prototype_top1':normal_metrics['prototype_top1'],
                        'cross_char_map':normal_metrics['cross_char_map'],
                        'mean_cosine_to_normal':1.,'n':len(sample)}}
    for name in ('white_latent','no_std','wrong_std_same_script'):
        f,_ = embed(sample,name)
        m = retrieval_metrics(f,gf,labels[sample],labels[gallery],si[sample],si[gallery],chars[sample],chars[gallery],proto,slot_scripts)
        probes[name]={'prototype_top1':m['prototype_top1'],'cross_char_map':m['cross_char_map'],
                      'mean_cosine_to_normal':float(np.mean((normal*f).sum(1))), 'n':len(sample)}
    eigen = np.linalg.eigvalsh(np.cov(train_features.T)).clip(0)
    weights = eigen/max(eigen.sum(),1e-12)
    effective_rank = float(np.exp(-(weights*np.log(weights+1e-12)).sum()))
    similarity = proto@proto.T
    mask=(slot_scripts[:,None]==slot_scripts[None,:]) & ~np.eye(len(proto),dtype=bool)
    # End-to-end differentiability check is required for future generator supervision.
    zz=z[sample[:2]].float().to(a.device).requires_grad_(True)
    output=model(zz,g[gi[sample[:2]]].to(a.device),torch.as_tensor(si[sample[:2]],device=a.device))
    target=torch.from_numpy(proto[labels[sample[:2]]]).to(a.device)
    gradient=torch.autograd.grad((1-(output['embedding']*target).sum(1)).mean(),zz)[0]
    assert torch.isfinite(gradient).all() and gradient.abs().sum()>0
    report={'selected_step':ck['step'],'best_sha256':sha(run/'best.pt'),
        'acceptance_tool_sha256':sha(__file__),
        'prototype_sha256':sha(run/'prototypes.pt'),'checkpoint_and_prototype_replay':'passed',
        'capped_metric_replay':replay,'full_holdout_exported_prototypes':full,
        'train_slot_ce_top1':float(np.mean(train_predictions==labels[split['train']])),
        'embedding_covariance_effective_rank':effective_rank,
        'same_script_prototype_cosine_mean':float(similarity[mask].mean()),
        'same_script_prototype_cosine_max':float(similarity[mask].max()),
        'input_probes':probes,'frozen_encoder_input_gradient_rms':float(gradient.square().mean().sqrt()),
        'limitations':manifest['limitations']+['Input corruption probes are OOD sensitivity tests, not image-quality measurements',
            'full_holdout uses exported ALL-train prototypes; training final_report used capped-gallery prototypes',
            'same-character 2AFC is prototype-targeted, not the old critic logit-based T1 protocol; not directly comparable']}
    (run/'acceptance.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print('ENCODER_ACCEPTANCE_COMPLETE',flush=True)


if __name__=='__main__':
    main()
