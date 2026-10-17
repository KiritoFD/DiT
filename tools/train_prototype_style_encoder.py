"""Train v89's standalone, cross-character style encoder; never touches DiT.

Fresh initialization ensures character-held-out validation is not contaminated
by the old critic's pretraining. Old critic is evaluated as a historical probe,
NOT as a clean holdout baseline (it may have seen these characters/sources).
"""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.model.style_encoder import build_style_encoder, augment_style_inputs
from src.loss.style_encoder_loss import cross_character_supcon, std_separation,same_character_hard_nce
from src.utils.style_encoder_data import CrossCharacterSampler, make_splits


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def source_balanced_prototypes(features, slots, sources, n_slots):
    """Equal weight per source-directory group; L2-normalize each group first."""
    result = []
    for slot in range(n_slots):
        means = []
        for source in sorted(set(sources[slots == slot])):
            x = features[(slots == slot) & (sources == source)].mean(0)
            means.append(x / max(np.linalg.norm(x), 1e-12))
        if not means:
            raise ValueError(f'No training references for slot {slot}')
        mean = np.mean(means, 0)
        result.append(mean / max(np.linalg.norm(mean), 1e-12))
    return np.stack(result)


def retrieval_metrics(query, gallery, qslots, gslots, qscripts, gscripts, qchars, gchars, prototypes, slot_scripts):
    maps, correct, labels, chance = [], [], [], []
    for start in range(0, len(query), 128):
        sim = query[start:start+128] @ gallery.T
        psim = query[start:start+128] @ prototypes.T
        for j in range(len(sim)):
            i = start+j
            candidates = np.flatnonzero((gscripts == qscripts[i]) & (gchars != qchars[i]))
            if not len(candidates):
                continue
            relevant = gslots[candidates] == qslots[i]
            if not relevant.any() or relevant.all():
                continue
            ranked = relevant[np.argsort(-sim[j, candidates], kind='stable')]
            maps.append(float(((np.cumsum(ranked)/(np.arange(len(ranked))+1))*ranked).sum()/ranked.sum()))
            allowed = np.flatnonzero(slot_scripts == qscripts[i])
            correct.append(int(allowed[np.argmax(psim[j, allowed])] == qslots[i]))
            labels.append(int(qslots[i])); chance.append(1/len(allowed))
    by = {int(s): float(np.mean([v for v,l in zip(correct,labels) if l==s])) for s in sorted(set(labels))}
    return {'cross_char_map': float(np.mean(maps)) if maps else None,
            'prototype_top1': float(np.mean(correct)) if correct else None,
            'prototype_macro_top1': float(np.mean(list(by.values()))) if by else None,
            'script_conditioned_chance': float(np.mean(chance)) if chance else None,
            'n_valid': len(maps), 'n_total': len(query), 'per_slot_top1': by}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', default='scratch/disc_lat_full.npz')
    ap.add_argument('--csv', default='exp-std/csv/train.csv')
    ap.add_argument('--out', required=True)
    ap.add_argument('--steps', type=int, default=20000)
    ap.add_argument('--batch', type=int, default=512)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--eval-every', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=89)
    ap.add_argument('--arch',choices=['legacy','local_stats'],default='legacy')
    ap.add_argument('--hard-weight',type=float,default=0.)
    ap.add_argument('--ink-width-stats',default='')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--skip-vae-replay', action='store_true', help='Only for CPU/synthetic tests')
    a = ap.parse_args()
    if a.batch % 4 or a.batch < 8 or a.steps < 1:
        raise ValueError('batch must be divisible by 4 and steps positive')
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(a.seed); np.random.seed(a.seed); random.seed(a.seed)
    device = torch.device(a.device)
    amp = lambda: torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type=='cuda')
    d = np.load(a.data, allow_pickle=True)
    writers, scripts, chars, sources = [d[k].astype(str) for k in ('writer','script','char','work')]
    std_rows = d['std_idx'].astype(np.int64)
    z = torch.from_numpy(d['z'].copy())
    zs = torch.from_numpy(d['z_std'].copy())
    g = 1-torch.from_numpy(d['g_std'].copy()).float().unsqueeze(1)/255
    assert z.shape[1:] == (4,32,32) and zs.shape[1:] == (4,32,32)
    assert (std_rows >= 0).all() and (std_rows < len(g)).all()
    assert np.all(d['std_script'][std_rows].astype(str) == scripts)
    assert np.all(d['std_char'][std_rows].astype(str) == chars)
    rows = [r for r in csv.DictReader(open(a.csv,encoding='utf-8-sig')) if not (r.get('aug') or '').strip()]
    assert len(rows) == len(z), 'Cache/CSV row count mismatch'
    assert all((r['calligrapher'].strip(),r['script'].strip(),r['character'].strip()) == (w,s,c)
               for r,w,s,c in zip(rows,writers,scripts,chars)), 'Cache/CSV order mismatch'
    replay = []
    if not a.skip_vae_replay:
        from diffusers import AutoencoderKL
        from PIL import Image
        vae = AutoencoderKL.from_pretrained('data/pretrained/sd-vae-ft-ema').to(device).eval()
        with torch.no_grad():
            for i in (0, len(rows)//2, len(rows)-1):
                x = torch.from_numpy(np.array(Image.open(rows[i]['image_path']).convert('RGB'))).permute(2,0,1)[None].float().to(device)/127.5-1
                actual = vae.encode(x).latent_dist.mode().float().cpu()*.18215
                error = float((actual-z[i:i+1].float()).abs().max())
                relative = float((actual-z[i:i+1].float()).square().mean().sqrt()/actual.square().mean().sqrt())
                # FP16 cache and different encode batch sizes need an RMS check,
                # not a sub-quantization absolute threshold on the largest value.
                assert error < .02 and relative < .002, f'VAE replay failed: {i} max={error}, relative_rms={relative}'
                replay.append({'row':i,'max_abs_error':error,'relative_rms_error':relative})
        del vae
        torch.cuda.empty_cache()
    real_slots = sorted({w+'·'+s for w,s in zip(writers,scripts)})
    script_names = sorted(set(scripts))
    slots = real_slots + ['std·'+s for s in script_names]
    s2i, k2i = {s:i for i,s in enumerate(script_names)}, {s:i for i,s in enumerate(slots)}
    si = np.array([s2i[s] for s in scripts], dtype=np.int64)
    labels = np.array([k2i[w+'·'+s] for w,s in zip(writers,scripts)], dtype=np.int64)
    char_ids = np.array([ord(c) for c in chars], dtype=np.int64)
    reserved = {r['character'].strip() for r in csv.DictReader(open('exp-std/csv/eval200_fixed.csv',encoding='utf-8-sig'))}
    split = make_splits(chars, labels, sources, reserved, a.seed)
    # Reject exact latent duplicates across splits, even if source metadata differs.
    hashes = [hashlib.blake2b(v.numpy().tobytes(),digest_size=16).hexdigest() for v in z]
    held_hashes = {hashes[i] for k,v in split.items() if k!='train' for i in v}
    old_train_n = len(split['train'])
    split['train'] = np.array([i for i in split['train'] if hashes[i] not in held_hashes], dtype=np.int64)
    assert set(labels[split['train']]) == set(range(len(real_slots)))
    statistics=None
    if a.ink_width_stats:
        stats_cache=np.load(a.ink_width_stats)
        if str(stats_cache['data_sha256'])!=sha(a.data):
            raise ValueError('Ink/width statistics belong to a different latent dataset')
        statistics=stats_cache['stats']
        assert statistics.shape==(len(z),2)
    sampler = CrossCharacterSampler(split['train'], labels, si, chars, sources, a.seed,
                                    balance_sources=a.arch=='local_stats',statistics=statistics)
    # Fixed gallery, source-balanced cap. All exported references come from train only.
    rng = np.random.default_rng(a.seed)
    by_group = defaultdict(list)
    for i in split['train']:
        by_group[(labels[i], sources[i])].append(int(i))
    gallery = np.array([i for k in sorted(by_group) for i in rng.choice(by_group[k], min(128,len(by_group[k])), replace=False)])
    eval_indices = {}
    for key in ('val_char','test_char','test_source'):
        # Deterministic slot-balanced cap; keep all smaller groups.
        eval_indices[key] = np.array([i for s in range(len(real_slots)) for i in
            rng.choice(split[key][labels[split[key]]==s], min(128,int((labels[split[key]]==s).sum())), replace=False)],dtype=np.int64)
        if not len(eval_indices[key]):
            raise ValueError(f'Empty evaluation split: {key}')
    manifest = {'args':vars(a),'slots':slots,'scripts':script_names,'embedding_dim':256,
        'architecture':a.arch,
        'regularization':('joint affine + latent gain/noise + STD dropout + dropout.2 + cosine CE label_smoothing.05; source-balanced anchors'
                          if a.arch=='local_stats' else 'legacy'),
        'explicit_same_char_hard_nce_weight':a.hard_weight,
        'selection':('ink/width-matched val cross-character triplet 2AFC, then prototype macro Top1'
                      if a.hard_weight>0 else 'prototype macro Top1, then cross-character mAP'),
        'loss':'CE(all real+STD) + 0.3 SupCon(real full batch, same-script negatives, different-char positives) + 0.1 matched-STD triplet(margin=.2)',
        'init':'fresh; old critic never used to initialize training',
        'data_sha256':sha(a.data),'csv_sha256':sha(a.csv),'vae_replay':replay,
        'latent_contract':'SD-EMA posterior.mode()*0.18215; shift=0',
        'std_contract':'ink=1 background=0,64x64,sigma=.65; encoder internally converts to legacy white=1',
        'split_sizes':{k:len(v) for k,v in split.items()},
        'eval_sizes':{k:len(v) for k,v in eval_indices.items()},
        'duplicate_train_rows_excluded':old_train_n-len(split['train']),
        'limitations':['work metadata is a SOURCE DIRECTORY PROXY, NOT verified artwork IDs',
                        'old critic historical probe may have seen held-out characters and sources',
                        'prototypes use training split only; test sets never used for checkpoint selection'],
        'source_sha256':{p:sha(p) for p in ['tools/train_prototype_style_encoder.py','src/model/style_encoder.py','src/utils/style_encoder_data.py','src/loss/style_encoder_loss.py']}}
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    np.savez_compressed(out/'split_indices.npz', **split, gallery=gallery, **{'eval_'+k:v for k,v in eval_indices.items()})
    print('PREFLIGHT',json.dumps(manifest,ensure_ascii=False),flush=True)
    slot_scripts = np.array([s2i[k.rsplit('·',1)[1]] for k in real_slots])
    val_pairs=None
    if a.hard_weight>0:
        if statistics is None:
            raise ValueError('Explicit hard metric training requires matching statistics')
        from tools.analysis.accept_teacher_matched import matched_pairs,score_pairs
        val_pairs,_=matched_pairs(eval_indices['val_char'],gallery,labels,si,chars,sources,statistics)
        if len(val_pairs)<50:
            raise ValueError('Insufficient matched validation pairs for checkpoint selection')
        print('MATCHED_VALIDATION_PAIRS',len(val_pairs),flush=True)

    def batch(indices):
        indices = np.asarray(indices)
        return (z[indices].to(device).float(), g[std_rows[indices]].to(device),
                torch.as_tensor(si[indices],device=device))

    @torch.no_grad()
    def embed(model, indices, old=False):
        features, predictions = [], []
        model.eval()
        for start in range(0,len(indices),256):
            zz, gg, ss = batch(indices[start:start+256])
            with amp():
                value = model(zz, 1-gg if old else gg, ss)
            features.append(F.normalize(value['feat'].float(),dim=-1).cpu().numpy() if old else value['embedding'].cpu().numpy())
            predictions.append(value['slot_logits'].float().argmax(1).cpu().numpy())
        return np.concatenate(features), np.concatenate(predictions)

    @torch.no_grad()
    def evaluate(model, key, old=False):
        gf, _ = embed(model, gallery, old)
        proto = source_balanced_prototypes(gf,labels[gallery],sources[gallery],len(real_slots))
        indices = eval_indices[key]
        qf, pred = embed(model, indices, old)
        m = retrieval_metrics(qf,gf,labels[indices],labels[gallery],si[indices],si[gallery],chars[indices],chars[gallery],proto,slot_scripts)
        if old:
            mapped = np.array([k2i[model._historical_slots[int(p)]] for p in pred])
        else:
            mapped = pred
        m['slot_ce_top1'] = float((mapped==labels[indices]).mean())
        if val_pairs is not None and key=='val_char':
            all_features=np.zeros((len(z),256),np.float32)
            all_features[gallery]=gf;all_features[indices]=qf
            matched=score_pairs(all_features,val_pairs)
            m['matched_crosschar_2afc']=matched['accuracy']
            m['matched_crosschar_n']=matched['n']
        return m, proto

    # Historical representation audit is performed BEFORE training, without weight reuse.
    from tools.train_writer_brush_critic import WriterBrushCritic
    historical = torch.load('assets/writer_brush_critic.pt',map_location='cpu',weights_only=False)
    assert list(historical['scripts']) == script_names
    old_model = WriterBrushCritic(len(historical['slots']),len(script_names)).to(device)
    old_model.load_state_dict(historical['model'],strict=True)
    old_model._historical_slots = historical['slots']
    baseline = {key:evaluate(old_model,key,True)[0] for key in eval_indices}
    (out/'historical_critic_probe.json').write_text(json.dumps(baseline,ensure_ascii=False,indent=2))
    print('HISTORICAL_CRITIC',json.dumps(baseline,ensure_ascii=False),flush=True)
    del old_model, historical
    torch.cuda.empty_cache()
    model = build_style_encoder(len(slots),len(script_names),a.arch).to(device)
    optimizer = torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,lambda n:min(1.,(n+1)/200)*(.1+.9*(1+math.cos(math.pi*n/a.steps))/2))
    best = (-1.,-1.)
    started = time.monotonic()
    history = open(out/'metrics.jsonl','a',encoding='utf-8',buffering=1)
    n = a.batch//4
    for step in range(1,a.steps+1):
        model.train()
        indices, sampling = sampler.sample(n)
        zz, gg, ss = batch(indices)
        anchors = indices[:n]
        zi = std_rows[anchors]
        zz = torch.cat([zz,zs[zi].to(device).float()])
        gg = torch.cat([gg,g[zi].to(device)])
        ss = torch.cat([ss,torch.as_tensor(si[anchors],device=device)])
        target = torch.as_tensor(np.r_[labels[indices], len(real_slots)+si[anchors]],device=device)
        if a.arch=='local_stats':
            zz,gg=augment_style_inputs(zz,gg)
        optimizer.zero_grad(set_to_none=True)
        with amp():
            value = model(zz,gg,ss)
        e = value['embedding']
        ce = F.cross_entropy(value['slot_logits'].float(),target,label_smoothing=.05 if a.arch=='local_stats' else 0.)
        con, valid = cross_character_supcon(e[:3*n],target[:3*n],ss[:3*n],torch.as_tensor(char_ids[indices],device=device))
        neutral = std_separation(e[:n],e[n:2*n],e[3*n:])
        hard=ce.new_zeros(())
        if a.hard_weight>0:
            hard_valid=torch.as_tensor(chars[indices[:n]]==chars[indices[2*n:3*n]],device=device)
            hard=same_character_hard_nce(e[:n],e[n:2*n],e[2*n:3*n],hard_valid)
        loss = ce+.3*con+.1*neutral+a.hard_weight*hard
        if not torch.isfinite(loss):
            raise FloatingPointError(f'Nonfinite loss at step {step}')
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
        optimizer.step(); scheduler.step()
        if step==1 or step%100==0:
            seconds = time.monotonic()-started
            record = {'step':step,'loss':loss.item(),'ce':ce.item(),'supcon':con.item(),'std':neutral.item(),
                'hard_nce':hard.item(),'supcon_valid_fraction':valid.item(),'grad_norm':grad.item(),**sampling,
                'steps_per_second':step/seconds,'eta_seconds':seconds/step*(a.steps-step),
                'peak_gib':torch.cuda.max_memory_allocated()/1024**3 if device.type=='cuda' else 0}
            history.write(json.dumps(record)+'\n'); print('TRAIN',json.dumps(record),flush=True)
        if step%a.eval_every==0 or step==a.steps:
            metrics, prototypes = evaluate(model,'val_char')
            record = {'step':step,'validation':metrics}
            history.write(json.dumps(record)+'\n'); print('VALIDATION',json.dumps(record),flush=True)
            state = {'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
                'step':step,'slots':slots,'scripts':script_names,'args':vars(a),'validation':metrics,
                'architecture':a.arch,
                'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if device.type=='cuda' else [],
                'sampler_rng':sampler.rng.bit_generator.state,'manifest_sha256':sha(out/'manifest.json')}
            torch.save(state,out/'last.pt')
            score = ((metrics['matched_crosschar_2afc'],metrics['prototype_macro_top1']) if val_pairs is not None
                     else (metrics['prototype_macro_top1'],metrics['cross_char_map']))
            if score > best:
                best = score
                torch.save(state,out/'best.pt')
            if step in (1000,5000,10000,20000):
                torch.save(state,out/f'step{step:07d}.pt')
    state = torch.load(out/'best.pt',map_location='cpu',weights_only=False)
    model.load_state_dict(state['model'],strict=True)
    final = {key:evaluate(model,key)[0] for key in eval_indices}
    # Export prototypes from ALL train-only references, not merely capped gallery.
    features,_ = embed(model,split['train'])
    prototypes = source_balanced_prototypes(features,labels[split['train']],sources[split['train']],len(real_slots))
    torch.save({'prototypes':torch.from_numpy(prototypes),'slots':real_slots,'scripts':script_names,
                'encoder_checkpoint':'best.pt','encoder_step':state['step'],
                'aggregation':'normalize per source-directory mean; normalize equal-source mean',
                'reference_indices':split['train'],'manifest_sha256':sha(out/'manifest.json')},out/'prototypes.pt')
    (out/'final_results.json').write_text(json.dumps({'selected_step':state['step'],'metrics':final,
            'historical_critic':baseline,'limitations':manifest['limitations']},ensure_ascii=False,indent=2))
    history.close()
    print('ENCODER_TRAINING_COMPLETE',json.dumps(final),flush=True)


if __name__=='__main__':
    main()
