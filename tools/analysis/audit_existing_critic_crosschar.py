"""Read-only cross-character audit of the existing frozen latent Critic.

Uses the Critic's ORIGINAL white-one STD convention, one STD per glyph.
No model fitting, checkpoint selection, generator training or GT conditioning.
"""
import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.train_writer_brush_critic import WriterBrushCritic
from tools.analysis.accept_teacher_matched import matched_pairs, score_pairs
from tools.train_prototype_style_encoder import retrieval_metrics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ckpt', default='assets/writer_brush_critic.pt')
    p.add_argument('--data', default='scratch/disc_lat_full.npz')
    p.add_argument('--stats', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--holdout-frac', type=float, default=.3)
    a = p.parse_args()
    torch.set_num_threads(4)
    d = np.load(a.data, allow_pickle=True)
    w, s, c, source = [d[k].astype(str) for k in ('writer', 'script', 'char', 'work')]
    works = sorted(set(source)); random.Random(a.seed).shuffle(works)
    hold = set(works[:max(1, int(len(works)*a.holdout_frac))])
    train = np.flatnonzero(~np.isin(source, list(hold)))
    query = np.flatnonzero(np.isin(source, list(hold)))
    ck = torch.load(a.ckpt, map_location='cpu', weights_only=False)
    model = WriterBrushCritic(len(ck['slots']), len(ck['scripts'])).cuda().eval().requires_grad_(False)
    model.load_state_dict(ck['model'], strict=True)
    labels = np.array([ck['slots'].index(x+'·'+y) for x,y in zip(w,s)])
    si = np.array([ck['scripts'].index(x) for x in s])
    gi = d['std_idx'].astype(np.int64)
    assert np.array_equal(d['std_char'][gi].astype(str), c)
    assert np.array_equal(d['std_script'][gi].astype(str), s)
    z = torch.from_numpy(d['z'].copy()).float()
    g = torch.from_numpy(d['g_std'].copy()).float()[:,None]/255
    features = np.empty((len(w),256),np.float32)
    brush = np.empty(len(w),np.float32)
    with torch.no_grad():
        for start in range(0,len(w),256):
            ii = np.arange(start,min(start+256,len(w)))
            o = model(z[ii].cuda(),g[gi[ii]].cuda(),torch.from_numpy(si[ii]).cuda())
            features[ii] = F.normalize(o['feat'],dim=-1).cpu().numpy()
            brush[ii] = o['brush_logit'].cpu().numpy()
    slots = [x for x in ck['slots'] if not x.startswith('std·')]
    proto = np.stack([features[train[labels[train]==k]].mean(0) if np.any(labels[train]==k)
                      else np.zeros(256) for k in range(len(slots))])
    proto /= np.maximum(np.linalg.norm(proto,axis=1,keepdims=True),1e-12)
    statcache = np.load(a.stats)
    digest = hashlib.sha256(Path(a.data).read_bytes()).hexdigest()
    assert str(statcache['data_sha256']) == digest
    stats = statcache['stats']; assert stats.shape == (len(w),2)
    pairs, cross = matched_pairs(query,train,w,s,c,source,stats)
    metrics = retrieval_metrics(features[query],features[train],labels[query],labels[train],
        si[query],si[train],c[query],c[train],proto,
        np.array([ck['scripts'].index(x.rsplit('·',1)[1]) for x in slots]))
    # Correctly paired STD negatives and differentiability; never update weights.
    ii = query[:128]; zz = z[ii].cuda().requires_grad_(True)
    real = model(zz,g[gi[ii]].cuda(),torch.from_numpy(si[ii]).cuda())
    with torch.no_grad():
        std = model(torch.from_numpy(d['z_std'][gi[ii]].copy()).cuda().float(),
                    g[gi[ii]].cuda(),torch.from_numpy(si[ii]).cuda())
    target = torch.from_numpy(proto[labels[ii]]).cuda().float()
    loss = -(F.normalize(real['feat'],dim=-1)*target).sum(-1).mean()
    grad = torch.autograd.grad(loss,zz)[0]
    report = dict(ckpt_sha256=hashlib.sha256(Path(a.ckpt).read_bytes()).hexdigest(),
        data_sha256=digest, historical_results=ck.get('results'),
        protocol='native Critic source holdout reconstructed from original script defaults',
        limitations=['Checkpoint does not save split/seed; reconstructed split is NOT independently verified.',
                     'Sources are directory proxies, not verified artworks.',
                     'No clean character-heldout claim: historical Critic may have trained on those characters.',
                     'Pair-level Wilson intervals reuse references; not independent trials.',
                     'No DINO/CLIP superiority claim without identical-protocol baselines.'],
        seed=a.seed, holdout_frac=a.holdout_frac, n_train=len(train),n_query=len(query),
        retrieval=metrics, matched_crosschar={**score_pairs(features,pairs),'cross_source_positive_n':int(cross)},
        brush_real_positive=float((brush[query]>0).mean()),
        paired_std=dict(n=len(ii), real_brush_above_std=float((real['brush_logit'].detach()>std['brush_logit']).float().mean())),
        gradient=dict(loss=float(loss.detach()),norm=float(grad.norm()),finite=bool(torch.isfinite(grad).all()),
                      frozen_parameter_grads=all(x.grad is None for x in model.parameters())))
    out = Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)


if __name__ == '__main__':
    main()
