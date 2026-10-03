#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Frozen-DINO + ranking head (RankNet). Upgrade style judge AUC 0.606 -> target 0.75+.

Input : assets/dino_meanstd_50k.npy (51036,768) = DINO patch mean+std, frozen.
Loss  : pairwise ranking, POS=same-cal-diff-char, NEG=same-char-diff-cal (anti-shortcut).
Eval  : held-out same-char-pool enrichment (ref 3.36x) + pairwise AUC (ref 0.606).
"""
import argparse, csv, glob, os, re, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

os.chdir('/root/Workspace/xy/DiT')
ap = argparse.ArgumentParser()
ap.add_argument('--steps', type=int, default=4000)
ap.add_argument('--batch', type=int, default=1024)
ap.add_argument('--lr', type=float, default=1e-3)
ap.add_argument('--kneg', type=int, default=8)
ap.add_argument('--eval-every', type=int, default=1000)
ap.add_argument('--out', type=str, default='assets/style_rank_dino.pt')
a = ap.parse_args()
DEV = 'cuda'
torch.manual_seed(0); np.random.seed(0)


def rid(p):
    m = re.search(r'(\d+)\.png$', str(p))
    return int(m.group(1)) if m else None


print('[1] load DINO mean+std ...', flush=True)
X = np.load('assets/dino_meanstd_50k.npy')
ids = np.load('data/dino_cache/50k_v1/ids.npy')
pos = {int(i): k for k, i in enumerate(ids)}
tr = list(csv.DictReader(open('assets/train_50k_v2_fixed.csv', encoding='utf-8')))
CALLS = sorted({r.get('calligrapher') for r in tr})
C2I = {c: i for i, c in enumerate(CALLS)}
K, CAL, CH = [], [], []
for r in tr:
    i = rid(r.get('image_path', ''))
    k = pos.get(i) if i is not None else None
    if k is None:
        continue
    K.append(k); CAL.append(C2I[r.get('calligrapher')]); CH.append(str(r.get('character', '')))
K, CAL, CH = np.array(K), np.array(CAL), np.array(CH)
rng = np.random.RandomState(0)
chars = np.array(sorted(set(CH))); rng.shuffle(chars)
test_chars = set(chars[:int(len(chars) * 0.15)].tolist())
m_char = np.array([c in test_chars for c in CH])
ridx = rng.permutation(len(K)); m_rand = np.zeros(len(K), bool)
m_rand[ridx[:int(len(K) * 0.1)]] = True
tr_final = (~m_char) & (~m_rand)
print('[1] char-test %d / rand-test %d / train %d' % (m_char.sum(), m_rand.sum(), tr_final.sum()), flush=True)
TRK, TRC, TRH = K[tr_final], CAL[tr_final], CH[tr_final]
XT = torch.from_numpy(X[TRK]).to(DEV)
by_cal, by_char = {}, {}
for j in range(len(TRC)):
    by_cal.setdefault(int(TRC[j]), []).append(j)
    by_char.setdefault(TRH[j], []).append(j)
POS, NEG = [], []
for j in range(len(TRC)):
    v = [x for x in by_cal.get(int(TRC[j]), []) if TRH[x] != TRH[j]]
    POS.append(np.array(v) if v else np.array([j]))
    v = [x for x in by_char.get(TRH[j], []) if int(TRC[x]) != int(TRC[j])]
    NEG.append(np.array(v) if v else np.array([j]))
print('[1] pairs: pos-avg %.0f / neg-avg %.1f' % (np.mean([len(p) for p in POS]), np.mean([len(n) for n in NEG])), flush=True)


class Ranker(nn.Module):
    """DINO 768 -> 128 normalized embedding."""
    def __init__(self, din=768, d=128, p=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(din, 512), nn.GELU(), nn.Dropout(p),
            nn.Linear(512, 256), nn.GELU(),
            nn.Linear(256, d))
    def forward(self, x): return F.normalize(self.net(x), dim=-1)


model = Ranker().to(DEV)
print('[1] params %s' % format(sum(p.numel() for p in model.parameters()), ','), flush=True)
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.steps)

EVAL_I = {'seen': K[m_rand & (~m_char)], 'unseen': K[m_char]}
EVAL_C = {'seen': CAL[m_rand & (~m_char)], 'unseen': CAL[m_char]}
EVAL_H = CH[m_rand & (~m_char)]


@torch.no_grad()
def emb_of(ii):
    out = []
    for s in range(0, len(ii), 4096):
        x = torch.from_numpy(X[ii[s:s + 4096]]).to(DEV)
        out.append(model(x).float().cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def evaluate(model, tag):
    print('  ---- eval %s ----' % tag, flush=True)
    pool = {}
    for j in range(len(TRC)): pool.setdefault(TRH[j], []).append(int(TRK[j]))
    pool = {k: np.array(v[:12]) for k, v in pool.items() if len(v) >= 2}
    CAL_OF = {int(TRK[j]): int(TRC[j]) for j in range(len(TRC))}
    ii = EVAL_I['seen']; Zq = emb_of(ii)
    ycl = EVAL_C['seen']; ych = EVAL_H
    Zp = {k: emb_of(c) for k, c in pool.items()}
    hit = tot = 0; base = 0.0
    for k in range(len(ii)):
        if ych[k] not in pool: continue
        cand = pool[ych[k]]; Zc = Zp[ych[k]]
        best = int(np.argmax(Zc @ Zq[k]))
        hit += int(CAL_OF[int(cand[best])] == int(ycl[k]))
        base += sum(1 for c in cand if CAL_OF[int(c)] == int(ycl[k])) / len(cand)
        tot += 1
    enr = (hit / tot) / (base / tot)
    print('  heldout-seen  enrich %.2fx (hit %.1f%% base %.1f%%, n=%d)'
          % (enr, hit / tot * 100, base / tot * 100, tot), flush=True)
    out_auc = {}
    for name in EVAL_I:
        ii2 = EVAL_I[name]; Z2 = emb_of(ii2); ys = EVAL_C[name]
        sub = np.random.RandomState(1).choice(len(ii2), min(3000, len(ii2)), replace=False)
        Zs = Z2[sub]; y2 = ys[sub]
        S = Zs @ Zs.T; iu = np.triu_indices(len(sub), 1)
        sims = S[iu]; lab = (y2[iu[0]] == y2[iu[1]])
        order = np.argsort(sims); ranks = np.empty(len(sims)); ranks[order] = np.arange(1, len(sims) + 1)
        npos = lab.sum(); nneg = (~lab).sum()
        auc = (ranks[lab].sum() - npos * (npos + 1) / 2) / (npos * nneg)
        out_auc[name] = auc
        print('  heldout-%s  AUC %.3f (n=%d)' % (name, auc, len(ii2)), flush=True)
    return out_auc['unseen'], enr


print('\n[2] train %d steps (batch %d, ranking) ...' % (a.steps, a.batch), flush=True)
t0 = time.time()
for step in range(a.steps):
    bi = np.random.randint(0, len(TRK), a.batch)
    pj = np.array([POS[j][np.random.randint(len(POS[j]))] for j in bi])
    nj = np.stack([NEG[j][np.random.choice(len(NEG[j]), a.kneg,
                                           replace=len(NEG[j]) < a.kneg)] for j in bi])
    zq = model(XT[bi])
    zp = model(XT[pj])
    zn = model(XT[nj.reshape(-1)]).view(len(bi), a.kneg, -1)
    s_pos = (zq * zp).sum(-1)
    s_neg = torch.bmm(zn, zq.unsqueeze(-1)).squeeze(-1)
    loss = F.softplus(s_neg - s_pos.unsqueeze(-1)).mean()
    loss = loss + (torch.logsumexp(torch.cat([s_pos.unsqueeze(-1), s_neg], 1) / 0.1, 1) - s_pos / 0.1).mean()
    opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
    if (step + 1) % 500 == 0:
        print('  step %5d loss %.4f %.0fs' % (step + 1, float(loss), time.time() - t0), flush=True)
    if (step + 1) % a.eval_every == 0 and (step + 1) < a.steps:
        model.eval(); evaluate(model, '@%d' % (step + 1)); model.train()

model.eval()
auc_u, enr = evaluate(model, 'final')
torch.save(model.state_dict(), a.out)
print('\nsaved %s' % a.out, flush=True)
print('\n===== verdict (ref: latent 2.9M AUC 0.606 / enrich 3.13x; DINO train-free enrich 2.24x) =====')
print('  AUC(unseen-char) %.3f   enrich %.2fx' % (auc_u, enr))
print('  >=0.75 -> qualified as diffusion style-loss; 0.65~0.75 -> eval-probe only; <0.65 -> ranking failed')
