#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""style_encoder_latent 的小模型/大 batch 扫描版（自包含）。

网格: ch ∈ {16, 32} × batch ∈ {1024, 2048}，各 2500 步，LR 随 batch 线性放大。
每个配置训完立即评测（留出 top1/top5 + 同字池富集 + pairwise AUC），分别存 ckpt。
参照: 2.9M 大模型 (ch96 b512) => 字没见过 23.3% / 富集 3.13x / AUC 0.606。
"""
import argparse, csv, glob, os, re, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

os.chdir('/root/Workspace/xy/DiT')
ap = argparse.ArgumentParser()
ap.add_argument('--steps', type=int, default=2500)
ap.add_argument('--eval-every', type=int, default=1250)
ap.add_argument('--kneg', type=int, default=8)
a = ap.parse_args()
DEV = 'cuda'
torch.manual_seed(0); np.random.seed(0)

def rid(p):
    m = re.search(r'(\d+)\.png$', str(p))
    return int(m.group(1)) if m else None

print('[1] 载入 latent ...', flush=True)
NMAX = 52457
arr = np.zeros((NMAX, 4, 32, 32), np.float16); has = np.zeros(NMAX, bool)
for f in sorted(glob.glob('data/50k/shards_img/*.npz')):
    z = np.load(f); arr[z['img_ids']] = z['latents']; has[z['img_ids']] = True
tr = list(csv.DictReader(open('assets/train_50k_v2_fixed.csv', encoding='utf-8')))
CALLS = sorted({r.get('calligrapher') for r in tr}); C2I = {c: i for i, c in enumerate(CALLS)}
IDX, CAL, CH = [], [], []
for r in tr:
    i = rid(r.get('image_path', ''))
    if i is None or not has[i]: continue
    IDX.append(i); CAL.append(C2I[r.get('calligrapher')]); CH.append(str(r.get('character', '')))
IDX, CAL, CH = np.array(IDX), np.array(CAL), np.array(CH)
rng = np.random.RandomState(0)
chars = np.array(sorted(set(CH))); rng.shuffle(chars)
test_chars = set(chars[:int(len(chars) * 0.15)].tolist())
m_char = np.array([c in test_chars for c in CH])
ridx = rng.permutation(len(IDX)); m_rand = np.zeros(len(IDX), bool)
m_rand[ridx[:int(len(IDX) * 0.1)]] = True
tr_final = (~m_char) & (~m_rand)
class Enc(nn.Module):
    def __init__(self, nc=45, d=256, ch=32):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(4, ch, 3, 1, 1), nn.GELU(), nn.GroupNorm(min(8, ch), ch),
            nn.Conv2d(ch, ch * 2, 4, 2, 1), nn.GELU(), nn.GroupNorm(min(8, ch * 2), ch * 2),
            nn.Conv2d(ch * 2, ch * 4, 4, 2, 1), nn.GELU(), nn.GroupNorm(min(16, ch * 4), ch * 4),
            nn.Conv2d(ch * 4, ch * 4, 3, 1, 1), nn.GELU(), nn.GroupNorm(min(16, ch * 4), ch * 4),
            nn.AdaptiveAvgPool2d(1))
        self.head = nn.Linear(ch * 4, 256); self.clf = nn.Linear(256, nc)
    def emb(self, x): return F.normalize(self.head(self.body(x).flatten(1)), dim=-1)
    def forward(self, x):
        z = self.emb(x); return z, self.clf(z)


TRI, TRC, TRH = IDX[tr_final], CAL[tr_final], CH[tr_final]
TRX_all = torch.from_numpy(arr[TRI].astype(np.float32)).to(DEV)
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
print('[1] 配对表: 正均 %.0f / 负均 %.1f' % (np.mean([len(p) for p in POS]), np.mean([len(n) for n in NEG])), flush=True)

EVAL_I = {'字见过': IDX[m_rand & (~m_char)], '字没见过': IDX[m_char]}
EVAL_C = {'字见过': CAL[m_rand & (~m_char)], '字没见过': CAL[m_char]}


@torch.no_grad()
def emb_of(model, ii):
    out = []
    for s in range(0, len(ii), 4096):
        x = torch.from_numpy(arr[ii[s:s + 4096]].astype(np.float32)).to(DEV)
        out.append(model.emb(x).float().cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def evaluate(model, tag):
    print('  ---- eval %s ----' % tag, flush=True)
    t1_unseen = None
    for name in EVAL_I:
        Z = emb_of(model, EVAL_I[name])
        lg = model.clf(torch.from_numpy(Z).to(DEV)).float().cpu().numpy()
        o = np.argsort(-lg, 1); yy = EVAL_C[name]
        t1 = (o[:, 0] == yy).mean(); t5 = (o[:, :5] == yy[:, None]).any(1).mean()
        if name == '字没见过':
            t1_unseen = t1
        print('  留出·%s  top1 %5.2f%%  top5 %5.2f%% (n=%d)' % (name, t1 * 100, t5 * 100, len(Z)), flush=True)
    pool = {}
    for j in range(len(TRC)): pool.setdefault(TRH[j], []).append(int(TRI[j]))
    pool = {k: np.array(v[:12]) for k, v in pool.items() if len(v) >= 2}
    CAL_OF = {int(TRI[j]): int(TRC[j]) for j in range(len(TRC))}
    Zp = {k: emb_of(model, c) for k, c in pool.items()}
    ii = EVAL_I['字见过']; Zq = emb_of(model, ii)
    ycl, ych = CAL[m_rand & (~m_char)], CH[m_rand & (~m_char)]
    hit = tot = 0; base = 0.0
    for k in range(len(ii)):
        if ych[k] not in Zp: continue
        cand = pool[ych[k]]; Zc = Zp[ych[k]]
        best = int(np.argmax(Zc @ Zq[k]))
        hit += int(CAL_OF[int(cand[best])] == int(ycl[k]))
        base += sum(1 for c in cand if CAL_OF[int(c)] == int(ycl[k])) / len(cand)
        tot += 1
    enr = (hit / tot) / (base / tot)
    sub = np.random.RandomState(1).choice(len(ii), min(3000, len(ii)), replace=False)
    Zs = Zq[sub]; ys = ycl[sub]
    S = Zs @ Zs.T; iu = np.triu_indices(len(sub), 1)
    sims = S[iu]; lab = (ys[iu[0]] == ys[iu[1]])
    order = np.argsort(sims); ranks = np.empty(len(sims)); ranks[order] = np.arange(1, len(sims) + 1)
    npos = lab.sum(); nneg = (~lab).sum()
    auc = (ranks[lab].sum() - npos * (npos + 1) / 2) / (npos * nneg)
    print('  同字池富集 %.2fx (命中 %.1f%% 基线 %.1f%%)  AUC %.3f' % (enr, hit / tot * 100, base / tot * 100, auc), flush=True)
    return t1_unseen, enr, auc


GRID = [(16, 4096), (32, 3072)]
RESULTS = []
for ch, bs in GRID:
    print('\n===== ch=%d batch=%d =====' % (ch, bs), flush=True)
    torch.manual_seed(0); np.random.seed(0)
    model = Enc(ch=ch).to(DEV)
    nparam = sum(p.numel() for p in model.parameters())
    print('  参数 %s' % format(nparam, ','), flush=True)
    lr = 8e-4 * (bs / 512.0)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.02)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.steps)
    t0 = time.time()
    for step in range(a.steps):
        bi = np.random.randint(0, len(TRX_all), bs)
        x = TRX_all[bi]
        z, lg = model(x)
        loss = F.cross_entropy(lg, torch.from_numpy(TRC[bi]).to(DEV))
        pj = np.array([POS[j][np.random.randint(len(POS[j]))] for j in bi])
        nj = np.stack([NEG[j][np.random.choice(len(NEG[j]), a.kneg,
                                               replace=len(NEG[j]) < a.kneg)] for j in bi])
        zp = model.emb(TRX_all[pj])
        zn = model.emb(TRX_all[nj.reshape(-1)]).view(len(bi), a.kneg, -1)
        s_pos = (z * zp).sum(-1) / 0.1
        s_neg = torch.bmm(zn, z.unsqueeze(-1)).squeeze(-1) / 0.1
        loss = loss + (torch.logsumexp(torch.cat([s_pos.unsqueeze(-1), s_neg], 1), 1) - s_pos).mean()
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
        if (step + 1) % 500 == 0:
            print('  step %5d loss %.4f %.0fs' % (step + 1, float(loss), time.time() - t0), flush=True)
        if (step + 1) % a.eval_every == 0 and (step + 1) < a.steps:
            model.eval(); evaluate(model, '%d@%d' % (ch, step + 1)); model.train()
    model.eval()
    t1u, enr, auc = evaluate(model, 'ch%d final' % ch)
    out = 'assets/style_enc_s%db%d.pt' % (ch, bs)
    torch.save(model.state_dict(), out)
    RESULTS.append((ch, bs, nparam, t1u, enr, auc, out))
    print('  saved %s' % out, flush=True)

print('\n===== 汇总 (参照: 2.9M ch96 b512 => 字没见过 23.3% / 富集 3.13x / AUC 0.606) =====', flush=True)
for ch, bs, nparam, t1u, enr, auc, out in RESULTS:
    print('  ch=%-3d b=%-5d (%s 参数)  字没见过 %5.2f%%  富集 %.2fx  AUC %.3f  %s'
          % (ch, bs, format(nparam, ','), t1u * 100, enr, auc, out), flush=True)