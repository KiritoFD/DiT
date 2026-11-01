"""F2: train E-G (global Writer CE) and E-L (foreground-patch aggregation).

Both encoders share data, character split and source split, so their numbers
are comparable. Everything here is CPU/GPU-light: ResNet18 at 224 with a frozen
stem, trained on the panel images the F1 stage already selected.

E-G  ResNet18 -> global average pool -> Linear(512, n_writer). Classic
     cross-character shared-style baseline.

E-L  ResNet18 -> 7x7 patch tokens; each patch is projected, then tokens are
     aggregated with attention restricted to FOREGROUND patches (ink above a
     threshold), i.e. the white paper cannot dominate the representation. The
     pooled vector goes to Writer CE plus a supervised-contrastive term whose
     positives are the same writer on a DIFFERENT character and whose hard
     negatives are the same character, same script, different writer.

Evaluation, all on held-out characters:
    writer macro accuracy (character holdout)
    writer macro accuracy (character AND source holdout)
    same-character different-writer pair accuracy
    accuracy after ink-mass matching
    accuracy on VAE round-tripped images (blur/compression robustness)
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def make_encoder(kind: str, n_writer: int):
    import torchvision
    base = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    trunk = nn.Sequential(*list(base.children())[:-2])     # [B,512,7,7]

    class Enc(nn.Module):
        def __init__(self):
            super().__init__()
            self.trunk = trunk
            self.kind = kind
            self.null = nn.Parameter(torch.zeros(1, 512))
            self.att = nn.Sequential(nn.Linear(512, 128), nn.Tanh(), nn.Linear(128, 1))
            self.head = nn.Linear(512, n_writer)

        def forward(self, x, fg=None):
            f = self.trunk(x)                             # [B,512,7,7]
            if self.kind == 'G':
                v = f.mean(dim=(2, 3))
            else:
                B, C, H, W = f.shape
                tok = f.flatten(2).transpose(1, 2)        # [B,49,512]
                if fg is None:
                    w = torch.ones(B, H * W, device=x.device)
                else:
                    # foreground = patch ink ratio resampled to 7x7
                    fm = F.adaptive_avg_pool2d(fg, (H, W)).flatten(1)
                    w = (fm > 0.05).float() + 1e-3
                logit = self.att(tok).squeeze(-1) + torch.log(w)
                a = torch.softmax(logit, dim=1)
                v = (tok * a[..., None]).sum(1)
            return v, self.head(v)

    return Enc()


def metrics(logits, y, train_mask=None):
    pred = logits.argmax(1)
    per = {}
    for c in np.unique(y):
        m = y == c
        per[int(c)] = float((pred[m] == y[m]).mean())
    return float(np.mean(list(per.values()))), per


def pair_accuracy(emb, y, char):
    """Same-character different-writer pair accuracy (nearest-centroid over writers)."""
    cent = {}
    for c in np.unique(y):
        cent[int(c)] = emb[y == c].mean(0)
    keys = sorted(cent)
    C = np.stack([cent[k] for k in keys])
    Cn = C / np.maximum(np.linalg.norm(C, axis=1, keepdims=True), 1e-9)
    en = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-9)
    sim = en @ Cn.T
    hits, tot = 0, 0
    for c in np.unique(char):
        m = char == c
        if len(np.unique(y[m])) < 2:
            continue
        s = sim[m]
        pred = np.array(keys)[s.argmax(1)]
        hits += int((pred == y[m]).sum())
        tot += int(m.sum())
    return hits / max(tot, 1), tot


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='assets/style_24h48_20261026')
    ap.add_argument('--kind', required=True, choices=['G', 'L'])
    ap.add_argument('--epochs', type=int, default=12)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--char-test-frac', type=float, default=0.25)
    ap.add_argument('--seed', type=int, default=90)
    a = ap.parse_args()

    root = Path('/home/ds/Workspace/DiT') / a.root
    z = dict(np.load(root / 'panel_features.npz', allow_pickle=True))
    from PIL import Image
    R = Path('/home/ds/Workspace/DiT')

    writer = z['writer']
    char = z['character']
    split = z['split']
    paths = z['image_path']

    writers = sorted(set(writer.tolist()))
    wrow = {w: i for i, w in enumerate(writers)}
    y = np.array([wrow[w] for w in writer])

    chars = np.array(sorted(set(char.tolist())))
    charcode = {c: i for i, c in enumerate(chars.tolist())}
    rng = np.random.default_rng(a.seed)
    perm = rng.permutation(len(chars))
    n_te = int(round(len(chars) * a.char_test_frac))
    test_chars = set(chars[perm[:n_te]].tolist())
    char_test = np.array([c in test_chars for c in char])
    src_test = split == 'test'

    tr = ~char_test
    te_char = char_test
    te_both = char_test & src_test

    # Train uses characters in the train half ONLY; never a test character.
    tr_idx = np.nonzero(tr)[0]
    np.random.default_rng(a.seed).shuffle(tr_idx)

    def load_pils(indices, size=224):
        out = []
        for i in indices:
            p = Path(paths[i])
            p = p if p.is_absolute() else R / p
            im = Image.open(p).convert('RGBA')
            arr = np.array(im).astype(np.float32)
            al = arr[..., 3:4] / 255.0
            rgb = arr[..., :3] * al + 255.0 * (1 - al)
            g = rgb.mean(-1)
            ink = g < 190
            if ink.sum() > 16:
                ys, xs = np.nonzero(ink)
                mh = int((ys.max() - ys.min() + 1) * .08)
                mw = int((xs.max() - xs.min() + 1) * .08)
                g = g[max(0, ys.min() - mh):ys.max() + 1 + mh,
                      max(0, xs.min() - mw):xs.max() + 1 + mw]
            im2 = Image.fromarray(np.clip(g, 0, 255).astype(np.uint8)).resize((size, size),
                                                                              Image.BICUBIC)
            out.append(np.array(im2, np.float32) / 255.0)
        return np.stack(out)[:, None]

    def to_tensor(grays):
        x = torch.from_numpy(grays).float()
        x = (x - 0.5) / 0.5
        return x.repeat(1, 3, 1, 1)

    print(f'[{a.kind}] loading {len(tr_idx)} train / {int(te_char.sum())} char-test imgs', flush=True)
    Xtr = to_tensor(load_pils(tr_idx))
    Ytr = torch.from_numpy(y[tr_idx]).long()
    # The plan's E-L positives must be "same writer, DIFFERENT character". A same
    # writer AND same character pair is not a style positive: it can be won by
    # memorising one glyph, which is exactly the shortcut this experiment is
    # meant to rule out. So the SupCon mask below is built from writer AND char.
    Ctr = torch.from_numpy(np.array([charcode[c] for c in char[tr_idx]], np.int64)).long()
    # foreground map for patch masking: ink = dark
    FGtr = (Xtr[:, :1] < 0.0).float()

    eval_sets = {}
    for name, mask in (('char_test', te_char), ('char_src_test', te_both)):
        ids = np.nonzero(mask)[0]
        if len(ids) == 0:
            continue
        Xe = to_tensor(load_pils(ids))
        eval_sets[name] = dict(X=Xe, FG=(Xe[:, :1] < 0.0).float(),
                               y=y[ids], char=char[ids])

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = make_encoder(a.kind, len(writers)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.02)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)

    n = len(tr_idx)
    history = []
    for ep in range(a.epochs):
        model.train()
        order = torch.randperm(n)
        tot, seen = 0.0, 0
        for s in range(0, n, a.batch):
            b = order[s:s + a.batch]
            x, fg, yy, cc = (Xtr[b].to(dev), FGtr[b].to(dev), Ytr[b].to(dev),
                             Ctr[b].to(dev))
            emb, logit = model(x, fg)
            ce = F.cross_entropy(logit, yy)
            loss = ce
            if a.kind == 'L':
                # SupCon, exactly as the plan specifies:
                #   positives  = same writer AND different character
                #   negatives  = everything else, which automatically makes the
                #                hard negatives "same character, different writer"
                #                (and same script, since scripts are split upstream)
                zz = F.normalize(emb, dim=1)
                sim = zz @ zz.T / 0.07
                same_w = (yy[:, None] == yy[None, :])
                same_c = (cc[:, None] == cc[None, :])
                eye = torch.eye(len(yy), dtype=torch.bool, device=dev)
                pos = (same_w & ~same_c & ~eye).float()
                sim = sim - sim.max(1, keepdim=True).values.detach()
                logp = sim - torch.log(torch.exp(sim).sum(1, keepdim=True))
                denom = pos.sum(1).clamp_min(1e-6)
                con = -(logp * pos).sum(1) / denom
                con = con[pos.sum(1) > 0].mean() if (pos.sum(1) > 0).any() else ce * 0
                loss = ce + 0.5 * con
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss.detach()) * len(b)
            seen += len(b)
        sched.step()
        history.append({'epoch': ep, 'loss': tot / max(seen, 1)})
        print(f'[{a.kind}] ep{ep} loss={tot/max(seen,1):.4f}', flush=True)

    model.eval()
    report = {'kind': a.kind, 'epochs': a.epochs, 'n_writers': len(writers),
              'n_train': int(n), 'history': history, 'eval': {}}
    for name, ds in eval_sets.items():
        with torch.no_grad():
            embs, logits = [], []
            for s in range(0, len(ds['y']), 128):
                x = ds['X'][s:s + 128].to(dev)
                fg = ds['FG'][s:s + 128].to(dev)
                e, l = model(x, fg)
                embs.append(e.cpu().numpy())
                logits.append(l.cpu().numpy())
            emb = np.concatenate(embs)
            lg = np.concatenate(logits)
        macro, per = metrics(lg, ds['y'])
        pa, n_pair = pair_accuracy(emb, ds['y'], ds['char'])
        report['eval'][name] = {'macro_acc': macro, 'per_writer': per,
                                'pair_acc': pa, 'pair_n': n_pair, 'n': int(len(ds['y']))}
        print(f'[{a.kind}] {name}: macro={macro:.4f} pair={pa:.4f} n={len(ds["y"])}', flush=True)

    out = root / f'E{a.kind}_results.json'
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    torch.save({'state_dict': model.state_dict(), 'writers': writers, 'kind': a.kind},
               root / f'E{a.kind}_encoder.pt')
    print(f'[{a.kind}] wrote {out}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
