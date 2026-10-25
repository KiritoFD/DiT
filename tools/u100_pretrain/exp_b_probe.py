"""Probe what the writer_slot checkpoints actually learned — no training.

Representations per image: raw DINO CLS (768D, cached bank), z = StyleHead(f)
(128D), h = bridge(z) (32D — the vector the frozen generator actually consumed).
Tests (all leave-char-out where a centroid is involved):
  T1 cross-char writer retrieval: nearest writer centroid (computed WITHOUT the
     target char) identifies the writer; macro accuracy.
  T2 same-char hard negative 2AFC: is the nearest same-writer-other-char point
     closer than the nearest different-writer-SAME-char point?
  T3 held-out feature prediction: ridge from STD patch PCA-128 to the writer
     pair difference delta(c) in each space, LOO chars; compared against the
     pair-mean baseline B0. Question: is the learned space's delta more
     predictable than raw DINO's?
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import sys
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
TOP10_RAW = {601: 0, 703: 1, 479: 2, 956: 3, 9001: 4, 760: 5, 672: 6, 365: 7, 461: 8, 53: 9}


def head_bridge(ckpt_path):
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state = {k.removeprefix('_orig_mod.'): v for k, v in (ck.get('ema') or ck['delta']).items()}
    hw = {k[len('reference_head.'):]: v for k, v in state.items() if k.startswith('reference_head.')}
    bw = {k[len('reference_bridge.'):]: v for k, v in state.items() if k.startswith('reference_bridge.')}
    assert hw and bw, 'no adapter weights in checkpoint'
    head = torch.nn.Sequential(torch.nn.Linear(768, 256), torch.nn.SiLU(), torch.nn.Linear(256, 128))
    head.load_state_dict({k.removeprefix('net.'): v for k, v in hw.items()})
    bridge = torch.nn.Linear(128, 32)
    bridge.load_state_dict(bw)
    return head.eval(), bridge.eval(), int(ck['train_steps'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', default='assets/u100_style48_20261024/runs')
    parser.add_argument('--bank', default='assets/u100_pretrain_20261024/style_bank_top10.npz')
    parser.add_argument('--std-root', default='assets/generic48_kxl_top10_20261022/std_bbox64')
    parser.add_argument('--images-csv', default='assets/generic48_kxl_top10_20261022/top10_train.csv')
    parser.add_argument('--out', default='assets/u100_pretrain_20261024/writer_slot_probe.json')
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists(): raise FileExistsError(out)
    d = np.load(args.bank)
    F = d['features']; W = d['writers']; C = d['chars']
    steps_list = sorted(int(p.name.split('.')[0]) for p in Path(args.run).glob('*/checkpoints/*.pt'))
    results = {}

    def loo_retrieval(V, Wv, Cv):
        pts = defaultdict(list)
        for i in range(len(V)): pts[(int(Wv[i]), int(Cv[i]))].append(V[i])
        pm = {k: np.mean(v, 0) for k, v in pts.items()}
        by_char = defaultdict(set)
        for (w, c) in pm: by_char[c].add(w)
        by_writer = defaultdict(list)
        for (w, c) in pm: by_writer[w].append(c)
        per_writer = {}
        for w in range(10):
            cs = [c for c in by_writer[w] if len(by_char[c]) >= 2 and len(by_writer[w]) >= 3]
            if not cs: continue
            hit = 0
            for c in cs:
                q = pm[(w, c)]
                cents = {}
                for w2 in range(10):
                    oc = [x for x in by_writer[w2] if x != c]
                    if oc: cents[w2] = np.mean([pm[(w2, x)] for x in oc], 0)
                best = max(cents, key=lambda w2: float(q @ cents[w2] / (np.linalg.norm(q) * np.linalg.norm(cents[w2]) + 1e-9)))
                hit += int(best == w)
            per_writer[w] = round(hit / len(cs), 4)
        macro = float(np.mean(list(per_writer.values())))
        return dict(macro_top1=round(macro, 4), per_writer=per_writer)

    def afc(V, Wv, Cv):
        pts = defaultdict(list)
        for i in range(len(V)): pts[(int(Wv[i]), int(Cv[i]))].append(V[i])
        pm = {k: np.mean(v, 0) for k, v in pts.items()}
        by_char = defaultdict(set); by_writer = defaultdict(list)
        for (w, c) in pm: by_char[c].add(w); by_writer[w].append(c)
        win = 0; n = 0
        for (w, c), q in pm.items():
            sw = [pm[(w, c2)] for c2 in by_writer[w] if c2 != c]
            dw = [pm[(w2, c)] for w2 in by_char[c] if w2 != w]
            if not sw or not dw: continue
            dsw = min(float(np.linalg.norm(q - s)) for s in sw)
            ddw = min(float(np.linalg.norm(q - s)) for s in dw)
            win += int(dsw < ddw); n += 1
        return dict(win_rate=round(win / n, 4), n=n)

    for step in steps_list:
        print('PROBE step', step, flush=True)
        ckpt = next(Path(args.run).glob(f'*/checkpoints/{step:07d}.pt'))
        head, bridge, _ = head_bridge(ckpt)
        with torch.no_grad():
            z = torch.nn.functional.normalize(head(torch.from_numpy(F)).float(), dim=-1).numpy()
            h = torch.nn.functional.normalize(bridge(torch.from_numpy(z)).float(), dim=-1).numpy()
        results[f'slot{step//1000}k'] = dict(
            t1_retrieval=dict(dino=loo_retrieval(F, W, C), z128=loo_retrieval(z, W, C), h32=loo_retrieval(h, W, C)),
            t2_hardneg=dict(dino=afc(F, W, C), z128=afc(z, W, C), h32=afc(h, W, C)))

    # T3: STD patch PCA ridge -> held-out delta, in each space
    rows = [r for r in csv.DictReader(open(args.images_csv, encoding='utf-8-sig')) if r['generic'] == '0']
    wc = defaultdict(list)
    for r in rows: wc[(TOP10_RAW[int(r['calligrapher_id'])], ord(r['character']))].append(r['image_path'])
    panel = sorted(c for c in {c for (w, c) in wc} if all(len(wc.get((w, c), [])) >= 2 for w in range(6)))
    std_rows = {(int(r['script_id']), r['character']): r['path']
                for r in csv.DictReader(open(Path(args.std_root) / 'index.csv', encoding='utf-8-sig'))}
    from transformers import AutoModel
    model = AutoModel.from_pretrained('facebook/dinov2-base', local_files_only=True).cuda().eval().requires_grad_(False)
    mean = torch.tensor([.485, .456, .406])[None, :, None, None].cuda()
    std = torch.tensor([.229, .224, .225])[None, :, None, None].cuda()

    def dino_patch(imgs):
        outs = []
        with torch.inference_mode():
            for s in range(0, len(imgs), 32):
                x = torch.from_numpy(np.stack(imgs[s:s + 32])).cuda().float() / 255
                x = torch.nn.functional.interpolate(x, (224, 224), mode='bicubic', align_corners=False, antialias=True).clamp_(0, 1).expand(-1, 3, -1, -1)
                x = (x - mean) / std
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    t = model(pixel_values=x).last_hidden_state[:, 1:]
                outs.append(t.float().cpu().numpy())
        return np.concatenate(outs)
    std_imgs = []
    for c in panel:
        p = std_rows.get((0, chr(c))) or next(v for k, v in std_rows.items() if k[1] == chr(c))
        std_imgs.append(np.asarray(Image.open(Path(args.std_root) / p).convert('L').resize((224, 224), Image.BICUBIC))[None])
    Sp = dino_patch(std_imgs).reshape(len(panel), -1)
    Xc = Sp - Sp.mean(0)
    Gx = Xc @ Xc.T
    evals, evecs = np.linalg.eigh(Gx)
    order = np.argsort(-evals)[:128]
    Xp = (evecs[:, order] * np.sqrt(np.clip(evals[order], 0, None))[None, :])
    Xp /= (np.linalg.norm(Xp, axis=0, keepdims=True) + 1e-9)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=1))   # persist T1/T2 before T3
    panel_idx = {c: i for i, c in enumerate(panel)}
    bank_pos = {int(v): i for i, v in enumerate(d['ids'])}
    pid_of = {}
    for r in rows:
        k = (TOP10_RAW[int(r['calligrapher_id'])], ord(r['character']))
        pid_of.setdefault(k, int(r['img_id']))
    t3 = {}
    for step in steps_list:
        print('T3 step', step, flush=True)
        head, bridge, _ = head_bridge(next(Path(args.run).glob(f'*/checkpoints/{step:07d}.pt')))
        with torch.no_grad():
            z = torch.nn.functional.normalize(head(torch.from_numpy(F)).float(), dim=-1).numpy()
        spaces = dict(dino=F, z128=z)
        per = []
        for A in range(6):
            for B in range(A + 1, 6):
                for space, V in spaces.items():
                    Y = np.stack([V[bank_pos[pid_of[(A, c)]]] - V[bank_pos[pid_of[(B, c)]]] for c in panel])
                    cos0, cos2, r0, r2 = [], [], [], []
                    for j in range(len(panel)):
                        others = [i for i in range(len(panel)) if i != j]
                        b0 = Y[others].mean(0)
                        coef = np.linalg.solve(Xp[others].T @ Xp[others] + np.eye(128), Xp[others].T @ (Y[others] - b0))
                        b2 = b0 + Xp[j] @ coef
                        tgt = Y[j]
                        for b, cs in ((b0, cos0), (b2, cos2)):
                            cs.append(float(b @ tgt / (np.linalg.norm(b) * np.linalg.norm(tgt) + 1e-9)))
                        r0.append(float(((b0 - tgt) ** 2).mean() / (Y ** 2).mean()))
                        r2.append(float(((b2 - tgt) ** 2).mean() / (Y ** 2).mean()))
                per.append(dict(pair=f'{A}-{B}', cos_B0=round(float(np.mean(cos0)), 4), cos_B2=round(float(np.mean(cos2)), 4),
                                relmse_B0=round(float(np.mean(r0)), 4), relmse_B2=round(float(np.mean(r2)), 4)))
        t3[f'slot{step//1000}k'] = dict(
            z128=dict(cos_B0=round(float(np.mean([p['cos_B0'] for p in per])), 4),
                      cos_B2=round(float(np.mean([p['cos_B2'] for p in per])), 4),
                      relmse_B0=round(float(np.mean([p['relmse_B0'] for p in per])), 4),
                      relmse_B2=round(float(np.mean([p['relmse_B2'] for p in per])), 4)))
    results['t3_feature_prediction'] = t3
    out.write_text(json.dumps(results, indent=1))   # final overwrite with T3
    print('PROBE_DONE', json.dumps(results, ensure_ascii=False)[:2400])


if __name__ == '__main__':
    main()
