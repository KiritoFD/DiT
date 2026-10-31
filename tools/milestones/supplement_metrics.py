#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""本地 GPU 为里程碑面板补定量指标（2026-10-31）。

- OCR：读光微调判别器（assets/p125_20261030/ocr/duguang_ft_ext.pt）peak-timestep
  Top-1/Top-5/Margin，字符标签来自 --labels JSON {idx: char} 或 eval_stdskel_batch.csv。
- SSIM：g{i}.png vs gt{i}.png 逐对（灰度、高斯加权本地实现），作视觉参考。

用法:
  python tools/milestones/supplement_metrics.py --dir assets/milestones_samples/S0_85k \
      --labels-json <labels.json> --out assets/milestones_samples/S0_85k/metrics_local.json
"""
import argparse, csv, json, os, re, sys
import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image

sys.path.insert(0, '.')
os.chdir(os.path.join(os.path.dirname(__file__), '..', '..'))


def _num(files, prefix):
    out = []
    for p in files:
        m = re.search(prefix + r'(\d+)\.png$', p)
        if m:
            out.append((int(m.group(1)), p))
    return dict(sorted(out))


def load_ocr():
    from tools.p125.duguang_model import DuguangOCR
    ck = torch.load('assets/p125_20261030/ocr/duguang_ft_ext.pt', map_location='cpu', weights_only=False)
    vocab = [l.strip() for l in open('assets/p125_20261030/ocr/vocab.txt', encoding='utf-8')]
    net = DuguangOCR(vocab_size=len(vocab) + 1)
    net.load_state_dict(ck['model'], strict=False)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    return net.to(dev).eval(), {c: i + 1 for i, c in enumerate(vocab)}, vocab, dev


def ocr_image(net, dev, path):
    img = Image.open(path).convert('L')
    w, h = img.size
    tw = int(np.clip(round(32 * w / h), 16, 320))
    img = img.resize((tw, 32), Image.Resampling.BICUBIC)
    x = np.asarray(img, dtype=np.float32) / 255
    x = (x - 0.5) / 0.5
    with torch.no_grad():
        logits = net(torch.from_numpy(x)[None, None].to(dev))
    conf = logits.max(-1).values
    peak = conf.argmax(-1).item()
    return logits[0, peak].cpu()


def ssim_pair(a_path, b_path):
    """单尺度高斯加权 SSIM（灰度，7x7 窗，K1=.01 K2=.03），视觉参考口径。"""
    def gray(p, side=256):
        im = Image.open(p).convert('L').resize((side, side), Image.Resampling.BICUBIC)
        return np.asarray(im, dtype=np.float64) / 255
    a, b = gray(a_path), gray(b_path)
    C1, C2 = (0.01 * 1) ** 2, (0.03 * 1) ** 2
    kernel = np.outer(np.hanning(7), np.hanning(7)); kernel /= kernel.sum()
    from scipy.signal import fftconvolve
    mu_a = fftconvolve(a, kernel, mode='valid')
    mu_b = fftconvolve(b, kernel, mode='valid')
    saa = fftconvolve(a * a, kernel, mode='valid') - mu_a ** 2
    sbb = fftconvolve(b * b, kernel, mode='valid') - mu_b ** 2
    sab = fftconvolve(a * b, kernel, mode='valid') - mu_a * mu_b
    m = (2 * mu_a * mu_b + C1) / (mu_a ** 2 + mu_b ** 2 + C1)
    s = (2 * sab + C2) / (saa + sbb + C2)
    return float((m * s).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True)
    ap.add_argument('--labels-json', default='', help='{idx: char}；否则用 --batch-csv')
    ap.add_argument('--batch-csv', default='', help='远端评测的 eval_stdskel_batch.csv（idx,char 列）')
    ap.add_argument('--out', default='')
    a = ap.parse_args()

    if a.labels_json:
        chars = {int(k): v for k, v in json.load(open(a.labels_json, encoding='utf-8')).items()}
    elif a.batch_csv:
        chars = {}
        for r in csv.DictReader(open(a.batch_csv, encoding='utf-8-sig')):
            if r.get('set') == 'eval200fix' or 'idx' in r:
                try:
                    chars[int(r['idx'])] = r.get('char') or r.get('character')
                except (ValueError, TypeError):
                    pass
    else:
        sys.exit('need --labels-json or --batch-csv')

    g = _num(os.listdir(a.dir), 'g')
    g = {k: os.path.join(a.dir, v) for k, v in g.items() if not os.path.basename(v).startswith('gt')}
    gt = {k: os.path.join(a.dir, v) for k, v in _num(os.listdir(a.dir), 'gt').items()}

    net, c2i, vocab, dev = load_ocr()
    print(f'ocr on {dev}')
    rows = []
    top1 = top5 = n = 0
    ssims = []
    for idx, path in sorted(g.items()):
        ch = chars.get(idx)
        rec = {'idx': idx}
        if ch and ch in c2i:
            logits = ocr_image(net, dev, path)
            top5v = torch.topk(logits, 5).indices.tolist()
            preds = [vocab[j - 1] if 0 < j < len(vocab) else '?' for j in top5v]
            rec['char'] = ch
            rec['pred'] = preds[0]
            rec['top1'] = int(preds[0] == ch)
            rec['top5'] = int(ch in preds)
            n += 1; top1 += rec['top1']; top5 += rec['top5']
        if idx in gt:
            rec['ssim'] = round(ssim_pair(path, gt[idx]), 4)
            ssims.append(rec['ssim'])
        rows.append(rec)
    out = {'dir': a.dir, 'n_ocr': n,
           'ocr_top1': round(100 * top1 / n, 2) if n else None,
           'ocr_top5': round(100 * top5 / n, 2) if n else None,
           'n_ssim': len(ssims),
           'ssim_g_gt_mean': round(float(np.mean(ssims)), 4) if ssims else None,
           'rows': rows}
    dst = a.out or os.path.join(a.dir, 'metrics_local.json')
    json.dump(out, open(dst, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('LOCAL METRICS', dst,
          f"top1={out['ocr_top1']} top5={out['ocr_top5']} ssim={out['ssim_g_gt_mean']}")


if __name__ == '__main__':
    main()
