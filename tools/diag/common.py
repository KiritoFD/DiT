"""Shared CPU plumbing for the phase-2 diagnosis suite (T1--T4).

Rules of engagement on the 4090:
  * the GPU belongs to the running abc chain -- every diagnostic here is CPU-only;
  * training/eval directories are opened read-only, reports land in
    `assets/diag_20261031/` (atomic writes, small files);
  * thread pools are pinned modestly (`--threads`, default 32) and the process
    is `nice`d so the other session's eval/OCR keeps its headroom.
"""
import argparse
import json
import os
import csv
from pathlib import Path

import numpy as np
import torch


def pin(threads=32):
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(2)
    except RuntimeError:
        pass
    torch.set_grad_enabled(False)
    try:
        os.nice(10)
    except Exception:
        pass


def base_args(extra=()):
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', default='assets/p125_20261030/execution/runs/P3A/checkpoints/0020000.pt')
    p.add_argument('--csv', default='assets/p125_20261030/stage_p2.csv')
    p.add_argument('--shards', default='assets/p125_20261030/shards_all')
    p.add_argument('--out', default='assets/diag_20261031')
    p.add_argument('--n', type=int, default=96)
    p.add_argument('--threads', type=int, default=32)
    p.add_argument('--device', default='cpu', help='cpu | cuda')
    for flag, kw in extra:
        p.add_argument(flag, **kw)
    return p.parse_args()


def load_model(checkpoint, device='cpu'):
    from src.unet.io import load
    return load(checkpoint, device, use_ema=True)


def eval_args_of(config):
    from src.unet.io import eval_args
    return eval_args(config)


def diffusion_of(args):
    from src.loss import flow_kwargs_from
    from src.eval.sampler import build_diffusion
    steps = int(getattr(args, 'eval_steps', 50) or 50)
    kind = str(getattr(args, 'diffusion_type', 'flow') or 'flow')
    return build_diffusion(steps, kind, flow_kwargs_from(args))


def std_store_of(args):
    """P125 conditions: g comes from the STD-soft glyph store (raw masks)."""
    from src.utils.std_soft_glyph import StdSoftGlyphStore
    root = getattr(args, 'std_soft_root', '')
    if not root:
        raise ValueError('config has no std_soft_root: this diagnostic assumes the P125 conditioning')
    store = StdSoftGlyphStore(root, is_train=False,
                              eval_sigma=float(getattr(args, 'std_soft_eval_sigma', 0.65)),
                              preload=True, strict=True)
    return store


def shard_index(shards_dir):
    idx = {}
    for path in sorted(Path(shards_dir).glob('shard_*.npz')):
        with np.load(path) as d:
            for j, iid in enumerate(d['img_ids']):
                idx[int(iid)] = (str(path), int(j))
    return idx


def shard_latents(shards_dir, img_ids, which='latents', device='cpu'):
    """Return {img_id: fp32 tensor} for the requested ids (missing ids skipped)."""
    idx = shard_index(shards_dir)
    need = {}
    for iid in img_ids:
        if int(iid) in idx:
            path, j = idx[int(iid)]
            need.setdefault(path, []).append((int(iid), j))
    out = {}
    for path, pairs in need.items():
        with np.load(path) as d:
            for iid, j in pairs:
                out[iid] = torch.from_numpy(np.asarray(d[which][j])).float().to(device)
    return out


def read_rows(csv_path, n=None):
    with open(csv_path, encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    return rows[:n] if n else rows


NULL_WRITER_OFFSET = 47  # P125 table: rows 0..46 writers, row 47 = null


def pick_rows(rows, n, per_slot=6):
    """Diverse sampler over a training CSV: cap rows per writer slot,
    Top10 slots first, generic slots sparser.

    Stage CSVs (stage_p2.csv etc.) have no `writer_trusted` column — absence
    means "this row is a training row", so it is accepted rather than dropped.
    """
    top, gen, out = [], [], []
    for r in rows:
        try:
            s = int(r['writer_slot'] if 'writer_slot' in r else r['calligrapher_id'])
        except (KeyError, ValueError):
            continue
        trusted = r.get('writer_trusted')
        if trusted is not None and int(trusted) != 1:
            continue
        (top if s < 10 else gen).append(r)
    for pool, cap in ((top, per_slot), (gen, max(1, per_slot // 3))):
        seen = {}
        for r in pool:
            k = int(r['writer_slot'] if 'writer_slot' in r else r['calligrapher_id'])
            if seen.get(k, 0) >= cap:
                continue
            seen[k] = seen.get(k, 0) + 1
            out.append(r)
            if len(out) >= n:
                return out
    return out


def slot_of(row):
    """Writer slot for a CSV row (stage CSVs only carry calligrapher_id)."""
    return int(row['writer_slot'] if 'writer_slot' in row else row['calligrapher_id'])


def font_map(config):
    """CSV script_id (raw corpus id: 0=楷,3=行,4=隶) -> model script table row."""
    path = getattr(config, 'font_remap_json', '') or ''
    if not path:
        return lambda sid: int(sid)
    table = json.loads(Path(path).read_text(encoding='utf-8'))
    return lambda sid: int(table[str(sid)]) if str(sid) in table else int(table.get(sid, sid))


def null_writer(model):
    """Id used for the null writer, mirroring the training drop path."""
    return int(getattr(model, 'num_writers', 47))


# ---------------------------------------------------------------- OCR (CPU)
_OCR_VOCAB = '/root/.cache/modelscope/models/iic--cv_convnextTiny_ocr-recognition-handwritten_damo/snapshots/master/vocab.txt'


def ocr_setup(device='cpu'):
    """Load the handwriting OCR net (weights from the p125 panel tooling)."""
    import sys
    sys.path.insert(0, '.')
    torch.set_grad_enabled(False)
    ck = torch.load('assets/p125_20261030/ocr/duguang_ft_ext.pt', map_location='cpu', weights_only=False)
    vocab = [l.strip() for l in open(_OCR_VOCAB, encoding='utf-8')]
    char2idx = {c: i + 1 for i, c in enumerate(vocab)}
    from tools.p125.duguang_model import DuguangOCR
    net = DuguangOCR(vocab_size=len(vocab) + 1)
    net.load_state_dict(ck['model'], strict=False)
    net = net.to(device).eval()
    return net, char2idx


def _load_img(path):
    from PIL import Image
    img = Image.open(path).convert('L')
    w, h = img.size
    tw = int(np.clip(round(32 * w / h), 16, 320))
    img = img.resize((tw, 32), Image.Resampling.BICUBIC)
    x = np.asarray(img, dtype=np.float32) / 255
    return (x - 0.5) / 0.5


def ocr_score(net, images, targets, char2idx, batch=16, device=None):
    """CPU/GPU twin of tools/p125/ocr_panel_eval.score().

    `images` entries may be file paths or already-loaded 2-D float arrays.
    `targets` are character strings (one per image).
    """
    from torch.nn import functional as F
    dev = device or next(net.parameters()).device
    keep = []
    for img, tgt in zip(images, targets):
        if tgt not in char2idx:
            continue
        arr = _load_img(img) if isinstance(img, (str, Path)) else img
        keep.append((arr, tgt))
    if not keep:
        return None
    logits = []
    with torch.no_grad():
        for i in range(0, len(keep), batch):
            xs = torch.stack([torch.from_numpy(x[0])[None] for x in keep[i:i + batch]]).to(dev)
            logits.append(net(xs).cpu())
    logits = torch.cat(logits)
    tgt = torch.tensor([char2idx[t] for _, t in keep])
    conf = logits.max(-1).values
    peak = conf.argmax(-1)
    probs = F.softmax(logits[torch.arange(len(peak)), peak], dim=-1).numpy()
    order = np.argsort(-probs, axis=1)
    t = tgt.numpy()
    return dict(
        n=len(t),
        top1=round(100 * float((probs.argmax(1) == t).sum()) / len(t), 2),
        top5=round(100 * float(sum(1 for i in range(len(t)) if t[i] in order[i, :5])) / len(t), 2),
    )


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding='utf-8')
    tmp.replace(path)
    print('[diag] wrote', path, flush=True)


# ---------------------------------------------------------------- VAE (CPU)
def load_vae(config, device='cpu'):
    """SD VAE (encoder+decoder) exactly as the eval/training stacks load it."""
    from diffusers.models import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(config.vae_path, local_files_only=True)
    vae = vae.to(device).eval()
    sf = float(getattr(config, 'vae_scaling_factor', 0.18215))
    shift = float(getattr(config, 'vae_shift_factor', 0.0))
    return vae, sf, shift


def _vae_device(vae):
    try:
        return next(vae.parameters()).device
    except StopIteration:
        return torch.device('cpu')


def encode_images(vae, images, sf, shift, sample_mean=True):
    """images: (B,3,H,W) in [-1,1] -> latents (B,4,h,w) in the training scale."""
    dev = _vae_device(vae)
    x = images.to(dev)
    with torch.no_grad():
        post = vae.encode(x).latent_dist
        mean = post.mean() if callable(getattr(post, 'mean', None)) else post.mean
        lat = (mean if sample_mean else (post.sample() if callable(post.sample) else post.sample)) * sf + shift
    return lat.float()


def decode_latents(vae, latents, sf, shift, batch=4):
    """latents (B,4,32,32) in training scale -> images (B,3,256,256) in [-1,1]."""
    dev = _vae_device(vae)
    out = []
    with torch.no_grad():
        for i in range(0, len(latents), batch):
            dec = vae.decode((latents[i:i + batch].to(dev) - shift) / sf).sample
            out.append(dec.float().cpu())
    return torch.cat(out)


def gt_image_tensor(path, size=256):
    """CSV image_path -> (3,size,size) in [-1,1] (grayscale replicated)."""
    from PIL import Image
    import numpy as np
    from torch.nn import functional as F
    img = Image.open(path).convert('L').resize((size, size), Image.Resampling.BICUBIC)
    x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0)[None] * 2 - 1
    return x.expand(3, -1, -1)


def set_metrics(pred, gt, n=None):
    """Official eval metric block, collapsed to per-set means.

    `pred`/`gt` are (N,H,W,3) float arrays in [0,1] — the same layout
    `in_mem_eval.compute_set_metrics` consumes, so numbers stay comparable
    with the training-time panels (no re-implementation of SSIM/ink metrics).
    """
    import statistics as st
    from src.eval.in_mem_eval import compute_set_metrics
    n = n if n is not None else len(pred)
    m = compute_set_metrics(pred, gt, n, logger=lambda *_: None)
    mean = lambda xs: round(st.fmean(xs), 5) if len(xs) else None
    out = dict(n=n, mse=mean(m.mses), ssim=mean(m.ssims),
               sir=mean(m.sir), ink_ratio=mean(m.gtm))
    if getattr(m, 'has_ink', False):
        out.update(ink_ssim=mean(m.inks), ink_iou=mean(m.inkious),
                   skel_iou=mean(m.skels), frag=mean(m.frags),
                   hole=mean(m.holes_p), hole_gt=mean(m.holes_g))
    return out
