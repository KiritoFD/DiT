"""Frozen DINOv2-B CLS feature bank over the arm-100k training CSV.

Writers are the pipeline's own group ids (calligrapher_id column), so the
cross-character reference sampler and SupCon share one labeling with the
writer-ID control arm. Images load from repo-root relative paths, mirroring
the reference-line cache protocol (gray, bicubic 128, /255).
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from transformers import AutoModel

sys = None
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


class Images(Dataset):
    def __init__(self, rows): self.rows = rows
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        with Image.open(self.rows[i]['image_path']) as image:
            gray = np.asarray(image.convert('L').resize((128, 128), Image.Resampling.BICUBIC)).copy()
        return torch.from_numpy(gray)[None]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default='assets/u100_pretrain_20261024/train_100k.csv')
    parser.add_argument('--out', default='assets/u100_pretrain_20261024/style_bank.npz')
    parser.add_argument('--batch', type=int, default=192)
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists(): raise FileExistsError(out)
    rows = list(csv.DictReader(open(args.csv, encoding='utf-8-sig')))
    torch.set_num_threads(8); torch.backends.cudnn.benchmark = True
    model = AutoModel.from_pretrained('facebook/dinov2-base', local_files_only=True).cuda().eval().requires_grad_(False)
    mean = torch.tensor([.485, .456, .406], device='cuda')[None, :, None, None]
    std = torch.tensor([.229, .224, .225], device='cuda')[None, :, None, None]
    loader = DataLoader(Images(rows), batch_size=args.batch, num_workers=8, pin_memory=True)
    features = np.empty((len(rows), 768), np.float32)
    offset = 0; start = time.monotonic()
    with torch.inference_mode():
        for gray in loader:
            x = gray.cuda(non_blocking=True).float() / 255
            x = F.interpolate(x, (224, 224), mode='bicubic', align_corners=False, antialias=True).clamp_(0, 1).expand(-1, 3, -1, -1)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                feat = model(pixel_values=(x - mean) / std).last_hidden_state[:, 0]
            n = len(gray); features[offset:offset + n] = F.normalize(feat.float(), dim=-1).cpu().numpy(); offset += n
            if offset % (args.batch * 50) == 0: print('BANK', offset, len(rows), round(offset / (time.monotonic() - start), 1), 'img/s', flush=True)
    if not np.isfinite(features).all(): raise ValueError('nonfinite features')
    writers = np.array([int(r['calligrapher_id']) for r in rows], np.int64)
    chars = np.array([ord(r['character']) for r in rows], np.int64)
    fremap = json.loads(Path('assets/u100_pretrain_20261024/tables/font_remap.json').read_text())
    scripts = np.array([fremap[str(r['script_id'])] for r in rows], np.int64)
    ids = np.array([int(r['img_id']) for r in rows], np.int64)
    np.savez(out, ids=ids, features=features, writers=writers, chars=chars, scripts=scripts)
    out.with_suffix('.json').write_text(json.dumps(dict(
        model='facebook/dinov2-base', csv=str(args.csv),
        csv_sha256=hashlib.sha256(Path(args.csv).read_bytes()).hexdigest(),
        rows=len(rows), writers=int(len(np.unique(writers))), note='writers=pipeline group ids'), indent=2))
    print('STYLE_BANK_READY', len(rows), flush=True)


if __name__ == '__main__':
    main()
