"""Two-stage screening funnel for the 1,000 synthesized images.

Stage 1: Physical stroke and ink quality (frag, hole, ink density, STD IoU).
Stage 2: Independent Writer style verification via style_disc_ce_holdout.pt.
         - If top-1 matches target writer (or style margin > 0): PASS -> Retain [Writer=X, Char=Y].
         - If physical quality passes but writer fails: DEGRADE -> Mark [Writer=Null, Char=Y] (Generic).
         - If physical quality fails: DISCARD.
"""
import os
import sys
import json
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, '/home/ds/Workspace/DiT')
os.chdir('/home/ds/Workspace/DiT')

from src.utils.std_soft_glyph import load_font_config, FontCoverage, render_base_mask, soften_mask
from src.eval.metrics import frag_ratio, hole_ratio
from src.eval.metrics_ink import ink_iou


class Enc(nn.Module):
    def __init__(self, d=256, n_cls=26):
        super().__init__()
        from torchvision.models import resnet18
        base = resnet18(weights=None)
        self.body = nn.Sequential(*list(base.children())[:-1])
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(512, d))
        self.cls = nn.Linear(d, n_cls)

    def emb(self, x):
        return F.normalize(self.head(self.body(x)), dim=-1)

    def forward(self, x):
        e = self.head(self.body(x))
        return self.cls(e)


def main():
    out_dir = Path('/home/ds/Workspace/xy/qwen_synth_1000')
    targets_path = out_dir / 'targets_1000.json'
    targets = json.loads(targets_path.read_text(encoding='utf-8'))
    images_dir = out_dir / 'images'

    # Font coverage for rendering STD glyphs
    font_config, script_specs, font_dirs = load_font_config('configs/std_soft_fonts.json', [])
    coverage = FontCoverage()
    SID = {'楷': 0, '行': 3, '隶': 4}

    def render_std_np(script, char):
        face = None
        for f in script_specs[SID[script]]['faces']:
            if coverage.supports(f, char):
                face = f
                break
        if face is None:
            raise KeyError(f'no font for {script} {char!r}')
        mask = render_base_mask(char, face, render_size=512)
        soft = soften_mask(mask, 0.65).resize((256, 256), Image.Resampling.BILINEAR)
        gray = np.array(Image.eval(soft, lambda v: 255 - v), dtype=np.float32) / 255.0
        return np.stack([gray, gray, gray], axis=-1)

    # Load style discriminator
    disc_path = Path('assets/style_disc_ce_holdout.pt')
    ck = torch.load(disc_path, map_location='cpu', weights_only=False)
    all_keys = [tuple(k) if isinstance(k, list) else k for k in ck['all_keys']]
    key_to_idx = {k: i for i, k in enumerate(all_keys)}

    disc = Enc(d=ck['d'], n_cls=ck['n_cls']).cuda()
    disc.load_state_dict(ck['model'])
    disc.eval()

    MEAN = torch.tensor([0.485, 0.456, 0.406], device='cuda').view(1, 3, 1, 1)
    STD = torch.tensor([0.229, 0.224, 0.225], device='cuda').view(1, 3, 1, 1)

    results = []
    print(f'Starting two-stage evaluation on {len(targets)} samples...', flush=True)

    stage1_passes = 0
    stage2_passes = 0
    degraded_count = 0
    discarded_count = 0

    rejection_reasons = Counter()

    for idx, t in enumerate(targets):
        img_path = images_dir / t['filename']
        if not img_path.exists():
            results.append({**t, 'status': 'missing_file'})
            discarded_count += 1
            continue

        im = Image.open(img_path).convert('RGB')
        pred_np = np.array(im, dtype=np.float32) / 255.0
        std_np = render_std_np(t['script'], t['character'])

        # --- Stage 1: Physical stroke and ink quality ---
        ink_mask = pred_np.mean(axis=2) < 0.5
        ink_density = float(ink_mask.mean())
        frag = frag_ratio(pred_np, std_np, thresh=0.5)
        hole = hole_ratio(pred_np, thresh=0.5)
        overlap_iou = ink_iou(pred_np, std_np, thresh=0.5)

        reasons = []
        if ink_density < 0.04:
            reasons.append('faint_ink')
        elif ink_density > 0.45:
            reasons.append('ink_blot')

        if frag > 2.5:
            reasons.append('frag_broken')

        if hole > 0.08:
            reasons.append('hollow_holes')

        if overlap_iou < 0.12:
            reasons.append('misaligned')

        pass_stage1 = len(reasons) == 0
        if not pass_stage1:
            discarded_count += 1
            for r in reasons:
                rejection_reasons[r] += 1
            results.append({
                **t,
                'pass_stage1': False,
                'pass_stage2': False,
                'status': 'discarded',
                'reasons': reasons,
                'metrics': {
                    'ink_density': round(ink_density, 4),
                    'frag': round(frag, 4),
                    'hole': round(hole, 4),
                    'overlap_iou': round(overlap_iou, 4)
                }
            })
            continue

        stage1_passes += 1

        # --- Stage 2: Independent Writer style verification ---
        im_160 = im.resize((160, 160), Image.Resampling.BILINEAR)
        x = torch.from_numpy(np.array(im_160, dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).cuda()
        x_norm = (x - MEAN) / STD

        with torch.inference_mode():
            logits = disc(x_norm)[0]
            probs = F.softmax(logits, dim=0).cpu().numpy()

        target_key = (t['calligrapher'], t['script'])
        target_idx = key_to_idx.get(target_key)

        pred_idx = int(probs.argmax())
        pred_key = all_keys[pred_idx]
        pred_writer = pred_key[0]

        target_prob = float(probs[target_idx]) if target_idx is not None else 0.0

        # Writers prob sum
        writer_probs = defaultdict(float)
        for k_idx, (w, s) in enumerate(all_keys):
            writer_probs[w] += probs[k_idx]

        top_writer = max(writer_probs.keys(), key=lambda w: writer_probs[w])
        pass_stage2 = (pred_writer == t['calligrapher']) or (top_writer == t['calligrapher'])

        if pass_stage2:
            stage2_passes += 1
            status = 'verified_writer'
            final_writer_tag = t['calligrapher']
        else:
            degraded_count += 1
            status = 'degraded_to_null_writer'
            final_writer_tag = 'Null'

        results.append({
            **t,
            'pass_stage1': True,
            'pass_stage2': pass_stage2,
            'status': status,
            'assigned_writer': final_writer_tag,
            'pred_writer': pred_writer,
            'top_writer': top_writer,
            'target_prob': round(target_prob, 4),
            'metrics': {
                'ink_density': round(ink_density, 4),
                'frag': round(frag, 4),
                'hole': round(hole, 4),
                'overlap_iou': round(overlap_iou, 4)
            }
        })

        if (idx + 1) % 100 == 0 or (idx + 1) == len(targets):
            print(f'Evaluated [{idx+1}/{len(targets)}] | '
                  f'Pass Stage 1: {stage1_passes} | Verified Writer: {stage2_passes} | '
                  f'Degraded Generic: {degraded_count} | Discarded: {discarded_count}', flush=True)

    summary = {
        'total_generated': len(targets),
        'pass_stage1_total': stage1_passes,
        'pass_stage1_rate': round(stage1_passes / len(targets), 4),
        'pass_stage2_verified_writer': stage2_passes,
        'pass_stage2_rate': round(stage2_passes / len(targets), 4),
        'degraded_to_null_writer': degraded_count,
        'degraded_rate': round(degraded_count / len(targets), 4),
        'discarded': discarded_count,
        'discarded_rate': round(discarded_count / len(targets), 4),
        'rejection_reasons': dict(rejection_reasons),
        'per_writer_verified': dict(Counter(r['calligrapher'] for r in results if r['status'] == 'verified_writer')),
        'per_writer_degraded': dict(Counter(r['calligrapher'] for r in results if r['status'] == 'degraded_to_null_writer')),
        'per_writer_discarded': dict(Counter(r['calligrapher'] for r in results if r['status'] == 'discarded'))
    }

    report_path = out_dir / 'funnel_summary.json'
    report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')

    detail_path = out_dir / 'funnel_details.json'
    detail_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')

    print('\n================ FUNNEL REPORT ===============')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
