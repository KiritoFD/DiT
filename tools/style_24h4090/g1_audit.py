"""G1: what does P2 already use the Writer condition for?

No training. On a fixed set of real GT latents with their own STD condition,
we compare the flow-matching loss under three Writer conditions at several
timesteps:

    L_correct  : the sample's true writer
    L_wrong    : a same-script writer that genuinely wrote this character
    L_null     : the null writer row (unconditional calligrapher)

and we also generate Writer-swap images at the same STD and noise.

The point is not "is the writer classified correctly", but WHERE in the
diffusion trajectory and at WHICH feature scale (ink / width / centre of mass,
local edges, structural connectivity) the correct writer actually helps.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path('/root/Workspace/xy/DiT')
sys.path.insert(0, str(ROOT))

WRITERS = ['何绍基', '文徵明', '柳公权', '欧阳询', '王羲之',
           '米芾', '苏轼', '褚遂良', '赵孟頫', '颜真卿']
# Authoritative mapping: raw calligrapher id -> writer-table row (top10 only).
CALLIG_ID = {'何绍基': 53, '文徵明': 365, '柳公权': 461, '欧阳询': 479, '王羲之': 601,
             '米芾': 672, '苏轼': 703, '褚遂良': 760, '赵孟頫': 9001, '颜真卿': 956}


def writer_rows(root: Path) -> tuple[dict[str, int], int]:
    """Read exp-std/csv/callig_script_id_map_top10.json instead of guessing order.

    The file maps raw calligrapher id -> writer-table row. Compare as ints on
    both sides; a str/int mismatch here silently yields an empty map.
    """
    m = json.loads((root / 'exp-std/csv/callig_script_id_map_top10.json').read_text())
    raw_to_row = {int(k): int(v) for k, v in m['callig_map'].items()}
    name_of = {v: k for k, v in CALLIG_ID.items()}
    rows = {name_of[raw]: row for raw, row in raw_to_row.items() if raw in name_of}
    return rows, int(m['num_calligraphers'])


def load_inputs(csv_path: Path, limit: int, root: Path):
    rows = list(csv.DictReader(open(csv_path, encoding='utf-8-sig')))
    rows = [r for r in rows if not (r.get('aug') or '').strip()]
    return rows[:limit] if limit else rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--csv', default='exp-std/csv/eval200_fixed.csv')
    ap.add_argument('--out', default='assets/style_24h_20261026/G1')
    ap.add_argument('--limit', type=int, default=100)
    ap.add_argument('--t-points', default='0.15,0.35,0.55,0.75,0.9')
    args = ap.parse_args()

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    from src.unet.io import load, diffusion
    model, config, ck = load(args.ckpt, 'cuda', use_ema=True)
    process = diffusion(config)
    print(f'loaded {args.ckpt} step={ck.get("train_steps", "?")}', flush=True)

    rows = load_inputs(ROOT / args.csv, args.limit, ROOT)
    print(f'{len(rows)} GT rows', flush=True)

    # Latent cache is shared with the eval protocol; rebuild if absent.
    cache = out / 'gt_latents.npz'
    if not cache.exists():
        from diffusers import AutoencoderKL
        vae = AutoencoderKL.from_pretrained(config.vae_path, torch_dtype=torch.bfloat16,
                                            local_files_only=True).cuda().eval()
        from PIL import Image
        zs, gs = [], []
        for r in rows:
            im = Image.open(ROOT / r['image_path']).convert('RGBA').resize((256, 256),
                                                                           Image.LANCZOS)
            a = np.array(im).astype(np.float32)
            alpha = a[..., 3:4] / 255.0
            rgb = a[..., :3] * alpha + 255.0 * (1 - alpha)
            x = torch.from_numpy(rgb).permute(2, 0, 1).float() / 127.5 - 1
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                z = vae.encode(x.unsqueeze(0).cuda().to(torch.bfloat16)).latent_dist.mode()
            zs.append(z.float().cpu().reshape(4, 32, 32))
        from PIL import Image
        gs = []
        for r in rows:
            gp = ROOT / r['std_path']
            if gp.exists():
                g = np.array(Image.open(gp).convert('L'), np.float32) / 255.0
                gs.append(1.0 - g if g.mean() > 0.5 else g)
            else:
                gs.append(np.zeros((256, 256), np.float32))
        np.savez_compressed(cache, z=torch.stack(zs).numpy(), g=np.stack(gs))
    data = np.load(cache)
    Z = torch.from_numpy(data['z']).float().cuda()
    # condition g must be [B,1,64,64] exactly as the trainer feeds it
    import torch.nn.functional as Fn
    Graw = torch.from_numpy(data['g']).float().unsqueeze(1)
    G = Fn.interpolate(Graw, (64, 64), mode='bilinear', align_corners=False).cuda()
    N = len(rows)
    print(f'latents {tuple(Z.shape)} glyphs {tuple(G.shape)}', flush=True)

    wrow, n_writers = writer_rows(ROOT)
    r_name = {v: k for k, v in wrow.items()}
    print('writer rows', wrow, flush=True)
    # Script ids are REMAPPED: raw {楷:0, 行:3, 隶:4} -> table {0,1,2} via font_remap.
    # Feeding raw ids 3/4 straight into y_script overruns the 4-row table and
    # raises "device-side assert triggered" far from the real cause.
    font_remap = json.loads((ROOT / config.font_remap_json).read_text())
    sidx = {k: int(font_remap[str(v)]) for k, v in {'楷': 0, '行': 3, '隶': 4}.items()}
    print('script rows', sidx, flush=True)
    y_scr = torch.tensor([sidx[r['script']] for r in rows], device='cuda')
    y_true = torch.tensor([wrow[r['calligrapher']] for r in rows], device='cuda')
    null_id = int(model.writer.weight.shape[0] - 1)
    print(f'writer table size {model.writer.weight.shape[0]}, null row {null_id}', flush=True)

    # Same-script legal wrong writer that actually has this character in GT.
    char_writers = defaultdict(set)
    for r in load_inputs(ROOT / 'exp-std/csv/train_top10_noaug.csv', 0, ROOT):
        char_writers[(r['script'], r['character'])].add(r['calligrapher'])
    wrong_y, wrong_name = [], []
    rng = np.random.default_rng(90)
    for r in rows:
        cands = sorted(char_writers.get((r['script'], r['character']), set())
                       - {r['calligrapher']})
        if not cands:
            cands = sorted({w for w in WRITERS if w != r['calligrapher']})
        w = cands[int(rng.integers(len(cands)))]
        wrong_y.append(wrow[w])
        wrong_name.append(w)
    y_wrong = torch.tensor(wrong_y, device='cuda')

    t_points = [float(x) for x in args.t_points.split(',')]
    noise = torch.randn(Z.shape, generator=torch.Generator('cpu').manual_seed(90)).cuda()

    def fm_loss(y_cond):
        """Per-sample FM loss at fixed (t, noise) for a given writer condition."""
        res = {}
        for t in t_points:
            tv = torch.full((N,), t, device='cuda')
            x_t = (1 - tv.view(-1, 1, 1, 1)) * Z + tv.view(-1, 1, 1, 1) * noise
            target = noise - Z
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                pred = model(x_t, tv, y_cond, None, g=G, y_script=y_scr)
            err = ((pred.float() - target) ** 2).mean(dim=(1, 2, 3))
            res[t] = err.cpu().numpy()
        return res

    L_correct = fm_loss(y_true)
    L_wrong = fm_loss(y_wrong)
    L_null = fm_loss(torch.full_like(y_true, null_id))

    report = {'ckpt': args.ckpt, 'n': N, 't_points': t_points, 'per_t': {}}
    for t in t_points:
        c, w, n_ = L_correct[t], L_wrong[t], L_null[t]
        d_wrong = c - w                       # negative = correct writer better
        d_null = c - n_
        boot = []
        for _ in range(2000):
            idx = rng.integers(0, N, N)
            boot.append(float(d_wrong[idx].mean()))
        report['per_t'][str(t)] = {
            'L_correct': float(c.mean()), 'L_wrong': float(w.mean()),
            'L_null': float(n_.mean()),
            'delta_correct_minus_wrong': float(d_wrong.mean()),
            'delta_ci95': [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            'delta_correct_minus_null': float(d_null.mean()),
            'frac_correct_better_than_wrong': float((d_wrong < 0).mean()),
        }
        print(f't={t:.2f} Lc={c.mean():.5f} Lw={w.mean():.5f} Ln={n_.mean():.5f} '
              f'd(w)={d_wrong.mean():+.5f} ci={report["per_t"][str(t)]["delta_ci95"]}', flush=True)

    (out / 'g1_fm_loss.json').write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                         encoding='utf-8')

    # ---- Writer-swap images: same STD, same noise, only Writer replaced ----
    from src.eval.sampler import sample_latents
    swap_n = min(24, N)
    swap_rows = list(range(swap_n))
    conds_true = [(int(y_true[i]), -1) for i in swap_rows]
    conds_wrong = [(int(y_wrong[i]), -1) for i in swap_rows]
    z0 = noise[swap_rows].cpu()
    swap_info = {'rows': swap_rows,
                 'true_rows': [int(y_true[i]) for i in swap_rows],
                 'true': [r_name[int(y_true[i])] for i in swap_rows],
                 'wrong': [wrong_name[i] for i in swap_rows],
                 'characters': [rows[i]['character'] for i in swap_rows]}
    (out / 'swap_pairs.json').write_text(json.dumps(swap_info, ensure_ascii=False, indent=2),
                                         encoding='utf-8')
    print('G1 done ->', out, flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
