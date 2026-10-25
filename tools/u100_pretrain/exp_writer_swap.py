"""Writer-swap counterfactual: fixed STD + fixed noise, switch writer rows.

For the first --chars val100 characters, generate one image per writer row
(0..9) with identical noise. Measurements:
  1. writer probe Top-1 over the swapped generations (chance 10%): does the
     condition actually steer identity?
  2. feature spread: mean pairwise DINO distance between different-writer
     generations of the same char, vs same-writer different-char distance of
     real data for the same chars (normalized ratio).
  3. poster grid (writers x chars).
"""
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
TOP10_RAW = {601: 0, 703: 1, 479: 2, 956: 3, 9001: 4, 760: 5, 672: 6, 365: 7, 461: 8, 53: 9}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--val100', default='assets/u100_pretrain_20261024/validation_val100.csv')
    parser.add_argument('--std-root', default='assets/generic48_kxl_top10_20261022/std_bbox64')
    parser.add_argument('--bank', default='assets/u100_pretrain_20261024/style_bank_top10.npz')
    parser.add_argument('--probe', default='assets/u100_pretrain_20261024/writer_probe.pt')
    parser.add_argument('--chars', type=int, default=12)
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--out', default='assets/u100_pretrain_20261024/writer_swap')
    args = parser.parse_args()
    out = Path(args.out); (out / 'gen').mkdir(parents=True, exist_ok=True)
    from src.unet.io import load
    model, config, ck = load(args.checkpoint, 'cuda', use_ema=True)
    model.eval()
    from src.utils.std_soft_glyph import StdSoftGlyphStore
    store = StdSoftGlyphStore(args.std_root, sigma_min=0, sigma_max=0, eval_sigma=float(config.std_soft_eval_sigma))
    fremap = json.loads((Path(config.condition_tables_root) / 'font_remap.json').read_text())
    vrow = list(csv.DictReader(open(args.val100, encoding='utf-8-sig')))[:args.chars]

    from src.eval.decoder import _get_vae, _get_lpips, _lpips_per_sample
    from src.eval.scoring import _ssim
    from src.eval.metrics_ink import ink_iou
    vae = _get_vae('cuda', config.vae_path, float(config.vae_shift_factor))
    lp = _get_lpips()
    Wn = 10
    gen = np.zeros((args.chars, Wn, 4, 32, 32), dtype=np.float32)
    rows_meta = vrow
    g_batch = torch.stack([store.get(int(r['script_id']), r['character'], sigma=float(config.std_soft_eval_sigma)) for r in rows_meta]).cuda()
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        for w in range(Wn):
            wv = torch.full((args.chars,), w, dtype=torch.long, device='cuda')
            sv = torch.tensor([fremap[r['script_id']] for r in rows_meta], device='cuda')
            noise = torch.Generator(device='cuda'); noise.manual_seed(90)
            z = torch.randn(args.chars, 4, 32, 32, device='cuda', generator=noise)
            ts = np.linspace(0, 1, args.steps + 1)
            for i in range(args.steps):
                t = torch.full((args.chars,), ts[i], device='cuda')
                v1 = model(z, t, wv, g=g_batch, y_script=sv)
                z2 = z + (ts[i + 1] - ts[i]) * v1
                if i + 1 < args.steps:
                    t2 = torch.full((args.chars,), ts[i + 1], device='cuda')
                    v2 = model(z2, t2, wv, g=g_batch, y_script=sv)
                    z = z + 0.5 * (ts[i + 1] - ts[i]) * (v1 + v2)
                else:
                    z = z2
            gen[:, w] = z.float().cpu().numpy()
    # decode
    dec = []
    with torch.inference_mode():
        for w in range(Wn):
            zz = torch.from_numpy(gen[:, w]).cuda() / config.vae_scaling_factor
            im = vae.decode(zz).sample.clamp_(-1, 1)
            im = ((im / 2 + 0.5).clamp(0, 1) * 255).byte().cpu().numpy()
            dec.append(im)
    # save pngs + probe + distances
    from transformers import AutoModel
    dino = AutoModel.from_pretrained('facebook/dinov2-base', local_files_only=True).cuda().eval().requires_grad_(False)
    dm = torch.tensor([.485, .456, .406])[None, :, None, None].cuda()
    ds = torch.tensor([.229, .224, .225])[None, :, None, None].cuda()
    probe = torch.load(args.probe, map_location='cpu', weights_only=False)
    feats = np.zeros((args.chars, Wn, 768), dtype=np.float32)
    preds = np.zeros((args.chars, Wn), dtype=np.int64)
    with torch.inference_mode():
        for w in range(Wn):
            gray = (dec[w].transpose(0, 2, 3, 1).mean(3))[:, None].astype(np.float32)
            for ci in range(args.chars):
                Image.fromarray(gray[ci, :, :, 0].astype(np.uint8)).save(out / 'gen' / f'w{w}_c{ci}.png')
            x = torch.from_numpy(gray).cuda() / 255
            x = torch.nn.functional.interpolate(x, (224, 224), mode='bicubic', align_corners=False, antialias=True).clamp_(0, 1).expand(-1, 3, -1, -1)
            x = (x - dm) / ds
            with torch.autocast('cuda', dtype=torch.bfloat16):
                t = dino(pixel_values=x).last_hidden_state[:, 0]
            f = torch.nn.functional.normalize(t.float(), dim=-1).cpu().numpy()
            feats[:, w] = f
            logits = f @ probe['coef'].T + probe['intercept']
            preds[:, w] = logits.argmax(1)
    probe_top1 = float((preds == np.arange(Wn)[None, :]).mean())
    # spreads on same chars
    cross, same_writer_real = [], []
    bank = np.load(args.bank)
    blook = {int(v): i for i, v in enumerate(bank['ids'])}
    bfeat = {}
    # real per-writer features from bank rows matching (writer,char)
    cellf = {}
    bl = blook
    feats_by_id = {int(v): bank['features'][i] for i, v in enumerate(bank['ids'])}
    import csv as _csv
    for r in _csv.DictReader(open('/root/Workspace/xy/DiT/assets/generic48_kxl_top10_20261022/top10_train.csv', encoding='utf-8-sig')):
        if r['generic'] != '0': continue
        w = TOP10_RAW[int(r['calligrapher_id'])]; c = ord(r['character'])
        iid = int(r['img_id'])
        if iid in feats_by_id: cellf.setdefault((w, c), []).append(feats_by_id[iid])
    for ci, r in enumerate(rows_meta):
        w_real = TOP10_RAW[int(r['calligrapher_id'])]; c = ord(r['character'])
        gf = feats[ci]                                   # [Wn,768] generations
        d_cross = [float(np.linalg.norm(gf[a] - gf[b])) for a in range(Wn) for b in range(a + 1, Wn)]
        cross.append(float(np.mean(d_cross)))
        reals = [cellf[(w, c)] for w in range(10) if (w, c) in cellf and len(cellf[(w, c)]) > 0]
        if len(reals) >= 2:
            m = [np.mean(v, 0) for v in reals]
            d_real = [float(np.linalg.norm(m[a] - m[b])) for a in range(len(m)) for b in range(a + 1, len(m))]
            same_writer_real.append(float(np.mean([np.linalg.norm(np.array(v) - np.mean(v, 0)) for v in reals]) /
                                          (np.mean(d_real) + 1e-9)))
    spread_ratio = float(np.mean(cross) / (np.mean([np.linalg.norm(feats[ci, a] - feats[ci, b])
                                                     for ci in range(args.chars) for a in range(Wn) for b in range(a + 1, Wn)]) + 1e-9))
    # poster
    CELL = 150; PAD = 6
    names = {v: k for k, v in TOP10_RAW.items()}
    grid = Image.new('RGB', (PAD + (CELL + PAD) * (args.chars + 1), PAD + (CELL + PAD) * Wn), (14, 14, 18))
    from PIL import ImageDraw, ImageFont
    dr = ImageDraw.Draw(grid)
    try: font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 22)
    except Exception: font = ImageFont.load_default()
    for ci, r in enumerate(rows_meta):
        dr.text((PAD + (CELL + PAD) * (ci + 1) + 40, 8), r['character'], font=font, fill=(235, 235, 150))
    for w in range(Wn):
        dr.text((4, PAD + (CELL + PAD) * w + 60), f'w{w}', font=font, fill=(150, 225, 255))
        for ci in range(args.chars):
            im = Image.open(out / 'gen' / f'w{w}_c{ci}.png').resize((CELL, CELL))
            grid.paste(im, (PAD + (CELL + PAD) * (ci + 1), PAD + (CELL + PAD) * w))
    grid.save(out / 'writer_swap_poster.png')
    json.dump(dict(probe_top1=probe_top1, spread_cross_writer_mean=float(np.mean(cross)),
                   real_within_writer_over_real_cross=float(np.mean(same_writer_real)) if same_writer_real else None,
                   chars=[r['character'] for r in rows_meta]),
              open(out / 'swap_metrics.json', 'w'), indent=1)
    print('SWAP_DONE', json.dumps(dict(probe_top1=probe_top1, spread_cross_writer_mean=round(float(np.mean(cross)), 4))))


if __name__ == '__main__':
    main()
