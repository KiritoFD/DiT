import os, sys, json, csv
from pathlib import Path
import torch
import numpy as np
from PIL import Image
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.eval_gen_with_disc import Enc, ImgSet
from src.eval.style.bank import resolve

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_dir = ROOT / "assets/results/eval_dic768_progression"
    steps = [20000, 50000, 60000, 70000, 75000, 80000]
    
    # 1. First collect geometric metrics from CSVs
    print("=" * 110)
    print(f"{'Step & CFG':18s} | {'SSIM Mean':9s} | {'SSIM Med':9s} | {'LPIPS-VGG':9s} | {'Ink-IoU':9s} | {'Frag (断笔)':11s} | {'SoftInk':9s}")
    print("=" * 110)
    geo_results = {}
    for s in steps:
        for cfg in (1.0, 1.6):
            sum_csv = base_dir / f"step{s:07d}_cfg{cfg:.1f}/eval_stdskel_summary.csv"
            if sum_csv.exists():
                r = list(csv.DictReader(open(sum_csv)))[-1]
                geo_results[(s, cfg)] = {
                    "ssim": float(r["ssim_mean"]),
                    "ssim_med": float(r["ssim_med"]),
                    "lpips": float(r["lpips_mean"]),
                    "ink_iou": float(r.get("ink_iou_mean", 0.0)),
                    "frag": float(r["frag_ratio"]),
                    "soft_ink": float(r["soft_ink_ratio"])
                }
                print(f"Step{s:6d}_cfg{cfg:.1f}  | {float(r['ssim_mean']):9.4f} | {float(r['ssim_med']):9.4f} | {float(r['lpips_mean']):9.4f} | {float(r.get('ink_iou_mean', 0.0)):9.4f} | {float(r['frag_ratio']):11.3f} | {float(r['soft_ink_ratio']):9.3f}")

    # 2. Independent Discriminator Evaluation (N=181)
    disc_ck = torch.load(ROOT / 'assets/style_disc_ce_holdout.pt', map_location='cpu', weights_only=False)
    disc = Enc(d=disc_ck['d'], n_cls=disc_ck['n_cls']).to(device)
    disc.load_state_dict(disc_ck['model'])
    disc.eval()

    roots = [str(ROOT), str(ROOT / 'data'), str(ROOT / 'exp-std')]
    tr = list(csv.DictReader(open(ROOT / 'exp-std/csv/train_top10_aug_sym.csv', encoding='utf-8-sig')))
    by_slot = defaultdict(list)
    tr_paths = []
    for r in tr:
        if (r.get("aug") or "").strip(): continue
        p = resolve(r.get("image_path", ""), roots)
        if not p: continue
        w, s = r.get("calligrapher", "").strip(), r.get("script", "").strip()
        tr_paths.append(p)
        by_slot[(w, s)].append(len(tr_paths) - 1)
        
    IS_tr = ImgSet(tr_paths, tag="bank")
    E_tr = IS_tr.emb_all(disc, device)
    protos = {}
    for k, idxs in sorted(by_slot.items()):
        v = E_tr[idxs].mean(0)
        protos[k[0] + '·' + k[1]] = v / np.linalg.norm(v)
    slot_list = sorted(protos)
    P = np.stack([protos[k] for k in slot_list])
    slot_script = {k: k.split('·')[1] if '·' in k else '' for k in slot_list}

    eval_csv = ROOT / "exp-std/csv/eval200_fixed.csv"
    eval_rows = list(csv.DictReader(open(eval_csv, encoding="utf-8-sig")))[:187]
    calligraphers = ["何绍基", "文徵明", "柳公权", "欧阳询", "王羲之", "米芾", "苏轼", "褚遂良", "赵孟頫", "颜真卿"]

    print("\n" + "=" * 110)
    print(f"{'Step & CFG':18s} | {'Micro Top-1':11s} | {'Macro Top-1':11s} | {'Target Margin (Mean)':21s} | {'Margin Median':13s}")
    print("=" * 110)

    for s in steps:
        for cfg in (1.0, 1.6):
            gdir = base_dir / f"step{s:07d}_cfg{cfg:.1f}/eval_samples_ctrl/step{s:07d}/eval200fix"
            gen_imgs = []
            valid_indices = []
            for idx in range(187):
                imp = gdir / f"g{idx}.png"
                if not imp.exists(): continue
                r = eval_rows[idx]
                slot_key = r.get("calligrapher", "").strip() + '·' + r.get("script", "").strip()
                if slot_key not in protos:
                    continue # N=181
                im = Image.open(imp).convert("RGB").resize((160, 160), Image.BICUBIC)
                gen_imgs.append(np.asarray(im))
                valid_indices.append(idx)

            disc_batch = torch.from_numpy(np.stack(gen_imgs)).permute(0, 3, 1, 2).float().to(device) / 255.0
            mean = torch.tensor((0.485, 0.456, 0.406), device=device).view(1, 3, 1, 1)
            std = torch.tensor((0.229, 0.224, 0.225), device=device).view(1, 3, 1, 1)
            disc_batch = (disc_batch - mean) / std

            with torch.no_grad():
                emb = disc.emb(disc_batch).cpu().numpy()
                emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
                sims = emb @ P.T

            top1_hits = []
            margins = []
            by_cal = defaultdict(list)

            for i, idx in enumerate(valid_indices):
                r = eval_rows[idx]
                slot_key = r.get("calligrapher", "").strip() + '·' + r.get("script", "").strip()
                ci = slot_list.index(slot_key)
                is_hit = (int(np.argmax(sims[i])) == ci)
                top1_hits.append(1 if is_hit else 0)
                by_cal[r.get("calligrapher", "").strip()].append(1 if is_hit else 0)

                others = [sims[i][k2] for k2 in range(len(slot_list))
                          if k2 != ci and slot_script[slot_list[k2]] == r.get("script", "").strip()]
                if others:
                    margins.append(float(sims[i][ci] - max(others)))

            micro = float(np.mean(top1_hits)) * 100.0
            macro = float(np.mean([np.mean(by_cal[c]) for c in calligraphers if c in by_cal])) * 100.0
            m_mean = float(np.mean(margins))
            m_med = float(np.median(margins))

            print(f"Step{s:6d}_cfg{cfg:.1f}  | {micro:10.2f}% | {macro:10.2f}% | {m_mean:+20.4f} | {m_med:+12.4f}")

    print("=" * 110)

if __name__ == "__main__":
    main()
