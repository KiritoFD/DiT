import os, sys, json, csv
from pathlib import Path
import torch
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.unet.io import eval_args, diffusion, Config
from src.unet.model import LatentUNet
from src.eval.in_mem_eval import run_in_mem_eval
from tools.eval_gen_with_disc import Enc, ImgSet
from src.eval.style.bank import resolve

def run_eval_for_ckpt(ckpt_path, step, cfgs=(1.0, 1.6), device="cuda"):
    conf_dict = json.loads(open(ROOT / "src/unet/configs/connectivity/DiC_U100_b768.json").read())
    config = Config(**conf_dict)
    
    model = LatentUNet(tuple(config.channels), config.num_writers, config.num_scripts,
                       num_res_blocks=2, residual_cond_gate=True,
                       stage_specific_cond=True, stage_cond_mode='adapter', act_fn='gelu').to(device)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    weights = {k.removeprefix("_orig_mod."): v for k, v in ck["ema"].items()}
    model.load_state_dict(weights, strict=True)
    model.eval()
    
    results = {}
    for cfg in cfgs:
        eval_conf = eval_args(config)
        eval_conf.eval_cfg = float(cfg)
        eval_conf.eval_steps = 50
        eval_conf.in_mem_eval_batch = 32
        eval_conf.in_mem_eval_poster = False
        
        target_dir = ROOT / f"assets/results/eval_dic768_progression/step{step:07d}_cfg{cfg:.1f}"
        target_dir.mkdir(parents=True, exist_ok=True)
        
        with torch.no_grad():
            run_in_mem_eval(
                model, eval_conf, step, device,
                results_dir=str(target_dir)
            )
            
        # Read eval_stdskel_summary.csv
        sum_csv = target_dir / "eval_stdskel_summary.csv"
        metrics = {}
        if sum_csv.exists():
            rows = list(csv.DictReader(open(sum_csv)))
            if rows:
                r = rows[-1]
                metrics = {
                    "ssim": float(r["ssim_mean"]),
                    "ssim_med": float(r["ssim_med"]),
                    "lpips": float(r["lpips_mean"]),
                    "frag": float(r["frag_ratio"]),
                    "ink_iou": float(r.get("ink_iou_mean", 0.0)),
                    "soft_ink": float(r["soft_ink_ratio"])
                }
        results[cfg] = {
            "metrics": metrics,
            "gen_dir": target_dir / f"eval_samples_ctrl/step{step:07d}/eval200fix"
        }
    return results

def score_disc(gen_dirs_dict, device="cuda"):
    disc_ck = torch.load(ROOT / 'assets/style_disc_ce_holdout.pt', map_location='cpu', weights_only=False)
    disc = Enc(d=disc_ck['d'], n_cls=disc_ck['n_cls']).to(device)
    disc.load_state_dict(disc_ck['model'])
    disc.eval()

    roots = [str(ROOT), str(ROOT / 'data'), str(ROOT / 'exp-std')]
    tr = list(csv.DictReader(open(ROOT / 'exp-std/csv/train_top10_aug_sym.csv', encoding='utf-8-sig')))
    from collections import defaultdict
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

    disc_scores = {}
    for label, gdir in gen_dirs_dict.items():
        # Load generated images
        gen_imgs = []
        valid_indices = []
        for idx in range(187):
            imp = gdir / f"{idx:05d}.png"
            if not imp.exists(): continue
            r = eval_rows[idx]
            slot_key = r.get("calligrapher", "").strip() + '·' + r.get("script", "").strip()
            if slot_key not in protos:
                continue # Skip the 6 文徵明·隶 rows to guarantee exact N=181
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

        disc_scores[label] = {
            "n": len(valid_indices),
            "micro_top1": micro,
            "macro_top1": macro,
            "margin_mean": m_mean,
            "margin_med": m_med
        }
    return disc_scores

def main():
    ckpt_root = ROOT / "assets/results/unet_dic_100k_b768/20261018-063716-Unet-flow/checkpoints"
    steps_to_eval = [20000, 50000, 60000, 70000, 75000, 80000]
    
    print(f"Starting progression eval on Node 4090 for steps: {steps_to_eval} ...", flush=True)
    all_summary = {}
    gen_dirs_for_disc = {}
    
    for s in steps_to_eval:
        cp = ckpt_root / f"{s:07d}.pt"
        if not cp.exists():
            print(f"Warning: {cp} does not exist, skipping.")
            continue
        print(f"\n>>> Evaluating Step {s} ...", flush=True)
        res = run_eval_for_ckpt(cp, s, cfgs=(1.0, 1.6))
        all_summary[s] = res
        for cfg_val in (1.0, 1.6):
            gen_dirs_for_disc[f"Step{s}_cfg{cfg_val:.1f}"] = res[cfg_val]["gen_dir"]

    print("\n>>> Running Independent Discriminator Evaluation (N=181) ...", flush=True)
    disc_results = score_disc(gen_dirs_for_disc)
    
    # Save combined report
    combined_report = {
        "geometric": {str(s): {str(cfg): all_summary[s][cfg]["metrics"] for cfg in all_summary[s]} for s in all_summary},
        "discriminator": disc_results
    }
    out_json = ROOT / "assets/results/eval_dic768_progression/dic768_progression_eval.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(combined_report, f, ensure_ascii=False, indent=2)
    print("Saved combined report to:", out_json)

if __name__ == "__main__":
    main()
