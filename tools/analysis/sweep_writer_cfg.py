import os, sys, json, csv
from pathlib import Path
import torch
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.unet.io import load, eval_args, diffusion
from src.eval.in_mem_eval import run_in_mem_eval
from tools.eval_gen_with_disc import Enc, ImgSet
from src.eval.style.bank import resolve

def run_cfg_for_model(name, ckpt_path, cfg_scales, out_base, device="cuda"):
    model, config, ck = load(str(ckpt_path), device, use_ema=False)
    step = int(ck.get('train_steps', 55000))
    del ck
    
    # Load discriminator
    disc_ck = torch.load(ROOT / 'assets/style_disc_ce_holdout.pt', map_location='cpu', weights_only=False)
    all_keys = disc_ck['all_keys']
    disc = Enc(d=disc_ck['d'], n_cls=disc_ck['n_cls']).to(device)
    disc.load_state_dict(disc_ck['model'])
    disc.eval()
    
    # Build bank prototypes
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
    
    IS_tr = ImgSet(tr_paths[:2000], tag="bank")
    E_tr = IS_tr.emb_all(disc, device)
    protos = {}
    for k, idxs in sorted(by_slot.items()):
        v = E_tr[[i for i in idxs if i < len(E_tr)]].mean(0)
        protos[k[0] + '·' + k[1]] = v / np.linalg.norm(v)
    
    slot_list = sorted(protos)
    P = np.stack([protos[k] for k in slot_list])
    slot_script = {k: k.split('·')[1] if '·' in k else '' for k in slot_list}
    
    eval_csv = ROOT / 'exp-std/csv/eval200_fixed.csv'
    rows = list(csv.DictReader(open(eval_csv, encoding='utf-8-sig')))[:187]
    gt_meta = [(r.get("calligrapher", "").strip(), r.get("script", "").strip()) for r in rows]
    
    results = []
    
    for cfg in cfg_scales:
        print(f"\n>>> Sampling {name} at CFG={cfg} ...", flush=True)
        eval_conf = eval_args(config)
        eval_conf.eval_cfg = float(cfg)
        eval_conf.eval_steps = 50
        eval_conf.in_mem_eval_batch = 32
        eval_conf.in_mem_eval_poster = False
        
        target_dir = Path(out_base) / f"{name}_cfg{cfg:.1f}"
        target_dir.mkdir(parents=True, exist_ok=True)
        
        with torch.no_grad():
            values = run_in_mem_eval(
                model, eval_conf, step, device,
                results_dir=str(target_dir),
                diffusion_override=diffusion(config)
            )
        
        gen_dir = target_dir / f"eval_samples_ctrl/step{step:07d}/eval200fix"
        gen_paths = [str(gen_dir / f"g{i}.png") for i in range(187)]
        
        # Evaluate with discriminator (N=181 valid)
        IS_g = ImgSet(gen_paths, tag=f"{name}_cfg{cfg}")
        E = IS_g.emb_all(disc, device)
        E = E / np.linalg.norm(E, axis=1, keepdims=True)
        sims = E @ P.T
        
        top1_hits = []
        writer_margins = []
        by_cal = defaultdict(list)
        
        for j in range(187):
            w, s = gt_meta[j]
            key = w + '·' + s
            if key not in protos:
                continue # 6 skipped
            ci = slot_list.index(key)
            pred_ci = int(np.argmax(sims[j]))
            is_hit = (pred_ci == ci)
            top1_hits.append(1 if is_hit else 0)
            by_cal[w].append(1 if is_hit else 0)
            
            others = [sims[j][k2] for k2 in range(len(slot_list))
                      if k2 != ci and slot_script[slot_list[k2]] == s]
            if others:
                writer_margins.append(float(sims[j][ci] - max(others)))
        
        n_val = len(top1_hits)
        micro_top1 = np.mean(top1_hits) * 100.0
        macro_top1 = np.mean([np.mean(by_cal[c]) for c in by_cal]) * 100.0
        margin_mean = np.mean(writer_margins)
        margin_med = np.median(writer_margins)
        
        ssim_val = values.get('ssim', 0.0) if values else 0.0
        frag_val = values.get('frag_ratio', 0.0) if values else 0.0
        ink_iou = values.get('ink_iou', 0.0) if values else 0.0
        
        print(f"[{name} CFG={cfg:.1f}] N={n_val} | Micro Top1: {micro_top1:.2f}% | Macro Top1: {macro_top1:.2f}% | Margin Mean: {margin_mean:+.4f} | Margin Med: {margin_med:+.4f} | SSIM: {ssim_val:.4f} | Frag: {frag_val:.3f} | Ink-IoU: {ink_iou:.4f}")
        results.append({
            "model": name,
            "cfg": cfg,
            "n_val": n_val,
            "micro_top1": micro_top1,
            "macro_top1": macro_top1,
            "margin_mean": margin_mean,
            "margin_med": margin_med,
            "ssim": ssim_val,
            "frag": frag_val,
            "ink_iou": ink_iou
        })
        
    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Baseline_55k")
    parser.add_argument("--ckpt", default="/home/ds/Workspace/DiT/assets/results/unet_connectivity_100000/baseline_resume/20261018-030921-Unet-flow/checkpoints/0055000.pt")
    parser.add_argument("--cfgs", nargs="+", type=float, default=[1.0, 1.2, 1.4, 1.6])
    parser.add_argument("--out", default="assets/results/cfg_sweep")
    args = parser.parse_args()
    
    res = run_cfg_for_model(args.model, args.ckpt, args.cfgs, args.out)
    with open(f"{args.out}/{args.model}_results.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
