import os, sys, json, csv
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.unet.io import load, eval_args, diffusion, Config
from src.unet.model import LatentUNet
from src.eval.in_mem_eval import _get_cache
from src.eval.sampler import sample_latents
from src.utils.vae_scaling import resolve_vae_scaling
from tools.eval_gen_with_disc import Enc, ImgSet
from src.eval.style.bank import resolve
from src.eval.style.encoder import StyleEncoder

def build_model(name, ckpt_path, device, is_4090=False):
    if is_4090:
        conf_dict = json.loads(open(ROOT / "src/unet/configs/connectivity/DiC_U100_b768.json").read())
        config = Config(**conf_dict)
        model = LatentUNet(tuple(config.channels), config.num_writers, config.num_scripts,
                           num_res_blocks=2, residual_cond_gate=True,
                           stage_specific_cond=True, stage_cond_mode='adapter', act_fn='gelu').to(device)
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        weights = {k.removeprefix("_orig_mod."): v for k, v in ck["ema"].items()}
        model.load_state_dict(weights, strict=True)
        model.eval()
    else:
        model, config, ck = load(str(ckpt_path), device, use_ema=False)
    return model, config

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Select 10 diverse test samples (1 per writer) from eval200_fixed.csv
    eval_csv = ROOT / "exp-std/csv/eval200_fixed.csv"
    rows = list(csv.DictReader(open(eval_csv, encoding="utf-8-sig")))[:187]
    calligraphers = ["何绍基", "文徵明", "柳公权", "欧阳询", "王羲之", "米芾", "苏轼", "褚遂良", "赵孟頫", "颜真卿"]
    name_to_cid = {
        "何绍基": 53, "文徵明": 365, "柳公权": 461, "欧阳询": 479, "王羲之": 601,
        "米芾": 672, "苏轼": 703, "褚遂良": 760, "赵孟頫": 9001, "颜真卿": 956
    }
    script_to_sid = {"楷": 0, "行": 3, "隶": 4}

    callig_remap = json.loads(open(ROOT / "assets/triple_tables_best_minimal/callig_remap.json").read())
    script_remap = json.loads(open(ROOT / "assets/triple_tables_best_minimal/font_remap.json").read())
    c2w = {k: int(v) for k, v in callig_remap.items()}
    s2s = {k: int(v) for k, v in script_remap.items()}
    
    selected_indices = []
    seen_w = set()
    for idx, r in enumerate(rows):
        w = r.get("calligrapher", "").strip()
        if w in calligraphers and w not in seen_w and r.get("script", "").strip() in ("楷", "行"):
            selected_indices.append(idx)
            seen_w.add(w)
        if len(selected_indices) == 10:
            break
            
    print(f"Selected {len(selected_indices)} anchor evaluation items: {selected_indices}")
    for idx in selected_indices:
        r = rows[idx]
        print(f"  idx {idx:3d}: {r.get('calligrapher')}·{r.get('script')} - {r.get('character')}")

    # Load cache for std_soft glyphs
    conf_dict = json.loads(open(ROOT / "src/unet/configs/connectivity/DiC_U100_b768.json").read())
    args = eval_args(Config(**conf_dict))
    cache = _get_cache(str(eval_csv), 187, None, '', args)
    
    # Load VAE for decoding
    from diffusers.models import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(str(ROOT / "pretrained_models/sd-vae-ft-ema")).to(device).eval()
    sf, _ = resolve_vae_scaling(args)
    
    # Load independent discriminator
    disc_ck = torch.load(ROOT / 'assets/style_disc_ce_holdout.pt', map_location='cpu', weights_only=False)
    disc = Enc(d=disc_ck['d'], n_cls=disc_ck['n_cls']).to(device)
    disc.load_state_dict(disc_ck['model'])
    disc.eval()
    
    # Load character classifier (7765 classes) for Content Preservation (OCR proxy)
    charenc = StyleEncoder.load(str(ROOT / 'pretrained_models/eval_classifier.pt'), device=device)
    
    # Build bank prototypes for 22 slots
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

    # Models to test
    model_targets = [
        ("Base_55k_cfg1.0", "/home/ds/Workspace/DiT/assets/results/unet_connectivity_100000/baseline_resume/20261018-030921-Unet-flow/checkpoints/0055000.pt", False, 1.0),
        ("Base_55k_cfg1.4", "/home/ds/Workspace/DiT/assets/results/unet_connectivity_100000/baseline_resume/20261018-030921-Unet-flow/checkpoints/0055000.pt", False, 1.4),
        ("Gate_75k_cfg1.0", "/home/ds/Workspace/DiT/assets/results/unet_dic/B_residual_gate/20261018-064328-Unet-flow/checkpoints/0075000.pt", False, 1.0),
        ("Gate_75k_cfg1.4", "/home/ds/Workspace/DiT/assets/results/unet_dic/B_residual_gate/20261018-064328-Unet-flow/checkpoints/0075000.pt", False, 1.4),
        ("DiC768_50k_cfg1.0", "/home/ds/Workspace/DiT/0050000_ema.pt", True, 1.0),
        ("DiC768_50k_cfg1.4", "/home/ds/Workspace/DiT/0050000_ema.pt", True, 1.4),
    ]

    all_swap_results = {}

    for mod_name, ckpt_p, is_4090, cfg_val in model_targets:
        print(f"\n========================================================")
        print(f"Running Writer Swap for: {mod_name} (CFG={cfg_val})")
        print(f"========================================================")
        model, config = build_model(mod_name, ckpt_p, device, is_4090=is_4090)
        diff = diffusion(config)

        swap_noises = []
        swap_skels = []
        swap_callig = []
        swap_script = []
        swap_meta = []

        for item_i, a_idx in enumerate(selected_indices):
            r = rows[a_idx]
            z_fixed = cache["noise"][a_idx:a_idx+1]
            g_fixed = cache["skels_latent"][a_idx:a_idx+1]
            s_name = r.get("script", "").strip()
            ch_name = r.get("character", "").strip()
            orig_cal = r.get("calligrapher", "").strip()
            s_raw_id = script_to_sid[s_name]
            s_remapped = s2s[str(s_raw_id)]

            for w_name in calligraphers:
                c_raw_id = name_to_cid[w_name]
                w_remapped = c2w[str(c_raw_id)]
                swap_noises.append(z_fixed)
                swap_skels.append(g_fixed)
                swap_callig.append(w_remapped)
                swap_script.append(s_remapped)
                swap_meta.append({
                    "anchor_idx": a_idx,
                    "target_writer": w_name,
                    "target_script": s_name,
                    "character": ch_name,
                    "is_self": (w_name == orig_cal)
                })

        N_total = len(swap_meta) # 100
        noises_cat = torch.cat(swap_noises, dim=0) # (100, 4, 32, 32)
        skels_cat = torch.cat(swap_skels, dim=0)   # (100, 1, 64, 64)
        conds_tuple = [(swap_callig[i], 0) for i in range(N_total)]
        hier_conds = [swap_callig, [0]*N_total, swap_script]

        with torch.no_grad():
            latents = sample_latents(
                model, diff, noises_cat, conds_tuple,
                cfg_val, 32, device,
                skel=skels_cat, seed=0,
                hier_conds=hier_conds
            )

            imgs_list = []
            for b_start in range(0, N_total, 16):
                b_lat = latents[b_start:b_start+16].to(device) / sf
                dec = vae.decode(b_lat).sample
                dec = (dec.clamp(-1, 1) + 1.0) / 2.0
                imgs_list.append(dec.cpu())
            imgs_tensor = torch.cat(imgs_list, dim=0)

        # 1. Content preservation (OCR agreement using char_head of eval_classifier)
        with torch.no_grad():
            ch_inputs = StyleEncoder.to_tensor([(t.permute(1,2,0).numpy()*255).astype(np.uint8) for t in imgs_tensor]).to(device)
            char_logits, _, _ = charenc.net(ch_inputs)
            pred_char_ids = char_logits.argmax(dim=-1).cpu().tolist()

        ocr_agreements = []
        for a_i in range(10):
            group_pred_chars = pred_char_ids[a_i*10 : (a_i+1)*10]
            anchor_self_pred = group_pred_chars[calligraphers.index(rows[selected_indices[a_i]].get("calligrapher"))]
            for p_c in group_pred_chars:
                ocr_agreements.append(p_c == anchor_self_pred)
        content_preservation_rate = np.mean(ocr_agreements) * 100.0

        # 2. Independent Writer Top-1 and Target Margin
        disc_inputs = []
        for i in range(N_total):
            arr = (imgs_tensor[i].permute(1,2,0).numpy() * 255).astype(np.uint8)
            im = Image.fromarray(arr).resize((160, 160), Image.BICUBIC)
            disc_inputs.append(np.asarray(im))
        
        disc_batch = torch.from_numpy(np.stack(disc_inputs)).permute(0, 3, 1, 2).float().to(device) / 255.0
        mean = torch.tensor((0.485, 0.456, 0.406), device=device).view(1, 3, 1, 1)
        std = torch.tensor((0.229, 0.224, 0.225), device=device).view(1, 3, 1, 1)
        disc_batch = (disc_batch - mean) / std

        with torch.no_grad():
            emb_gen = disc.emb(disc_batch).cpu().numpy()
            emb_gen = emb_gen / np.linalg.norm(emb_gen, axis=1, keepdims=True)
            sims = emb_gen @ P.T

        swap_top1_hits = []
        swap_margins = []
        counterfactual_flips = []

        for i in range(N_total):
            m = swap_meta[i]
            slot_key = m["target_writer"] + '·' + m["target_script"]
            if slot_key not in protos:
                continue
            ci = slot_list.index(slot_key)
            pred_slot = slot_list[int(np.argmax(sims[i]))]
            is_hit = (pred_slot == slot_key)
            swap_top1_hits.append(1 if is_hit else 0)

            others = [sims[i][k2] for k2 in range(len(slot_list))
                      if k2 != ci and slot_script[slot_list[k2]] == m["target_script"]]
            if others:
                swap_margins.append(float(sims[i][ci] - max(others)))

            if not m["is_self"]:
                counterfactual_flips.append(1 if is_hit else 0)

        # 3. Inter-writer stroke visual divergence (L1 difference between different writers for same char)
        divergences = []
        for a_i in range(10):
            group_imgs = imgs_tensor[a_i*10 : (a_i+1)*10]
            for w1 in range(10):
                for w2 in range(w1 + 1, 10):
                    l1_diff = float(torch.abs(group_imgs[w1] - group_imgs[w2]).mean())
                    divergences.append(l1_diff)
        mean_divergence = np.mean(divergences)

        swap_acc = np.mean(swap_top1_hits) * 100.0
        cf_acc = np.mean(counterfactual_flips) * 100.0
        margin_mean = np.mean(swap_margins)
        margin_med = np.median(swap_margins)

        print(f"[{mod_name}] Content Preserv: {content_preservation_rate:.1f}% | Swap Top1: {swap_acc:.1f}% | Counterfactual Flips: {cf_acc:.1f}% | Target Margin: {margin_mean:+.4f} | Visual Div (L1): {mean_divergence:.4f}")

        all_swap_results[mod_name] = {
            "content_preservation_rate": content_preservation_rate,
            "swap_top1_accuracy": swap_acc,
            "counterfactual_flip_rate": cf_acc,
            "target_margin_mean": margin_mean,
            "target_margin_median": margin_med,
            "visual_divergence_l1": mean_divergence
        }

    out_p = ROOT / "assets/results/cfg_sweep/writer_swap_results.json"
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(all_swap_results, f, ensure_ascii=False, indent=2)
    print("\nSaved writer swap results to:", out_p)

if __name__ == "__main__":
    main()
