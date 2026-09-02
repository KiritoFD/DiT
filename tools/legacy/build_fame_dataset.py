# -*- coding: utf-8 -*-
"""
build_fame_dataset.py — 构建 fame（名家）数据集.

输出:
  5script/train_fame.csv              # 合并书家名后的训练表 (原 id 空间, 兼容 s20 warm-start)
  5script/eval_fame_strict.csv        # 组合泛化 eval (书家要素+字要素覆盖, (书家,字)组合未见过)
  final_latents_fame/shard_*.npz      # img VAE latents (f16 4x32x32)
  fame.npz                            # 全量合并版 (latents f16 + img_ids)
  final_skel_latents_fame/shard_*.npz # skel latents (ctrl 条件)
  fame_meta.json                      # 统计 + 翻转清单 + name merge 记录
极性: 全部白底黑字 (ink>0.5 的翻转, 原图备份 flip_backup_fame/)
"""
import os
import sys
import csv
import json
import shutil
import collections

import numpy as np

ROOT = "/root/Workspace/xy/DiT"
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.stdout.reconfigure(encoding="utf-8")

MERGE = {"赵孟": "赵孟頫", "郑燮": "郑板桥", "孫過庭": "孙过庭"}
NAMES = ["赵孟頫", "王羲之", "苏轼", "欧阳询", "颜真卿", "智永", "米芾", "褚遂良", "孙过庭",
         "黄庭坚", "文徵明", "怀素", "李阳冰", "祝允明", "何绍基", "赵佶", "赵之谦", "邓石如",
         "吴昌硕", "李邕", "王铎", "赵构", "金农", "沈尹默", "柳公权", "鲜于枢", "王献之",
         "董其昌", "唐寅", "欧阳通", "蔡襄", "虞世南", "李斯", "薛稷", "李世民", "朱耷",
         "张旭", "乾隆", "伊秉绶", "钟繇", "于右任", "郑板桥", "黄易", "武则天"]
NUM_CHARACTERS = 7026  # 每 script 的 char id 空间 (glyph_id = sid*7026+cid)
SCRIPT_ID = {"篆": 1, "隶": 4, "楷": 0, "行": 3, "草": 2}

SEED = 42


def canon(n):
    return MERGE.get(n, n)


def main():
    import torch
    from PIL import Image
    from diffusers.models import AutoencoderKL
    man = json.load(open("archive/final_manifest.json", encoding="utf-8"))

    # ---- 1) 选择 + 合并 ----
    sel = set(NAMES)
    entries = [e for e in man if canon(e.get("orig_calli", "")) in sel]
    print(f"[fame] entries: {len(entries)}")

    # ---- 2) name->id 映射 (沿用已有 csv 的 id; 缺失的分配空位) ----
    callig_map, char_map = {}, {}   # name -> id, (script,char)->id
    for csvp in ["5script/train_mid_clean.csv", "5script/train_top6.csv",
                 "5script/train_3top30_nobeike.csv", "5script/train.csv"]:
        if not os.path.exists(csvp):
            continue
        for r in csv.DictReader(open(csvp, encoding="utf-8")):
            callig_map.setdefault(r["calligrapher"], int(r["calligrapher_id"]))
            char_map.setdefault((r["script"], r["character"]), int(r["character_id"]))
    used_c = set(callig_map.values())
    next_c = max(used_c) + 1 if used_c else 0
    for n in sorted({canon(e["orig_calli"]) for e in entries}):
        if n not in callig_map:
            while next_c in used_c:
                next_c += 1
            callig_map[n] = next_c
            used_c.add(next_c)
            print(f"[id] new calligrapher {n} -> {next_c}")
    used_ch = set(char_map.values())
    next_ch = 0
    dino_idx = json.load(open("pretrained_models/dino_embeddings/glyph_dino_index.json", encoding="utf-8"))
    dino_pairs = set(tuple(x) for x in dino_idx.get("glyphs", []))
    for e in entries:
        key = (e["orig_script"], e["orig_char"])
        if key not in char_map:
            if key in dino_pairs:
                # 用 DINO 表里的 id (保证 embedding 有意义)
                char_map[key] = None  # 占位, 下面从 dino 反查
            else:
                while (next_ch in used_ch) or (next_ch >= NUM_CHARACTERS):
                    next_ch += 1
                char_map[key] = next_ch
                used_ch.add(next_ch)
    # dino 反查
    pair2dino = {}
    for gi, (sid, cid) in enumerate(dino_idx.get("glyphs", [])):
        pair2dino[(sid, cid)] = cid
    n_dino = n_new = 0
    for e in entries:
        key = (e["orig_script"], e["orig_char"])
        if char_map.get(key) is None:
            sid = SCRIPT_ID.get(e["orig_script"], 0)
            char_map[key] = pair2dino.get((sid, int(key[1] and 0) or 0), None)
    # dino index 只存 (sid,cid) 不含 char 文本, 无法反查文本 -> 直接顺序分配新 id
    for e in entries:
        key = (e["orig_script"], e["orig_char"])
        if char_map.get(key) is None:
            while (next_ch in used_ch) or (next_ch >= NUM_CHARACTERS):
                next_ch += 1
            char_map[key] = next_ch
            used_ch.add(next_ch)
            n_new += 1
    print(f"[id] chars total {len(char_map)}, newly allocated {n_new}")

    # ---- 3) 极性检查/翻转 + 翻转清单 ----
    os.makedirs("flip_backup_fame", exist_ok=True)
    import multiprocessing as mp
    def _check_ink(img_id):
        p = f"final_imgs_256/{img_id}.png"
        a = np.asarray(Image.open(p).convert("L").resize((64, 64)), np.float32) / 255.
        return img_id, bool((a < 0.5).mean() > 0.5)
    ids_all = [e["img_id"] for e in entries]
    flip = []
    with mp.Pool(min(64, os.cpu_count() or 32)) as pool:
        for n, (iid, is_inv) in enumerate(pool.imap_unordered(_check_ink, ids_all, chunksize=64)):
            if is_inv:
                flip.append(iid)
            if (n + 1) % 10000 == 0:
                print(f"[polarity] scanned {n+1}/{len(ids_all)}, flipped {len(flip)}", flush=True)
    print(f"[polarity] flipped {len(flip)}", flush=True)
    for iid in flip:
        p = f"final_imgs_256/{iid}.png"
        bk = f"flip_backup_fame/{iid}.png"
        if not os.path.exists(bk):
            os.replace(p, bk)
        im = Image.open(bk)
        Image.fromarray(255 - np.asarray(im.convert("L")), "L").convert(im.mode).save(p)
    print("[polarity] flips applied", flush=True)

    # ---- 4) train csv ----
    out_rows = []
    for e in entries:
        name = canon(e["orig_calli"])
        sid = SCRIPT_ID.get(e["orig_script"], 0)
        cid = char_map[(e["orig_script"], e["orig_char"])]
        out_rows.append({
            "image_path": f"final_imgs_256/{e['img_id']}.png",
            "calligrapher": name, "script": e["orig_script"], "character": e["orig_char"],
            "calligrapher_id": callig_map[name], "script_id": sid,
            "character_id": cid, "glyph_id": sid * NUM_CHARACTERS + cid,
        })
    with open("5script/train_fame.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    print(f"[csv] train_fame.csv: {len(out_rows)} rows")

    # ---- 5) VAE encode -> shards + fame.npz ----
    dev = torch.device("cuda")
    vae = AutoencoderKL.from_pretrained("pretrained_models/sd-vae-ft-ema").to(dev).eval()
    os.makedirs("final_latents_fame", exist_ok=True)
    BS = 48
    all_lat = np.empty((len(out_rows), 4, 32, 32), dtype=np.float16)
    all_ids = np.array([e["img_id"] for e in entries], dtype=np.int64)
    with torch.no_grad():
        for i in range(0, len(out_rows), BS):
            batch = out_rows[i:i + BS]
            x = torch.stack([tf_encode(Image.open(r["image_path"]).convert("RGB")) for r in batch]).to(dev)
            lat = (vae.encode(x).latent_dist.mode() * 0.18215).half().cpu().numpy()
            all_lat[i:i + len(batch)] = lat
            if (i // BS) % 100 == 0:
                print(f"[encode] {i}/{len(out_rows)}", flush=True)
    np.savez_compressed("fame.npz", latents=all_lat, img_ids=all_ids)
    NSHARD = 20
    per = (len(out_rows) + NSHARD - 1) // NSHARD
    for s in range(NSHARD):
        lo, hi = s * per, min((s + 1) * per, len(out_rows))
        np.savez_compressed(f"final_latents_fame/shard_{s:04d}.npz",
                            latents=all_lat[lo:hi], img_ids=all_ids[lo:hi])
    print(f"[encode] fame.npz + {NSHARD} shards done")

    # ---- 6) skel -> skel latents shards ----
    try:
        from skimage.morphology import skeletonize
    except ImportError:
        from scipy.ndimage import binary_erosion, generate_binary_structure
        def skeletonize(b):
            skel = np.zeros_like(b); img = b.copy()
            st = generate_binary_structure(2, 2)
            while img.any():
                er = binary_erosion(img, structure=st)
                skel |= img & ~er; img = er
            return skel
    from scipy.ndimage import binary_dilation, generate_binary_structure
    def dil3(b):
        return binary_dilation(b, structure=generate_binary_structure(2, 2), iterations=3)
    os.makedirs("final_skel3_fame", exist_ok=True)
    skel_lat = np.empty((len(out_rows), 4, 32, 32), dtype=np.float16)
    with torch.no_grad():
        for i in range(0, len(out_rows), BS):
            batch = out_rows[i:i + BS]
            sks = []
            for r in batch:
                iid = r["image_path"].split("/")[-1]
                a = np.asarray(Image.open(r["image_path"]).convert("L"))
                sk = skeletonize(a < 127)
                a3 = np.where(dil3(sk), 0, 255).astype("uint8")
                Image.fromarray(a3, "L").save(f"final_skel3_fame/{iid}")
                sks.append(a3)
            x = torch.from_numpy(np.stack(sks).astype(np.float32) / 255. * 2 - 1)[:, None].repeat(1, 3, 1, 1).to(dev)
            skel_lat[i:i + len(batch)] = (vae.encode(x).latent_dist.mode() * 0.18215).half().cpu().numpy()
            if (i // BS) % 100 == 0:
                print(f"[skel] {i}/{len(out_rows)}", flush=True)
    os.makedirs("final_skel_latents_fame", exist_ok=True)
    for s in range(NSHARD):
        lo, hi = s * per, min((s + 1) * per, len(out_rows))
        np.savez_compressed(f"final_skel_latents_fame/shard_{s:04d}.npz",
                            latents=skel_lat[lo:hi], img_ids=all_ids[lo:hi])
    print("[skel] shards done")

    # ---- 7) fame 严格 eval (组合泛化) ----
    import random
    random.seed(SEED)
    train_ids = {e["img_id"] for e in entries}
    train_pairs = set((r["calligrapher"], r["character"]) for r in out_rows)
    cand = [e for e in man
            if e["img_id"] not in train_ids
            and canon(e.get("orig_calli", "")) in sel
            and (canon(e["orig_calli"]), e["orig_char"]) not in train_pairs
            and e["orig_script"] in SCRIPT_ID]
    random.shuffle(cand)
    eval_rows = []
    seen_glyph = set()
    for e in cand:
        key = (canon(e["orig_calli"]), e["orig_char"])
        if key in seen_glyph:
            continue
        seen_glyph.add(key)
        name = canon(e["orig_calli"])
        sid = SCRIPT_ID[e["orig_script"]]
        cid = char_map[(e["orig_script"], e["orig_char"])]
        eval_rows.append({
            "image_path": f"final_imgs_256/{e['img_id']}.png",
            "calligrapher": name, "script": e["orig_script"], "character": e["orig_char"],
            "calligrapher_id": callig_map[name], "script_id": sid,
            "character_id": cid, "glyph_id": sid * NUM_CHARACTERS + cid,
        })
        if len(eval_rows) >= 500:
            break
    with open("5script/eval_fame_strict.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(eval_rows)
    print(f"[eval] eval_fame_strict.csv: {len(eval_rows)} rows")

    meta = {
        "n_train": len(out_rows), "n_eval": len(eval_rows),
        "people": len({r["calligrapher"] for r in out_rows}),
        "uniq_chars": len(set((r["script"], r["character"]) for r in out_rows)),
        "flipped": len(flip), "flip_ids": flip,
        "merge": MERGE,
        "new_calligrapher_ids": {n: i for n, i in callig_map.items() if i not in used_c or True and False},
    }
    json.dump(meta, open("fame_meta.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("[done]", {k: meta[k] for k in ("n_train", "n_eval", "people", "uniq_chars", "flipped")})


def tf_encode(im):
    import torchvision.transforms as T
    return T.Compose([
        T.Resize((256, 256), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(), T.Normalize([0.5] * 3, [0.5] * 3)])(im)


if __name__ == "__main__":
    main()
