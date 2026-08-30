# -*- coding: utf-8 -*-
"""build_fame_fast.py — fame 数据集快速版: CSV + GPU encode (极性检查/skel 后续补)."""
import os
import sys
import csv
import json
import collections

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

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
SCRIPT_ID = {"篆": 1, "隶": 4, "楷": 0, "行": 3, "草": 2}
NUM_CHARACTERS = 7026


def canon(n):
    return MERGE.get(n, n)


class ImgDS(Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        import torchvision.transforms as T
        tf = T.Compose([T.Resize((256, 256)), T.ToTensor(),
                        T.Normalize([0.5] * 3, [0.5] * 3)])
        return tf(Image.open(self.paths[i]).convert("RGB"))


def main():
    man = json.load(open("archive/final_manifest.json", encoding="utf-8"))
    sel = set(NAMES)
    entries = [e for e in man if canon(e.get("orig_calli", "")) in sel]
    print(f"[fame] entries {len(entries)}", flush=True)

    callig_map, char_map = {}, {}
    for csvp in ["5script/train_mid_clean.csv", "5script/train_top6.csv",
                 "5script/train_3top30_nobeike.csv", "5script/train.csv"]:
        if not os.path.exists(csvp):
            continue
        for r in csv.DictReader(open(csvp, encoding="utf-8")):
            callig_map.setdefault(r["calligrapher"], int(r["calligrapher_id"]))
            char_map.setdefault((r["script"], r["character"]), int(r["character_id"]))
    used_c = set(callig_map.values())
    next_c = 0
    for n in sorted({canon(e["orig_calli"]) for e in entries}):
        if n not in callig_map:
            while next_c in used_c:
                next_c += 1
            callig_map[n] = next_c
            used_c.add(next_c)
            print(f"[id] new calligrapher {n} -> {next_c}", flush=True)
    # character_id 是"每书体独立"的 0..7025 空间 (glyph_id = sid*7026+cid)
    used_ch = collections.defaultdict(set)
    for (sc, ch), cid in char_map.items():
        used_ch[sc].add(cid)
    next_ch = collections.Counter()
    for n, e in enumerate(entries):
        key = (e["orig_script"], e["orig_char"])
        if key not in char_map:
            sc = e["orig_script"]
            cid = next_ch[sc]
            while cid in used_ch[sc] or cid >= NUM_CHARACTERS:
                cid += 1
            next_ch[sc] = cid
            char_map[key] = cid
            used_ch[sc].add(cid)
        if (n + 1) % 20000 == 0:
            print(f"[id] chars {n+1}/{len(entries)}", flush=True)
    print(f"[id] chars {len(char_map)} (new {len(char_map) - sum(len(v) for k, v in [] ) or 0})", flush=True)

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
    # ---- 组合泛化 holdout: 从 fame 池里抽 (书家,字) 组合, 每组扣 1 张进 eval ----
    import random
    random.seed(42)
    combos = collections.defaultdict(list)
    for r in out_rows:
        combos[(r["calligrapher"], r["character"])].append(r)
    keys = [k for k, v in combos.items() if len(v) >= 2]
    random.shuffle(keys)
    eval_rows = []
    for k in keys:
        if len(eval_rows) >= 500:
            break
        eval_rows.append(combos[k].pop())
    out_rows = [r for v in combos.values() for r in v]
    eval_ids = {r["image_path"] for r in eval_rows}
    train_rows = [r for r in out_rows if r["image_path"] not in eval_ids]
    with open("5script/train_fame.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(train_rows)
    with open("5script/eval_fame_strict.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(eval_rows)
    print(f"[csv] train_fame.csv {len(train_rows)} rows | eval_fame_strict.csv {len(eval_rows)} rows", flush=True)

    # ---- GPU encode (DataLoader 32 worker 解码, batch 96) ----
    from diffusers.models import AutoencoderKL
    dev = torch.device("cuda")
    vae = AutoencoderKL.from_pretrained("pretrained_models/sd-vae-ft-ema").to(dev).eval()
    paths = sorted(set([r["image_path"] for r in out_rows] +
                       [r["image_path"] for r in eval_rows]))
    ds = ImgDS(paths)
    dl = DataLoader(ds, batch_size=96, num_workers=32, prefetch_factor=4)
    NSHARD = 20
    per = (len(paths) + NSHARD - 1) // NSHARD
    os.makedirs("final_latents_fame", exist_ok=True)
    all_ids = torch.tensor([int(p.split("/")[-1][:-4]) for p in paths],
                           dtype=torch.int64)
    with torch.no_grad():
        i = 0
        for xb in dl:
            lat = (vae.encode(xb.to(dev)).latent_dist.mode() * 0.18215).half().cpu().numpy()
            for s in range(NSHARD):
                lo, hi = s * per, min((s + 1) * per, len(paths))
                if i < hi and i + lat.shape[0] > lo:
                    lo2, hi2 = max(i, lo), min(i + lat.shape[0], hi)
                    np.save(f"final_latents_fame/.lat_{s}_{lo2}.npy",
                            lat[lo2 - i:hi2 - i])
            i += lat.shape[0]
            if i % 4800 == 0:
                print(f"[encode] {i}/{len(paths)}", flush=True)
    # 合并碎片 -> shards + fame.npz
    all_lat = np.empty((len(paths), 4, 32, 32), dtype=np.float16)
    for f in glob.glob("final_latents_fame/.lat_*.npy"):
        name = os.path.basename(f)[5:-4]  # {s}_{lo}
        s, lo = name.split("_")
        arr = np.load(f)
        s = int(s); lo = int(lo)
        all_lat[lo:lo + arr.shape[0]] = arr
        os.remove(f)
    for s in range(NSHARD):
        lo, hi = s * per, min((s + 1) * per, len(paths))
        np.savez_compressed(f"final_latents_fame/shard_{s:04d}.npz",
                            latents=all_lat[lo:hi], img_ids=all_ids[lo:hi].numpy())
    np.savez_compressed("fame.npz", latents=all_lat, img_ids=all_ids.numpy())
    json.dump({"n_train": len(train_rows), "n_eval": len(eval_rows),
               "people": len({r["calligrapher"] for r in out_rows}),
               "note": "polarity/skel/eval-img 待后续补充"},
              open("fame_meta.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("[DONE] fame.npz + shards", flush=True)


if __name__ == "__main__":
    main()
