#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""fs6_stylegrid.py — "同一个字 × 多个书家的风格行" 网格（few-shot 最直观的可视化）。

做法：把 v15a@150k 的 87 行已训表 + 5 个 few-shot 新行(87..91) 拼成一张 92 行表，
烘进一个 ckpt 副本（model/ema 一起换，args.num_calligraphers=92），再用 batch_eval
对同一批字、同一个骨架采样 7 种风格：5 个新书家(零训练数据的行) + 2 个训过的 pair 作参照。
注意 y_callig 这里直接用**原始行号**(batch_eval 不传 callig_script_map -> raw id 直查表)。

产物: /tmp/_stylegrid/run1/eval_samples_ctrl/**/g*.png (行序 == CSV 行序)
"""
import csv
import glob
import json
import os
import subprocess
import sys

import numpy as np
import torch

ROOT = "/root/Workspace/xy/DiT"
os.chdir(ROOT)
sys.path.insert(0, ROOT)

CK = sorted(glob.glob("assets/results/v15a_multistyle_k4/*/checkpoints/0150000.pt"))[-1]
TBL_KEY = "y_callig_embedder.embedding_table.weight"
TOPICS = ["沈周-行", "伊秉绶-行", "傅山-行", "伊秉绶-隶", "徐渭-行"]
CHARS = ["佛", "倚", "偶"]          # 沈周-行 eval 里的字（骨架借同一本）
OUTDIR = "/tmp/_stylegrid/run1"
N_NEW = len(TOPICS)

# ── 1) 拼 92 行表 ─────────────────────────────────────────────────────────────
d = torch.load(CK, map_location="cpu", weights_only=False)
sd = d.get("delta") or d.get("model") or d
# torch.compile 存盘带 _orig_mod. 前缀（batch_eval 载入时会自己剥掉），补丁要打在真实键上
key = TBL_KEY if TBL_KEY in sd else "_orig_mod." + TBL_KEY
tbl = sd[key].float()
assert tuple(tbl.shape) == (87, 1536), tbl.shape
rows_new = [torch.load(f"assets/fs6_row_{t}.pt", map_location="cpu",
                       weights_only=False)["embedding"].float().reshape(-1)
            for t in TOPICS]
combined = torch.cat([tbl, torch.stack(rows_new)], 0)          # (92,1536)
print(f"[tbl] {tuple(tbl.shape)} + {N_NEW} 行 -> {tuple(combined.shape)}")

sd[key] = combined.to(sd[key].dtype)
if isinstance(d.get("ema"), dict):
    ek = key if key in d["ema"] else ("_orig_mod." + TBL_KEY)
    if ek in d["ema"]:
        d["ema"][ek] = combined.to(d["ema"][ek].dtype)
ar = d.get("args")
if isinstance(ar, dict):
    ar["num_calligraphers"] = 87 + N_NEW
    ar["callig_emb_pretrained"] = ""       # 否则 apply_post_construction 会拿 87 行表断言炸掉
else:
    setattr(ar, "num_calligraphers", 87 + N_NEW)
    setattr(ar, "callig_emb_pretrained", "")
os.makedirs(f"{OUTDIR}/checkpoints", exist_ok=True)
ck_out = f"{OUTDIR}/checkpoints/0150000.pt"
torch.save(d, ck_out)
print(f"[ckpt] {ck_out}")

# ── 2) 两个"训过的 pair"作参照（米芾-行 + 董其昌-行） ─────────────────────────
rows50 = list(csv.DictReader(open("assets/train_50k_v2.csv", encoding="utf-8")))
id2name = {r["calligrapher_id"]: r["calligrapher"] for r in rows50}
m = json.load(open("assets/callig_script_id_map.json", encoding="utf-8"))
pid = {}
for k in m["pair_map"]:
    nm, sid = id2name.get(k.split(":")[0], "?"), k.split(":")[1]
    pid[(nm, sid)] = m["pair_map"][k]
ref = [(f"米芾-行(全量训)", pid[("米芾", "3")]), (f"董其昌-行(全量训)", pid[("董其昌", "3")])]
styles = [(t, 87 + i) for i, t in enumerate(TOPICS)] + ref
print("[styles]", styles)

# ── 3) CSV + 骨架 shard（3 个字 × 7 种风格，同一骨架重复用） ──────────────────
ev = {r["character"]: r for r in csv.DictReader(
    open("assets/fs6_沈周-行_eval.csv", encoding="utf-8"))}
for ch in CHARS:
    assert ch in ev, f"{ch} 不在沈周-行 eval 里"


csv_rows, paths, ids, names = [], [], [], []
i = 0
for sname, srow in styles:
    for ch in CHARS:
        r = ev[ch]
        csv_rows.append({
            "image_path": r["image_path"], "calligrapher": sname,
            "script": r["script"], "character": ch,
            "calligrapher_id": str(srow), "script_id": r["script_id"],
            "character_id": r["character_id"], "glyph_id": r["glyph_id"],
            "aug": "", "std_path": r["std_path"], "source": "stylegrid",
            "src_image_path": r["image_path"], "old_50k_id": "",
            "img_id": str(960000 + i),
        })
        paths.append(r["std_path"])
        ids.append(960000 + i)
        names.append(os.path.basename(r["std_path"]))
        i += 1
with open("assets/fs6_stylegrid.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
    w.writeheader()
    w.writerows(csv_rows)

# 骨架 latent 不重新编码（GPU 被 v15b 占满）：直接从 沈周-行 的主题 std shard 里复制
zmap = {}
for sp in glob.glob("data/50k/shards_fs6_沈周-行_std/shard_*.npz"):
    with np.load(sp) as z:
        for j, iid in enumerate(z["img_ids"]):
            zmap[int(iid)] = z["latents"][j]
lat = np.stack([zmap[int(ev[r["character"]]["img_id"])] for r in csv_rows]).astype(np.float16)
os.makedirs("data/50k/shards_fs6_stylegrid", exist_ok=True)
np.savez_compressed("data/50k/shards_fs6_stylegrid/shard_00000.npz",
                    latents=lat, img_ids=np.array(ids, dtype=np.int64),
                    names=np.array(names, dtype="U64"))
print(f"[shard] {lat.shape} -> data/50k/shards_fs6_stylegrid (复制自 沈周-行_std, 无 GPU)")

# ── 4) batch_eval 采样（显存按最小配给：dit 4 / vae 2） ───────────────────────
cmd = ["/opt/conda/envs/cu121/bin/python", "-u", "src/eval/batch_eval.py",
       "--results-dir", OUTDIR, "--ckpt-override", ck_out,
       "--sets", f"grid:assets/fs6_stylegrid.csv:{len(csv_rows)}",
       "--skel-shards", "data/50k/shards_fs6_stylegrid",
       "--save-samples", "--dit-batch", "4", "--vae-batch", "2",
       "--device", "cpu",              # GPU 被 v15b 占满(21.4G)，CPU 慢但不打扰
       "--force"]
print("[run]", " ".join(cmd))
rc = subprocess.run(cmd).returncode
outs = sorted(glob.glob(f"{OUTDIR}/**/grid/g*.png", recursive=True))
print(f"[done] rc={rc}  {len(outs)} 张 -> {OUTDIR}/**/grid/g*.png")
for i, r in enumerate(csv_rows):
    print(f"  g{i}.png <- {r['calligrapher']}  字={r['character']}")
