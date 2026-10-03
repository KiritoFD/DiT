"""校验 exp-std/csv/eval200.csv 与 exp-std/data/eval_real200_cache.pt 是同一批 200 条。"""
import csv
import os

import torch as th

ROOT = "/root/Workspace/xy/DiT"
os.chdir(ROOT)
CSV = "exp-std/csv/eval200.csv"
CACHE = "exp-std/data/eval_real200_cache.pt"

c = th.load(CACHE, map_location="cpu", weights_only=False)
cid = {int(i) for i in c["img_ids"]}
with open(CSV, encoding="utf-8") as f:
    rd = csv.DictReader(f)
    fns = rd.fieldnames
    rows = list(rd)
sid = {int(r["img_id"]) for r in rows}
print(f"[csv]  {CSV} n={len(rows)} 列={fns}")
print(f"[cache] n={len(cid)}")
print(f"[一致] 交集 {len(cid & sid)}  |  csv 独有 {len(sid - cid)}  cache 独有 {len(cid - sid)}")
print(f"[判定] {'✅ 同一批' if cid == sid else '❌ 不是同一批 —— 换 csv'}")
tr = "exp-std/csv/train.csv"
with open(tr, encoding="utf-8") as f:
    trid = {int(r["img_id"]) for r in csv.DictReader(f)}
print(f"[train] n={len(trid)} | 与 eval200 重叠 {len(trid & sid)} (必须为 0)")
