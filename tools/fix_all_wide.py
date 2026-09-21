"""按**宽松版**修复训练集和 eval 集（含单模型线索）。

宽松版 = 强确认(OCR&VLM) + 仅OCR + 仅VLM
用户要求: "先修复，按照宽松版本的都改掉"
"""
import csv
import os

from opencc import OpenCC

os.chdir("/root/Workspace/xy/DiT")
s2t = OpenCC("s2t")

# ── 训练集 ────────────────────────────────────────────────────────────
wide = list(csv.DictReader(open("assets/mismatch_wide.csv", encoding="utf-8")))
print(f"  宽松版训练集清单: {len(wide)}")
fix_tr = {r["image_path"]: r["char_true"] for r in wide}

rows = list(csv.DictReader(open("assets/train_50k_v2.csv", encoding="utf-8")))
std_of = {}
for r in rows:
    std_of.setdefault(r["character"], r["std_path"])

out = "assets/train_50k_v2_fixed.csv"
cols = list(rows[0].keys())
n = miss = 0
with open(out, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    for r in rows:
        t = fix_tr.get(r["image_path"])
        if t and t in std_of:
            r["character"] = t
            r["std_path"] = std_of[t]
            n += 1
        elif t:
            miss += 1
        w.writerow(r)
print(f"  ✓ {out}: 修 {n} 行, 跳过 {miss} 行(无 std)")

# ── eval 集 ───────────────────────────────────────────────────────────
o = {r["image_path"]: r for r in
     csv.DictReader(open("assets/_eval_ocr.csv", encoding="utf-8"))}
v = {r["image_path"]: r for r in
     csv.DictReader(open("assets/_eval_vlm.csv", encoding="utf-8"))}

fix_ev = {}
for ip, oo in o.items():
    a = oo["char_csv"]
    t = s2t.convert(a)
    if t == a:
        continue
    vv = v.get(ip)
    if oo["char_ocr"] == t or (vv is not None and vv["char_vlm"] == t):
        fix_ev[ip] = t
print(f"\n  宽松版 eval 清单: {len(fix_ev)}")

# std 映射（含训练集）
std_all = dict(std_of)
for src in ("assets/eval_v13_strict.csv", "assets/eval_v13_seen.csv"):
    if os.path.exists(src):
        for r in csv.DictReader(open(src, encoding="utf-8")):
            std_all.setdefault(r["character"], r["std_path"])

for src in ("assets/eval_v13_strict.csv", "assets/eval_v13_seen.csv"):
    if not os.path.exists(src):
        continue
    dst = src.replace(".csv", "_fixed.csv")
    rs = list(csv.DictReader(open(src, encoding="utf-8")))
    cols2 = list(rs[0].keys())
    m = m2 = 0
    with open(dst, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols2)
        w.writeheader()
        for r in rs:
            t = fix_ev.get(r["image_path"])
            if t and t in std_all:
                r["character"] = t
                r["std_path"] = std_all[t]
                m += 1
            elif t:
                m2 += 1
            w.writerow(r)
    print(f"  ✓ {dst}: 修 {m} 行" + (f", 跳过 {m2}" if m2 else ""))

# ── 汇总 ──────────────────────────────────────────────────────────────
print(f"\n  === 汇总 ===")
a = list(csv.DictReader(open("assets/train_50k_v2.csv", encoding="utf-8")))
b = list(csv.DictReader(open("assets/train_50k_v2_fixed.csv",
                             encoding="utf-8")))
print(f"    训练集: {sum(1 for x, y in zip(a, b) if x['character'] != y['character'])}"
      f" 行 character 变化 / {len(a)}")
for src in ("assets/eval_v13_strict.csv", "assets/eval_v13_seen.csv"):
    if not os.path.exists(src):
        continue
    x = list(csv.DictReader(open(src, encoding="utf-8")))
    y = list(csv.DictReader(open(src.replace(".csv", "_fixed.csv"),
                                 encoding="utf-8")))
    print(f"    {os.path.basename(src)}: "
          f"{sum(1 for i, j in zip(x, y) if i['character'] != j['character'])}"
          f" 行变化 / {len(x)}")
