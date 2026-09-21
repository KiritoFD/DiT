"""eval 集交叉验证 + 生成修复后的 eval csv。"""
import csv
import os
import shutil

from opencc import OpenCC

os.chdir("/root/Workspace/xy/DiT")
s2t = OpenCC("s2t")

o = {r["image_path"]: r for r in
     csv.DictReader(open("assets/_eval_ocr.csv", encoding="utf-8"))}
v = {r["image_path"]: r for r in
     csv.DictReader(open("assets/_eval_vlm.csv", encoding="utf-8"))}

strong, only_o, only_v = [], [], []
for ip, oo in o.items():
    a = oo["char_csv"]
    t = s2t.convert(a)
    if t == a:
        continue
    vv = v.get(ip)
    o_says = (oo["char_ocr"] == t)
    v_says = (vv is not None and vv["char_vlm"] == t)
    if o_says and v_says:
        strong.append((oo, vv))
    elif o_says:
        only_o.append(oo)
    elif v_says:
        only_v.append((oo, vv))

print("  === eval 集交叉验证 ===")
print(f"    ★★ OCR+VLM 都指向繁体: {len(strong)}")
print(f"       仅 OCR: {len(only_o)}   仅 VLM: {len(only_v)}")

print("\n  === 强确认明细 ===")
for oo, vv in strong:
    print(f"    {oo['char_csv']} -> {s2t.convert(oo['char_csv'])}  "
          f"ocr={oo['char_ocr']}({oo['conf']}) vlm={vv['char_vlm']}  "
          f"[{oo['script']}/{oo['calligrapher']}]")

print("\n  === 仅 OCR 线索 ===")
for oo in only_o:
    print(f"    {oo['char_csv']} -> {s2t.convert(oo['char_csv'])}  "
          f"conf={oo['conf']}  [{oo['script']}/{oo['calligrapher']}]")

print("\n  === 仅 VLM 线索 ===")
for oo, vv in only_v:
    print(f"    {oo['char_csv']} -> {vv['char_vlm']}  "
          f"[{oo['script']}/{oo['calligrapher']}]")

# 修复 eval 集（强确认 + 可选宽松）
fix = {oo["image_path"]: s2t.convert(oo["char_csv"]) for oo, _ in strong}

for src in ("assets/eval_v13_strict.csv", "assets/eval_v13_seen.csv"):
    if not os.path.exists(src):
        continue
    dst = src.replace(".csv", "_fixed.csv")
    rows = list(csv.DictReader(open(src, encoding="utf-8")))
    std_of = {r["character"]: r["std_path"] for r in rows}
    # std 需要从训练集查（eval 里可能没有该繁体字的 std）
    tr = list(csv.DictReader(open("assets/train_50k_v2.csv", encoding="utf-8")))
    for r in tr:
        std_of.setdefault(r["character"], r["std_path"])
    cols = list(rows[0].keys())
    n = 0
    with open(dst, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            t = fix.get(r["image_path"])
            if t and t in std_of:
                r["character"] = t
                r["std_path"] = std_of[t]
                n += 1
            w.writerow(r)
    print(f"\n  ✓ {dst}: 修了 {n} 行")
