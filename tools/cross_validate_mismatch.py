"""最终确认：用 OCR 与 VLM 的**双模型交叉验证**筛出高可信的简繁错配。

## 为什么要交叉
单模型都有错认（rapidocr 41% / VLM 44% 与 csv 一致）。
但如果 **两个独立模型**（一个是 OCR，一个是 VLM）都认为
GT 是 csv 的繁体版，那几乎不可能是巧合 -> 强确认。

判据:
  X = s2t(char_csv)，X != char_csv
  OCR 输出 == X  且  VLM 输出 == X   -> 强确认 ✓
  只有其中一个 == X                    -> 弱（单模型可能错）
"""
import csv
from collections import Counter, defaultdict

from opencc import OpenCC

s2t = OpenCC("s2t")

# 读 OCR 全量
ocr = {r["image_path"]: r for r in
       csv.DictReader(open("assets/ocr_gpu_scan.csv", encoding="utf-8"))}
print(f"  OCR 全量: {len(ocr)}")

# 读 VLM 结果（低置信度 + 候选）
vlm = {}
for f in ("assets/vlm_7b_lowconf.csv", "assets/vlm_7b_mismatch.csv"):
    try:
        for r in csv.DictReader(open(f, encoding="utf-8")):
            vlm[r["image_path"]] = r
    except FileNotFoundError:
        pass
print(f"  VLM 结果: {len(vlm)}")

strong = []   # OCR 和 VLM 都指向繁体
ocr_only = []
vlm_only = []
for ip, o in ocr.items():
    a = o["char_csv"]
    t = s2t.convert(a)
    if t == a:
        continue          # csv 本身就是繁体，不存在"标简体但写繁体"
    v = vlm.get(ip)
    o_says = (o["char_ocr"] == t)
    v_says = (v is not None and v["char_vlm"] == t)
    if o_says and v_says:
        strong.append((o, v))
    elif o_says:
        ocr_only.append(o)
    elif v_says:
        vlm_only.append((o, v))

print(f"\n  === 交叉验证结果 ===")
print(f"    ★★ OCR + VLM 都指向繁体: {len(strong)}")
print(f"       仅 OCR:                {len(ocr_only)}")
print(f"       仅 VLM:                {len(vlm_only)}")

print(f"\n  === 强确认的字符对 Top25 ===")
for (a, b), n in Counter((o["char_csv"], s2t.convert(o["char_csv"]))
                         for o, _ in strong).most_common(25):
    print(f"    {a} -> {b}: {n}")

print(f"\n  === 强确认按书体 ===")
print(f"    {dict(Counter(o['script'] for o, _ in strong))}")
print(f"\n  === 强确认按书家 Top12 ===")
for k, n in Counter(o["calligrapher"] for o, _ in strong).most_common(12):
    print(f"    {k}: {n}")

# 按置信度分层
cb = Counter("0.9+" if float(o["conf"]) >= 0.9 else
             "0.6-0.9" if float(o["conf"]) >= 0.6 else "<0.6"
             for o, _ in strong)
print(f"\n  === 强确认的 OCR 置信度分布 ===")
print(f"    {dict(cb)}")

with open("assets/mismatch_strong.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["idx", "image_path", "script", "calligrapher",
                "char_csv", "char_true", "conf_ocr"])
    for o, _ in strong:
        w.writerow([o["idx"], o["image_path"], o["script"], o["calligrapher"],
                    o["char_csv"], s2t.convert(o["char_csv"]), o["conf"]])
print(f"\n  ✓ written assets/mismatch_strong.csv ({len(strong)} 条)")
print(f"    占全量 {len(strong)/50786*100:.2f}%")

# 宽松版（含单模型）
with open("assets/mismatch_wide.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["idx", "image_path", "script", "calligrapher",
                "char_csv", "char_true", "source"])
    for o, _ in strong:
        w.writerow([o["idx"], o["image_path"], o["script"], o["calligrapher"],
                    o["char_csv"], s2t.convert(o["char_csv"]), "both"])
    for o in ocr_only:
        w.writerow([o["idx"], o["image_path"], o["script"], o["calligrapher"],
                    o["char_csv"], s2t.convert(o["char_csv"]), "ocr_only"])
    for o, _ in vlm_only:
        w.writerow([o["idx"], o["image_path"], o["script"], o["calligrapher"],
                    o["char_csv"], s2t.convert(o["char_csv"]), "vlm_only"])
print(f"  ✓ written assets/mismatch_wide.csv "
      f"({len(strong)+len(ocr_only)+len(vlm_only)} 条)")
