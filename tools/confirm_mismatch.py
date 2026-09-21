"""用 VL-7B 的结果确认 rapidocr 的 902 条"真错配候选"。

判据:
  OCR 说 GT 是繁体（char_ocr = s2t(char_csv)）
  VL-7B 也说 GT 是繁体（char_vlm == char_ocr）  -> **确认错配** ✓
  VL-7B 说 GT 是简体（char_vlm == char_csv）     -> OCR 错认 ✗
"""
import csv
from collections import Counter

from opencc import OpenCC

s2t = OpenCC("s2t")
t2s = OpenCC("t2s")

cand = list(csv.DictReader(open("assets/mismatch_real_candidates.csv",
                                encoding="utf-8")))
vlm = list(csv.DictReader(open("assets/vlm_7b_mismatch.csv", encoding="utf-8")))
print(f"  候选 {len(cand)}, VLM 结果 {len(vlm)}")

# 按 image_path 对齐
vm = {r["image_path"]: r for r in vlm}

confirm = []   # VLM 也说是繁体
reject = []    # VLM 说是简体（OCR 错认）
other = []
for c in cand:
    v = vm.get(c["image_path"])
    if not v:
        other.append(c)
        continue
    cv, vl = c["char_csv"], v["char_vlm"]
    if vl == c["char_ocr"]:
        confirm.append((c, vl))
    elif vl == cv:
        reject.append((c, vl))
    else:
        other.append((c, vl))

print(f"\n  === 结果 ===")
print(f"    ★ VLM 确认是繁体（真错配）: {len(confirm)} = "
      f"{len(confirm)/max(len(cand),1)*100:.1f}%")
print(f"    VLM 说是简体（OCR 错认）:   {len(reject)}")
print(f"    其他:                       {len(other)}")

print(f"\n  === 确认错配的字符对 Top20 ===")
for (a, b), n in Counter((c["char_csv"], c["char_ocr"])
                         for c, _ in confirm).most_common(20):
    print(f"    {a} -> {b}: {n}")

print(f"\n  === 确认错配按书家 Top12 ===")
for k, n in Counter(c["calligrapher"] for c, _ in confirm).most_common(12):
    print(f"    {k}: {n}")

print(f"\n  === 确认错配按书体 ===")
print(f"    {dict(Counter(c['script'] for c, _ in confirm))}")

print(f"\n  === 确认错配样例 15 ===")
for c, vl in confirm[:15]:
    print(f"    {c['char_csv']} -> {vl}  conf(ocr)={c['conf']}  "
          f"[{c['script']}/{c['calligrapher']}]")

# 导出确认的
with open("assets/mismatch_confirmed.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["idx", "image_path", "script", "calligrapher",
                "char_csv", "char_true", "conf_ocr"])
    for c, vl in confirm:
        w.writerow([c["idx"], c["image_path"], c["script"], c["calligrapher"],
                    c["char_csv"], vl, c["conf"]])
print(f"\n  written assets/mismatch_confirmed.csv ({len(confirm)} 条)")

# 顺便: 反过来的情况（csv 是繁体，VLM 说是简体 -> 反向错配）
print(f"\n  === 反向检查（csv 繁体 / VLM 简体）===")
rev = [(c, v) for c, v in ((x, vm.get(x["image_path"])) for x in cand)
       if v and t2s.convert(c["char_csv"]) == v["char_vlm"]
       and t2s.convert(c["char_csv"]) != c["char_csv"]]
print(f"    {len(rev)} 条（csv 标繁体但 GT 写简体）")
