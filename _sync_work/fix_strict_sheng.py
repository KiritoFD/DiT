"""把 strict 里那条错标的「升」改成「陞」(U+965E)。"""
import csv
import os

os.chdir("/root/Workspace/xy/DiT")
p = "assets/eval_v13_strict_fixed.csv"
SHENG_JIAN = "\u5347"      # 升 U+5347
SHENG_YI = "\u965e"        # 陞 U+965E
WRONG = "\u969e"           # 隞 U+969E（上一次误改成的）

rows = list(csv.DictReader(open(p, encoding="utf-8")))
cols = list(rows[0].keys())
n = 0
for r in rows:
    if r["old_50k_id"] == "044447":
        old = r["character"]
        if old in (SHENG_JIAN, WRONG):
            r["character"] = SHENG_YI
            n += 1
            print(f"  {r['old_50k_id']}: {old} (U+{ord(old):04X}) -> "
                  f"{SHENG_YI} (U+{ord(SHENG_YI):04X})")
        else:
            print(f"  {r['old_50k_id']}: 当前已是 {old} (U+{ord(old):04X})，跳过")
with open(p, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    w.writerows(rows)
print(f"  共改 {n} 条")

# 复验
rows = list(csv.DictReader(open(p, encoding="utf-8")))
for r in rows:
    if r["old_50k_id"] == "044447":
        ch = r["character"]
        print(f"  复验: character={ch} U+{ord(ch):04X}  "
              f"std_path={r['std_path']}  src={r['src_image_path']}")
