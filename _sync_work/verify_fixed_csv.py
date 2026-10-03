"""验证修复后的 csv 是否真的改了（上一个脚本的校验因原地修改失效）。"""
import csv

a = list(csv.DictReader(open("assets/train_50k_v2.csv", encoding="utf-8")))
b = list(csv.DictReader(open("assets/train_50k_v2_fixed.csv", encoding="utf-8")))
print(f"  原 {len(a)} 行, 修复 {len(b)} 行")

ch = [(x["character"], y["character"]) for x, y in zip(a, b)
      if x["character"] != y["character"]]
sp = [(x["std_path"], y["std_path"]) for x, y in zip(a, b)
      if x["std_path"] != y["std_path"]]
print(f"  character 变化: {len(ch)}")
print(f"  std_path 变化:  {len(sp)}")

print(f"\n  === 变化样例 12 ===")
for (x, y) in ch[:12]:
    print(f"    {x} -> {y}")

# 抽查：确认修复行的 std_path 确实指向繁体字的 std
conf = list(csv.DictReader(open("assets/mismatch_confirmed.csv",
                                encoding="utf-8")))
fixmap = {c["image_path"]: c["char_true"] for c in conf}
n_ok = 0
for x, y in zip(a, b):
    if x["image_path"] in fixmap:
        want = fixmap[x["image_path"]]
        if y["character"] == want:
            n_ok += 1
print(f"\n  === 抽查 ===")
print(f"    修复行中 character 正确: {n_ok}/{len(fixmap)}")
