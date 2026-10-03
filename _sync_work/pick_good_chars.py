"""从 eval 结果里挑出生成效果最好的字（strict ssim 最高），作为 gradio 默认展示。"""
import csv
import glob
import os

os.chdir("/root/Workspace/xy/DiT")

# 找 eval 逐样本 CSV（带字符列）
cands = []
for p in glob.glob("assets/results/*/eval_stdskel_batch.csv"):
    cands.append(p)
print(f"  候选 eval CSV: {len(cands)}")
for p in cands[:3]:
    print(f"    {p}")

# 用哪个：优先 v13_wd01
target = None
for p in cands:
    if "v13_wd01" in p:
        target = p
        break
if target is None and cands:
    target = cands[0]
if not target:
    raise SystemExit("没有 eval 逐样本 CSV")

print(f"\n  使用: {target}")
rows = list(csv.DictReader(open(target, encoding="utf-8")))
print(f"  列: {list(rows[0].keys())}")
print(f"  行数: {len(rows)}")

# 只需要 strict 集
strict = [r for r in rows if r.get("set") == "strict"]
print(f"  strict 样本: {len(strict)}")
if not strict:
    strict = rows

# 需要字符：CSV 里若有 character 列；否则用 idx 对齐 eval csv
if "character" in strict[0]:
    for r in strict:
        r["_char"] = r["character"]
else:
    ev = list(csv.DictReader(open("assets/eval_v13_strict.csv", encoding="utf-8")))
    for r in strict:
        try:
            i = int(r.get("idx", -1))
        except ValueError:
            i = -1
        r["_char"] = ev[i]["character"] if 0 <= i < len(ev) else "?"

# 取每个 eval step 最新的一批
steps = sorted(set(r.get("step", "") for r in strict))
print(f"  eval step: {steps[-3:]}")
last = steps[-1] if steps else None
sub = [r for r in strict if r.get("step") == last] or strict
print(f"  取 step={last}: {len(sub)} 条")


def ssim_of(r):
    try:
        return float(r.get("ssim", 0))
    except ValueError:
        return 0.0


sub.sort(key=ssim_of, reverse=True)
print(f"\n  === 效果最好的 20 个字（step {last}, strict）===")
for r in sub[:20]:
    print(f"    {r['_char']}  ssim={ssim_of(r):.4f}")

print("\n  Top10 单字串:", "".join(r["_char"] for r in sub[:10]))
