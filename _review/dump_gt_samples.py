"""直接看 GT 图: csv 里标为简体字的样本，GT 是不是实际写的繁体？

这比 OCR 更快 —— 肉眼/我直接看几张就能定性。
"""
import csv
import os
import shutil

os.chdir("/root/Workspace/xy/DiT")
rows = list(csv.DictReader(open("assets/train_50k_v2.csv", encoding="utf-8")))

# 挑"简体写法"的字，看它们的 GT 是不是写成了繁体
simp_chars = ["并", "乱", "来", "国", "学", "后", "书", "东", "乐", "风",
              "门", "马", "鸟", "龙", "云", "与", "为", "画", "万", "亚"]

out = "_review/gt_check"
os.makedirs(out, exist_ok=True)

picked = {}
for r in rows:
    c = r["character"]
    if c in simp_chars and c not in picked:
        picked[c] = (r["image_path"], r["script"], r["calligrapher"])

print(f"  找到 {len(picked)} 个简体字的 GT")
for c, (p, sc, cal) in sorted(picked.items()):
    src = p if os.path.isabs(p) else os.path.join("/root/Workspace/xy/DiT", p)
    if os.path.exists(src):
        shutil.copy(src, os.path.join(out, f"{c}.png"))
        print(f"    {c} [{sc}/{cal}] <- {p}")
    else:
        print(f"    {c}: ✗ {p}")

# 拼成对比图
try:
    from PIL import Image, ImageDraw

    keys = [c for c in simp_chars if os.path.exists(os.path.join(out, f"{c}.png"))]
    W = 120
    n = min(len(keys), 12)
    canvas = Image.new("RGB", (W * n, W + 24), "white")
    d = ImageDraw.Draw(canvas)
    for i, c in enumerate(keys[:n]):
        im = Image.open(os.path.join(out, f"{c}.png")).convert("RGB")
        im = im.resize((W - 10, W - 10))
        canvas.paste(im, (i * W + 5, 20))
    canvas.save(os.path.join(out, "gt_row.png"))
    print(f"\n  -> {out}/gt_row.png   (csv 标为: {' '.join(keys[:n])})")
except Exception as e:
    print(f"  拼图失败: {e}")
