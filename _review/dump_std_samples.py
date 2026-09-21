"""找几个繁体字的 std_path，把图拷出来直接看（最快验证简繁错配）。"""
import csv
import os
import shutil

os.chdir("/root/Workspace/xy/DiT")

rows = list(csv.DictReader(open("assets/train_50k_v2.csv", encoding="utf-8")))

# 挑几个有代表性的繁体字（简繁差异明显的）
targets = ["並", "亂", "來", "亞", "書", "國", "學", "門", "馬", "鳥",
           "龍", "東", "風", "雲", "萬", "與", "為", "後", "畫", "樂"]
# 以及对应的简体（如果存在）
simp_of = {"並": "并", "亂": "乱", "來": "来", "亞": "亚", "書": "书",
           "國": "国", "學": "学", "門": "门", "馬": "马", "鳥": "鸟",
           "龍": "龙", "東": "东", "風": "风", "雲": "云", "萬": "万",
           "與": "与", "為": "为", "後": "后", "畫": "画", "樂": "乐"}

out = "_review/std_check"
os.makedirs(out, exist_ok=True)

found = {}
for r in rows:
    c = r["character"]
    if c in targets and c not in found:
        found[c] = r["std_path"]
    if c in simp_of.values() and ("S_" + c) not in found:
        found["S_" + c] = r["std_path"]

print(f"  找到 {len(found)} 个")
for k, p in sorted(found.items()):
    src = p if os.path.isabs(p) else os.path.join("/root/Workspace/xy/DiT", p)
    if os.path.exists(src):
        dst = os.path.join(out, f"{k}.png")
        shutil.copy(src, dst)
        print(f"    {k} <- {p}")
    else:
        print(f"    {k}: ✗ 不存在 {p}")

# 拼成一张对比图（繁体上排，简体下排）
try:
    from PIL import Image, ImageDraw

    trad = [k for k in sorted(found) if not k.startswith("S_")][:10]
    simp = ["S_" + simp_of.get(k, "") for k in trad]
    W = 96
    canvas = Image.new("RGB", (W * len(trad), W * 2 + 20), "white")
    d = ImageDraw.Draw(canvas)
    for i, k in enumerate(trad):
        for row, kk in enumerate((k, simp[i])):
            p = os.path.join(out, f"{kk}.png")
            if os.path.exists(p):
                im = Image.open(p).convert("RGB").resize((W - 8, W - 8))
                canvas.paste(im, (i * W + 4, row * W + 4 + row * 10))
    canvas.save(os.path.join(out, "compare.png"))
    print(f"\n  拼图 -> {out}/compare.png  (上排=繁体字符的std, 下排=对应简体字符的std)")
except Exception as e:
    print(f"  拼图失败: {e}")
