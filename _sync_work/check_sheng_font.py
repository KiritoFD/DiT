"""查「陞」的字体支持 + 训练集里那条「陞」的 std 是怎么来的。"""
import csv
import os

os.chdir("/root/Workspace/xy/DiT")
SHENG_YI = "\u965e"

print("  === 训练集里那条「陞」 ===")
for r in csv.DictReader(open("assets/train_50k_v2_fixed.csv",
                             encoding="utf-8")):
    if r["character"] == SHENG_YI:
        print(f"    id={r['old_50k_id']} {r['calligrapher']}/{r['script']}")
        print(f"    std_path   = {r['std_path']}")
        print(f"    image_path = {r['image_path']}")
        print(f"    src        = {r['src_image_path']}")
        print(f"    std 存在: {os.path.exists(r['std_path'])}")
        break

print("\n  === 字体对「陞」的支持 ===")
from PIL import ImageFont
cands = []
for d in ("_fonts", "/usr/share/fonts", "/root/.fonts",
          "/usr/share/fonts/truetype"):
    if os.path.isdir(d):
        for root, dirs, files in os.walk(d):
            for f in files:
                if f.lower().endswith((".ttf", ".ttc", ".otf")):
                    cands.append(os.path.join(root, f))
print(f"    找到 {len(cands)} 个字体文件")
for fp in cands[:40]:
    try:
        ft = ImageFont.truetype(fp, 64)
        # 用 getmask 判断是否有该字形（缺字会画成 .notdef / 空）
        m = ft.getmask(SHENG_YI)
        bbox = m.getbbox()
        has = bbox is not None and (bbox[2] - bbox[0]) > 2
        if has:
            print(f"    ✓ {os.path.basename(fp)}  支持「陞」 bbox={bbox}")
    except Exception:
        pass
print("    （没列出的 = 不支持或缺字）")
