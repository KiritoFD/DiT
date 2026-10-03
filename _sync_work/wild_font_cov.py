"""检查 HCSU wild 的 (书体,字) 能否被我们现有字体渲染出标准骨架。

SCRIPT_FONT 只有 楷/行/隶 三档 -> 草/篆 只能 fallback 楷体(骨架书体不匹配, 有害)。
本脚本量化: 各书体下有多少 (书体,字) 组合能被**对应**字体渲染。
"""
import os, collections
from PIL import Image, ImageDraw, ImageFont

ROOT = "/root/Workspace/xy/HCSU/wild_extract"
FDIR = "/root/Workspace/xy/DiT/tools/fonts"

SCRIPT_FONT = {
    "楷": ["simkai.ttf", "STKAITI.TTF", "NotoSerifSC-VF.ttf"],
    "行": ["STXINGKA.TTF", "FZSTK.TTF"],
    "隶": ["SIMLI.TTF", "STLITI.TTF"],
}
FALLBACK = ["simkai.ttf", "simhei.ttf"]
AVAIL = set(os.listdir(FDIR)) if os.path.isdir(FDIR) else set()

# (书体,字) -> 样本数
pairs = collections.Counter()
for d in os.listdir(ROOT):
    dp = os.path.join(ROOT, d)
    if not os.path.isdir(dp):
        continue
    c, _, s = d.rpartition("-")
    for f in os.listdir(dp):
        if f.lower().endswith(".png"):
            pairs[(s, os.path.splitext(f)[0])] += 1

print("字体目录内容:", sorted(AVAIL))


def can_render(ch, fonts):
    """该字体能否真正画出这个字 (不是 .notdef 空框)。"""
    for f in fonts:
        fp = os.path.join(FDIR, f)
        if not os.path.exists(fp):
            continue
        try:
            font = ImageFont.truetype(fp, 200)
        except Exception:
            continue
        # 用 getmask 判断是否真画出了东西
        m = font.getmask(ch, mode="L")
        if m.size[0] > 0 and m.size[1] > 0:
            bbox = m.getbbox()
            if bbox is not None and (bbox[2] - bbox[0]) > 8 and (bbox[3] - bbox[1]) > 8:
                return f
    return None


print("\n=== 各书体: 对应字体渲染覆盖率 ===")
tot_ok = 0
for s in sorted({k[0] for k in pairs}, key=lambda x: -sum(v for k, v in pairs.items() if k[0] == x)):
    keys = [k for k in pairs if k[0] == s]
    n_img = sum(pairs[k] for k in keys)
    has_font = s in SCRIPT_FONT
    fonts = SCRIPT_FONT.get(s, []) if has_font else []
    ok_keys, ok_img = 0, 0
    for k in keys:
        if has_font and can_render(k[1], fonts):
            ok_keys += 1; ok_img += pairs[k]
    tag = "有对应字体" if has_font else "**无对应字体(只能楷体fallback)**"
    print(f"  {s}  {tag}")
    print(f"     组合 {ok_keys}/{len(keys)} 可渲染   样本 {ok_img}/{n_img} "
          f"({100*ok_img/max(1,n_img):.1f}%)")
    tot_ok += ok_img
print(f"\n  可用样本合计(书体匹配): {tot_ok} / {sum(pairs.values())}")

print("\n=== 若允许 fallback 到楷体(书体不匹配, 不推荐) ===")
tot_fb = 0
for s in sorted({k[0] for k in pairs}):
    keys = [k for k in pairs if k[0] == s]
    n_img = sum(pairs[k] for k in keys)
    ok = sum(pairs[k] for k in keys if can_render(k[1], SCRIPT_FONT.get(s, []) + FALLBACK))
    tot_fb += ok
    print(f"  {s}: {ok}/{n_img} ({100*ok/max(1,n_img):.1f}%)")
print(f"  合计: {tot_fb} / {sum(pairs.values())}")
