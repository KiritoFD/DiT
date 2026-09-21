"""gradio: 1) 默认书家换成学得最好的；2) 页面展示相关 GT 图。

默认书家选择依据（step 95000 strict + 训练样本数）:
  蔡襄   0.659 / 1592 样本 / 971 字     <- 学得好且样本多
  董其昌 0.640 / 1714 样本
  于右任 0.550 /  587 样本             <- 原默认，偏低

GT: 从 train_50k_v2.csv 建 (书家, 书体, 字) -> image_path 索引，
   生成时把对应的真实书法图一起展示，便于对比。
"""
import io
import os

os.chdir("/root/Workspace/xy/DiT")
P = "gradio_stdskel.py"
s = io.open(P, encoding="utf-8").read()

# ── 1) GT 索引（插在 _build_callig_index 之前）─────────────────────────
GT_CODE = '''
# ★ [2026-09-21] GT 索引: (书家, 书体, 字) -> 真实书法图路径
def _build_gt_index(csv_path):
    import csv as _csv
    idx = {}
    if not os.path.isfile(csv_path):
        return idx
    with open(csv_path, encoding="utf-8") as f:
        for r in _csv.DictReader(f):
            k = (str(r.get("calligrapher", "")), str(r.get("script", "")),
                 str(r.get("character", "")))
            p = r.get("image_path", "")
            if k not in idx and p:
                idx[k] = p
    return idx


GT_INDEX = _build_gt_index(args.train_csv)
print(f"[gt] {len(GT_INDEX)} (书家,书体,字) 组合", flush=True)


def _find_gt(callig, script, char):
    """找不到该组合时，退化到同书家同字（任意书体），再退化到同字任意书家。"""
    p = GT_INDEX.get((callig, script, char))
    if p:
        return p, f"{callig}/{script}/{char}"
    for (c, sc, ch), pp in GT_INDEX.items():
        if c == callig and ch == char:
            return pp, f"{callig}/{sc}/{char}(书体回退)"
    for (c, sc, ch), pp in GT_INDEX.items():
        if ch == char:
            return pp, f"{c}/{sc}/{char}(书家回退)"
    return None, None


'''
anchor = "def _build_callig_index("
if anchor in s and "_build_gt_index" not in s:
    s = s.replace(anchor, GT_CODE + anchor, 1)
    print("  ✓ 插入 GT 索引")
else:
    print("  ⚠ 跳过 GT 索引（已存在？）")

# ── 2) generate 返回 GT ───────────────────────────────────────────────
OLD_GEN = '''    return Image.fromarray((img * 255).astype("uint8")), \\
        f"OK char={char} script={script} callig={callig_name} steps={steps} cfg={cfg}"'''
NEW_GEN = '''    gp, gdesc = _find_gt(callig_name, script, char)
    gimg = None
    if gp and os.path.isfile(gp):
        try:
            gimg = Image.open(gp).convert("RGB")
        except Exception:
            gimg = None
    return (Image.fromarray((img * 255).astype("uint8")), gimg,
            f"OK char={char} script={script} callig={callig_name} "
            f"steps={steps} cfg={cfg}  |  GT: {gdesc or '无'}")'''
if OLD_GEN in s:
    s = s.replace(OLD_GEN, NEW_GEN, 1)
    print("  ✓ generate 返回 GT")
else:
    print("  ⚠ 没找到 generate 返回")

# ── 3) UI: 加 GT 图框 + 改默认书家 ────────────────────────────────────
OLD_UI2 = '''    img_out = gr.Image(label="生成结果", type="pil")'''
NEW_UI2 = '''    with gr.Row():
        img_out = gr.Image(label="生成结果", type="pil")
        gt_out = gr.Image(label="真实参考 (GT)", type="pil")'''
if OLD_UI2 in s:
    s = s.replace(OLD_UI2, NEW_UI2, 1)
    print("  ✓ 加 GT 图框")
else:
    print("  ⚠ 没找到 img_out")

# btn.click 的输出要加 gt_out
OLD_CLK = '''    btn.click(generate, [char_in, callig_in, script_in, steps_in, cfg_in, seed_in],
              [img_out, msg])'''
NEW_CLK = '''    btn.click(generate, [char_in, callig_in, script_in, steps_in, cfg_in, seed_in],
              [img_out, gt_out, msg])'''
if OLD_CLK in s:
    s = s.replace(OLD_CLK, NEW_CLK, 1)
    print("  ✓ btn.click 加 gt_out")
else:
    print("  ⚠ 没找到 btn.click")

# 默认书家: 学得最好的
OLD_DEF = '''value=CALLIG_NAMES[0] if CALLIG_NAMES else None'''
NEW_DEF = '''value=(PREFERRED_CALLIG if PREFERRED_CALLIG in CALLIG_NAMES
                               else (CALLIG_NAMES[0] if CALLIG_NAMES else None))'''
if OLD_DEF in s:
    s = s.replace(OLD_DEF, NEW_DEF, 1)
    print("  ✓ 默认书家改用优选")
else:
    print("  ⚠ 没找到书家默认值")

# 定义 PREFERRED_CALLIG（放在 CALLIG_NAMES 定义之后）
if "PREFERRED_CALLIG" in s and "PREFERRED_CALLIG =" not in s:
    # 在 _load_ckpt 之前加定义
    s = s.replace("def _load_ckpt(path):",
                  '# 学得最好的书家（strict 0.659 / 1592 训练样本；原默认"于右任"只有 0.550）\n'
                  'PREFERRED_CALLIG = "蔡襄"\n\n\ndef _load_ckpt(path):', 1)
    print("  ✓ 定义 PREFERRED_CALLIG")

io.open(P, "w", encoding="utf-8").write(s)
import ast

ast.parse(s)
print("  SYNTAX OK")
