"""gradio: 支持一次输入多个字，每个字起一个线程并行生成。

- 输入 "小止北" -> 3 个线程并行 -> Gallery 展示 3 张生成图 + 3 张 GT
- 每个字用独立 seed（seed+i），保证可复现
- 模型只读共享，CPU 推理下多线程安全
"""
import io
import os

os.chdir("/root/Workspace/xy/DiT")
P = "gradio_stdskel.py"
s = io.open(P, encoding="utf-8").read()

# ── 1) 多字并行生成函数（插在 UI 之前）─────────────────────────────────
MULTI = '''
def generate_multi(chars, callig_name, script, steps, cfg, seed, nthreads=4):
    """一次生成多个字。每个字一个线程并行。

    返回 (生成图列表, GT图列表, 状态文本)
    """
    chars = (chars or "").strip()
    if not chars:
        return [], [], "请输入至少一个汉字"
    # 去重保序
    seen = set()
    uniq = []
    for ch in chars:
        if ch not in seen:
            seen.add(ch)
            uniq.append(ch)
    if len(uniq) > 24:
        uniq = uniq[:24]

    def _one(i_ch):
        i, ch = i_ch
        try:
            img, gimg, msg = generate(ch, callig_name, script, steps, cfg,
                                      int(seed) + i)
        except Exception as e:
            return ch, None, None, f"✗ {ch}: {e}"
        return ch, img, gimg, msg

    from concurrent.futures import ThreadPoolExecutor
    outs = [None] * len(uniq)
    nthreads = max(1, min(int(nthreads), len(uniq)))
    with ThreadPoolExecutor(max_workers=nthreads) as ex:
        for k, (ch, img, gimg, msg) in enumerate(
                ex.map(_one, list(enumerate(uniq)))):
            outs[k] = (ch, img, gimg, msg)

    gens, gts, msgs = [], [], []
    for ch, img, gimg, msg in outs:
        if img is not None:
            gens.append(img)
            gts.append(gimg)
        msgs.append(msg)

    ok = len(gens)
    return gens, gts, (f"生成 {ok}/{len(uniq)} 个字（{nthreads} 线程并行）\\n"
                       + "\\n".join(msgs[:6]))


'''
anchor = "def _cur_title():"
if anchor in s and "def generate_multi" not in s:
    s = s.replace(anchor, MULTI + anchor, 1)
    print("  ✓ 插入 generate_multi")
else:
    print("  ⚠ 跳过 generate_multi")

# ── 2) UI: 输入框支持多字 + Gallery ────────────────────────────────────
OLD_CHAR = '''    with gr.Row():
        char_in = gr.Textbox(label="汉字 (单字) —— 笔画少的字效果更好: 小止北千代三此凶沙九", value="小")'''
NEW_CHAR = '''    with gr.Row():
        char_in = gr.Textbox(label="汉字 (可多字, 每字一个线程并行) —— 笔画少的字效果更好: 小止北千代三此凶沙九", value="小止北")'''
if OLD_CHAR in s:
    s = s.replace(OLD_CHAR, NEW_CHAR, 1)
    print("  ✓ 输入改为多字")
else:
    print("  ⚠ 没找到 char_in（可能已被改过），尝试宽松匹配")
    import re
    m = re.search(r'char_in = gr\.Textbox\([^\n]*\)', s)
    if m:
        s = s[:m.start()] + ('char_in = gr.Textbox(label="汉字 (可多字, 每字一个线程并行) —— '
                            '笔画少的字效果更好: 小止北千代三此凶沙九", value="小止北")') + s[m.end():]
        print("  ✓ 输入改为多字（宽松匹配）")

OLD_IMG = '''    with gr.Row():
        img_out = gr.Image(label="生成结果", type="pil")
        gt_out = gr.Image(label="真实参考 (GT)", type="pil")'''
NEW_IMG = '''    with gr.Row():
        nthread_in = gr.Slider(1, 8, value=4, step=1, label="并行线程数")
    with gr.Row():
        img_out = gr.Gallery(label="生成结果", columns=4, height=260)
        gt_out = gr.Gallery(label="真实参考 (GT)", columns=4, height=260)'''
if OLD_IMG in s:
    s = s.replace(OLD_IMG, NEW_IMG, 1)
    print("  ✓ 输出改 Gallery")
else:
    print("  ⚠ 没找到 img_out/gt_out 行")

OLD_CLK = '''    btn.click(generate, [char_in, callig_in, script_in, steps_in, cfg_in, seed_in],
              [img_out, gt_out, msg])'''
NEW_CLK = '''    btn.click(generate_multi,
              [char_in, callig_in, script_in, steps_in, cfg_in, seed_in, nthread_in],
              [img_out, gt_out, msg])'''
if OLD_CLK in s:
    s = s.replace(OLD_CLK, NEW_CLK, 1)
    print("  ✓ btn.click 改 generate_multi")
else:
    print("  ⚠ 没找到 btn.click")

io.open(P, "w", encoding="utf-8").write(s)
import ast

ast.parse(s)
print("  SYNTAX OK")
