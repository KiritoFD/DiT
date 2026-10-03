"""gradio: 回退多字 -> 单字；历史生成结果累积保留；GT 放在下方；回车即生成。"""
import io
import os
import re

os.chdir("/root/Workspace/xy/DiT")
P = "gradio_stdskel.py"
s = io.open(P, encoding="utf-8").read()

# ── 1) smoke 回到单字 generate ─────────────────────────────────────────
s = s.replace(
    '    _gens, _gts, _msg = generate_multi("小止北", CALLIG_NAMES[0], "楷", args.steps, args.cfg, 0, 4)\n'
    '    _img = _gens[0] if _gens else None\n',
    '    _img, _gimg, _msg = generate("小", CALLIG_NAMES[0], "楷", args.steps, args.cfg, 0)\n')
print("  ✓ smoke 回单字")

# ── 2) 重写 UI 主体 ────────────────────────────────────────────────────
i0 = s.index("        char_in = gr.Textbox(")
i1 = s.index("              [img_out, gt_out, msg])") + len("              [img_out, gt_out, msg])")

new_ui = '''        char_in = gr.Textbox(
            label="汉字 (单字, 输入后按回车即生成) —— 笔画少的字效果更好: 小止北千代三此凶沙九",
            value="小")
        callig_in = gr.Dropdown(CALLIG_NAMES, label="书家",
                                value=(PREFERRED_CALLIG if PREFERRED_CALLIG in CALLIG_NAMES
                                       else (CALLIG_NAMES[0] if CALLIG_NAMES else None)))
        script_in = gr.Radio(["楷", "行", "隶"], label="书体", value="楷")
    with gr.Row():
        steps_in = gr.Slider(10, 100, value=args.steps, step=10, label="采样步数")
        cfg_in = gr.Slider(0.0, 4.0, value=args.cfg, step=0.1, label="CFG")
        seed_in = gr.Number(value=0, label="seed", precision=0)
    btn = gr.Button("生成 (或按回车)", variant="primary")

    # ★ 历史累积：每次生成把结果 append 进来，页面上一直保留
    hist = gr.State({"gens": [], "gts": []})

    with gr.Accordion("切换 ckpt", open=False):
        ckpt_dd = gr.Dropdown([n for n, _ in CKPT_LIST], label="可用 ckpt",
                              value=(os.path.basename(os.path.dirname(
                                  os.path.dirname(os.path.dirname(CUR_CKPT))))
                                  + " @" + (os.path.basename(CUR_CKPT)
                                            .replace(".pt", "").lstrip("0") or "0"))
                              if CKPT_LIST else None)
        ckpt_btn = gr.Button("加载所选 ckpt")
        ckpt_msg = gr.Textbox(label="切换状态", interactive=False)

        def _on_switch(name):
            path = CKPT_MAP.get(name)
            if not path:
                return _cur_title(), f"✗ 未找到 {name}", gr.update()
            try:
                m = _load_ckpt(path)
            except Exception as e:
                import traceback
                traceback.print_exc()
                return _cur_title(), f"✗ 加载失败: {e}", gr.update()
            # ★ 同时刷新书家下拉，否则下拉还是旧表的书家 -> "未知书家"
            new_names = CALLIG_NAMES
            return (_cur_title(), m,
                    gr.update(choices=new_names,
                              value=(PREFERRED_CALLIG if PREFERRED_CALLIG in new_names
                                     else (new_names[0] if new_names else None))))

        ckpt_btn.click(_on_switch, [ckpt_dd], [md, ckpt_msg, callig_in])

    # 生成结果在上，GT 在下（一一对应）
    img_out = gr.Gallery(label="生成结果（历史累积，最新在最后）", columns=6,
                         height=300, allow_preview=True)
    gt_out = gr.Gallery(label="真实参考 GT（与上面一一对应）", columns=6,
                        height=300, allow_preview=True)
    msg = gr.Textbox(label="状态")

    def _on_gen(char, callig, script, steps, cfg, seed, state):
        img, gimg, m = generate(char, callig, script, steps, cfg, seed)
        if img is not None:
            state["gens"].append(img)
            state["gts"].append(gimg)
            # 最多保留 24 条，防止页面越来越慢
            state["gens"] = state["gens"][-24:]
            state["gts"] = state["gts"][-24:]
        return state["gens"], state["gts"], m, state

    _ins = [char_in, callig_in, script_in, steps_in, cfg_in, seed_in, hist]
    _outs = [img_out, gt_out, msg, hist]
    btn.click(_on_gen, _ins, _outs)
    # ★ 回车即生成
    char_in.submit(_on_gen, _ins, _outs)'''

s = s[:i0] + new_ui + s[i1:]
io.open(P, "w", encoding="utf-8").write(s)
import ast

ast.parse(s)
print("  ✓ UI 已重写（单字 + 历史累积 + GT 在下 + 回车提交）")
print("  SYNTAX OK")
