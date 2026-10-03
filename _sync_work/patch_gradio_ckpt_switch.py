"""给远端 gradio_stdskel.py 加:
  1) 动态标题（原来硬编码 "c41x_cos_e@390k"，换 ckpt 后不更新）
  2) ckpt 下拉切换（扫描 assets/results 下所有 ckpt，切换时从 ckpt args
     自动取 callig_map / latent_channels / scaling factor，重建模型）
"""
import io
import os

os.chdir("/root/Workspace/xy/DiT")
P = "gradio_stdskel.py"
s = io.open(P, encoding="utf-8").read()

# ── 1) 插一个"按 ckpt 重建一切"的函数（放在 generate 之前）──────────────
HELPER = '''
# ★ [2026-09-21] ckpt 热切换支持
#   原来标题硬编码 "c41x_cos_e@390k"，换 ckpt 后 UI 还显示旧名字；
#   且没有切换入口。这里: 扫描所有 ckpt + 一键重建模型/书家表/条件库。
def _scan_ckpts():
    """返回 [(显示名, 路径)]。显示名 = 实验目录/步数。"""
    import glob as _g
    out = []
    for p in sorted(_g.glob("assets/results/*/*/checkpoints/*.pt")):
        parts = p.split(os.sep)
        exp = parts[2] if len(parts) > 2 else "?"
        step = os.path.basename(p).replace(".pt", "").lstrip("0") or "0"
        out.append((f"{exp} @{step}k", p))
    # 最新步数在前
    out.sort(key=lambda kv: os.path.getmtime(kv[1]), reverse=True)
    return out


CKPT_LIST = _scan_ckpts()
CKPT_MAP = dict(CKPT_LIST)


def _callig_label_of(raw):
    """书家的显示名: 有 reverse map 就用，没有就用 raw id。"""
    return str(raw)


def _load_ckpt(path):
    """按给定 ckpt 重建 model / 书家表 / 条件参数。返回状态字符串。"""
    global model, cmap, raw2idx, name_of, CALLIG_NAMES, LC, sf, CUR_CKPT
    ck = torch.load(path, map_location="cpu", weights_only=False)
    a = ck.get("args", {})
    if not isinstance(a, dict):
        a = vars(a) if hasattr(a, "__dict__") else {}

    class _NS(dict):
        def __getattr__(self, k):
            try:
                return self[k]
            except KeyError:
                return None

    _ns = _NS(dict(a))
    _ns.setdefault("device", dev)
    _ns["cond_drop_all_prob"] = 0.05
    _ns["cond_drop_one_prob"] = 0.25
    _ns["cond_drop_which_glyph_prob"] = 0.5
    _ns["use_checkpoint"] = False
    _ns["learn_sigma"] = False
    _ns["glyph_drop_prob"] = 0.0
    _ns["attn_impl"] = "eager"
    _m = _cfs.build_model_from_args(_ns, dev)
    if a.get("freeze_callig_table"):
        _m.y_callig_embedder.freeze_table()
    _sd = ck.get("ema") or ck.get("model")
    if isinstance(_sd, dict):
        _sd = {k.replace("_orig_mod.", "", 1): v for k, v in _sd.items()}
        _ms, _us = _m.load_state_dict(_sd, strict=False)
        if _ms:
            print(f"[switch] ⚠ {len(_ms)} 个权重缺失", flush=True)
    model = _m.to(dev).eval()

    # 书家表: 优先 ckpt args 里记录的 map（v13 用 callig_id_map, v15 用 pair map）
    _mp = a.get("callig_id_map") or a.get("callig_script_map") or args.callig_map
    try:
        cmap, _n = load_callig_id_map(_mp)
    except Exception as e:
        return f"✗ 书家表加载失败 {_mp}: {e}"
    raw2idx = {r: i for r, i in cmap.items()}
    name_of = {str(r): r for r in cmap}
    CALLIG_NAMES = sorted(name_of.keys())
    LC = int(a.get("latent_channels") or 4)
    sf = float(a.get("vae_scaling_factor", 0.18215))
    CUR_CKPT = path
    _step = os.path.basename(path).replace(".pt", "").lstrip("0") or "0"
    return (f"✓ 已切换到 {os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(path))))}"
            f" @{_step}  (书家 {len(CALLIG_NAMES)} 个, latent_ch={LC})")


CUR_CKPT = args.ckpt

'''

anchor = "@torch.no_grad()\ndef generate("
if anchor in s:
    s = s.replace(anchor, HELPER + anchor, 1)
    print("  ✓ 插入 ckpt 切换函数")
else:
    print("  ⚠ 没找到 generate 定义")

# ── 2) UI: 动态标题 + 下拉 + 切换按钮 ──────────────────────────────────
OLD_UI = '''with gr.Blocks(title="fame 书法生成 (std-skel-g, CPU)") as demo:
    gr.Markdown("## fame 书法生成 — std-skel-g DiT (c41x_cos_e@390k)\\n"
                "CPU 推理; g=标准字形骨架 latent; 仅支持训练集出现过的字。")'''

NEW_UI = '''def _cur_title():
    _exp = os.path.basename(os.path.dirname(
        os.path.dirname(os.path.dirname(CUR_CKPT))))
    _st = os.path.basename(CUR_CKPT).replace(".pt", "").lstrip("0") or "0"
    return (f"## fame 书法生成 — std-skel-g DiT\\n"
            f"**当前: {_exp} @{_st}** · CPU · "
            f"g=标准字形骨架 latent · 库存 {len(bank)} 条")


with gr.Blocks(title="fame 书法生成 (std-skel-g, CPU)") as demo:
    md = gr.Markdown(_cur_title())'''

if OLD_UI in s:
    s = s.replace(OLD_UI, NEW_UI, 1)
    print("  ✓ 标题改动态")
else:
    print("  ⚠ 没找到原标题")

# ── 3) 加切换行 ────────────────────────────────────────────────────────
OLD_BTN = '''    btn = gr.Button("生成", variant="primary")'''
NEW_BTN = '''    btn = gr.Button("生成", variant="primary")
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
                return _cur_title(), f"✗ 未找到 {name}"
            try:
                m = _load_ckpt(path)
            except Exception as e:
                import traceback
                traceback.print_exc()
                return _cur_title(), f"✗ 加载失败: {e}"
            return _cur_title(), m

        ckpt_btn.click(_on_switch, [ckpt_dd], [md, ckpt_msg])'''

if OLD_BTN in s:
    s = s.replace(OLD_BTN, NEW_BTN, 1)
    print("  ✓ 加切换下拉")
else:
    print("  ⚠ 没找到生成按钮")

io.open(P, "w", encoding="utf-8").write(s)
print("  ✓ 已写入")
