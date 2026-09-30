#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tools/diag_check_v25_stdskel.py — 纯配置/实现核查 (不训练, 不碰 GPU 主进程)。
验证 v25_stdskel.json:
  1. 配置可被正常解析, 关键键类型正确;
  2. 用 train.py 同样的参数透传路径构模, 断言 model.deform_skel is None;
  3. 断言 deform 相关 loss 分支/w_deform_skel 全为关闭;
  4. 断言 skel_as_glyph_cond=True 且 skel_latent_shards_dir 指向 top10 shards_std;
  5. 断言 eval 侧构模同样不带 deform_skel (model_io 透传一致)。
全部在 CPU 上跑, 不占用 GPU。
"""
import os, sys, json

sys.stdout.reconfigure(encoding="utf-8")
ROOT = "/root/Workspace/xy/DiT"
os.chdir(ROOT)
sys.path.insert(0, ROOT)

CFG = "src/train/configs/v25_stdskel.json"
ok = True

def chk(cond, msg):
    global ok
    print(("  ✓ " if cond else "  ✗ ") + msg)
    if not cond:
        ok = False

print("=" * 78)
print("[1] 配置解析与关键键")
print("=" * 78)
cfg = json.load(open(CFG, encoding="utf-8"))
chk(cfg["deform_skel"] == 0, "deform_skel == 0 (SkelNet 关闭)")
chk(cfg["w_deform_skel"] == 0.0, "w_deform_skel == 0.0 (形变监督关闭)")
chk(cfg["deform_ckpt"] == "", "deform_ckpt 为空 (不载任何形变权重)")
chk(cfg["inst_skel_shards_dir"] == "", "inst_skel_shards_dir 为空 (不加载 GT 骨架 I/O)")
chk(cfg["skel_as_glyph_cond"] is True, "skel_as_glyph_cond == True (骨架作为 g 条件)")
chk(cfg["skel_latent_shards_dir"] == "data/top10_style23/shards_std",
    "skel_latent_shards_dir = data/top10_style23/shards_std")
chk(cfg["data_csv"] == "assets/train_top10_style23.csv", "训练集 = top10_style23 (38,583)")
chk(cfg["num_calligraphers"] == 23, "num_calligraphers == 23")
chk(cfg["cond_fusion_norm"] == "split", "cond_fusion_norm == split (与 v24 对齐)")
chk(cfg["max_steps"] == 150000, "max_steps == 150000 (与 v24 对齐)")

print()
print("=" * 78)
print("[2] 训练侧构模: 断言没有 deform_skel")
print("=" * 78)
# 复刻 train.py 的透传: 只造 DiT_2Cond, 不建 dataset/trainer
try:
    import torch
    from src.model.dit import DiT_2Cond
    # 用最小必要参数构一个与配置一致的小模型 (hidden/深度与 DiT-2Cond-S/2 相同)
    g = cfg.get
    kw = dict(
        input_size=32, patch_size=2, in_channels=4, hidden_size=384, depth=12,
        num_heads=6, num_calligraphers=cfg["num_calligraphers"],
        callig_embed_dim=cfg["callig_embed_dim"],
        condition_fusion=cfg["condition_fusion"],
        cond_fusion_norm=cfg["cond_fusion_norm"],
        use_char_cond=not cfg["no_char_cond"],
        # ★ 必须与 train.py 一致: use_glyph_cond = (w_glyph_cond>0) or skel_as_glyph_cond
        use_glyph_cond=(int(cfg.get("w_glyph_cond", 0) or 0) > 0
                        or bool(cfg["skel_as_glyph_cond"])),
        glyph_scale_init=float(cfg["glyph_scale_init"]),
        glyph_inject_layers=int(cfg["glyph_inject_layers"]),
        glyph_inject_mode=str(cfg["glyph_inject_mode"]),
        glyph_embedder_depth=int(cfg["glyph_embedder_depth"]),
        glyph_drop_prob=float(cfg["glyph_drop_prob"]),
        deform_skel=int(cfg["deform_skel"]),
        deform_width=int(cfg["deform_width"]),
        deform_grid=int(cfg["deform_grid"]),
        deform_max_off=float(cfg["deform_max_off"]),
        deform_coarse=int(cfg["deform_coarse"]),
        residual=int(cfg["residual"]),
        res_cap=float(cfg["res_cap"]),
        stroke_mod=int(cfg["stroke_mod"]),
        stroke_cap=float(cfg["stroke_cap"]),
        gate_radius=float(cfg["gate_radius"]),
        deform_dt_ch=int(cfg["deform_dt_ch"]),
        deform_topo=int(cfg.get("deform_topo", 0)),
        deform_ckpt=str(cfg["deform_ckpt"]),
    )
    model = DiT_2Cond(**kw)
    ds = getattr(model, "deform_skel", "MISSING")
    chk(ds is None, "model.deform_skel is None  (SkelNet 未被构造)")
    n_deform = sum(p.numel() for p in model.parameters())
    print("      模型总参数量: %s" % f"{n_deform:,}")
except Exception as e:
    import traceback; traceback.print_exc(); ok = False

print()
print("=" * 78)
print("[3] 前向: 断言 g 直通 (skel_latent -> glyph_embedder), 无形变")
print("=" * 78)
try:
    import torch
    model.eval()
    B = 2
    x2 = torch.randn(B, 4, 32, 32) * 0.1
    t2 = torch.full((B,), 0.5)
    y_c2 = torch.zeros(B, dtype=torch.long)   # 有效槽位 (char 条件关闭时被忽略)
    # g = skel_latent (标准骨架原样)
    gg = torch.randn(B, 4, 32, 32)
    with torch.no_grad():
        out = model(x2, t2, y_c2, y_c2, g=gg)
    o = out[0] if isinstance(out, (tuple, list)) else out
    chk(o is not None, "前向跑通, 输出类型=%s shape=%s" % (
        type(o).__name__, tuple(o.shape) if hasattr(o, 'shape') else "?"))
    # ★ 重要: ZeroAdaLNInjection 是 zero-init -> 未训练时对 g 的变化是恒等(设计如此, 非 bug)。
    #   正确的接线核查是: 注入器**确实存在**、数量正确、且其 zero-init 结构完好。
    n_inj = len(getattr(model, "glyph_inject_at", []) or [])
    chk(n_inj == 4, "glyph 逐层注入器数量 == 4 (实际 %d)" % n_inj)
    chk(getattr(model, "glyph_embedder", None) is not None,
        "glyph_embedder 已构建 (g 有编码通路)")
    # ★ 关键: DiT 的 final_layer 是 **zero-init** (标准做法: 输出层清零,
    #   让训练从 "预测 0" 平滑起步)。因此**未训练**时模型输出恒为 0 ——
    #   无论 g 怎么变, 输出都不变。这**不是**接线问题。
    #   正确的 g 敏感性核查必须**同时打破** final_layer 与逐层注入器的
    #   zero-init, 模拟"训练若干步后"的状态, 再看 g 是否改变输出。
    import torch.nn as nn
    _broke_final = 0
    with torch.no_grad():
        for _p in [model.final_layer.linear.weight, model.final_layer.linear.bias,
                   model.final_layer.adaLN_modulation[1].weight,
                   model.final_layer.adaLN_modulation[1].bias]:
            _p.normal_(0, 0.02)
            _broke_final += 1
    _broke = 0
    if getattr(model, "glyph_injections", None) is not None:
        for inj in model.glyph_injections:
            for m in inj.modules():
                if isinstance(m, nn.Linear) and m.weight.abs().sum() == 0:
                    with torch.no_grad():
                        m.weight.normal_(0, 0.02)
                    _broke += 1
    with torch.no_grad():
        out3 = model(x2, t2, y_c2, y_c2, g=gg)
        out4 = model(x2, t2, y_c2, y_c2, g=torch.randn(B, 4, 32, 32))
    o3 = out3[0] if isinstance(out3, (tuple, list)) else out3
    o4 = out4[0] if isinstance(out4, (tuple, list)) else out4
    d2 = (o3 - o4).abs().mean().item()
    # 同时验证梯度确实回传到 g (输入层加法 glyph_scale=0.6 提供直通梯度)
    gg2 = torch.randn(B, 4, 32, 32, requires_grad=True)
    out5 = model(x2, t2, y_c2, y_c2, g=gg2)
    out5 = out5[0] if isinstance(out5, (tuple, list)) else out5
    out5.sum().backward()
    _gfrac = (gg2.grad.abs() > 1e-12).float().mean().item()
    chk(d2 > 1e-6,
        "打破 final_layer+注入器 zero-init 后 g 确实被消费 "
        "(换 g 输出变化=%.6f, 破零 final=%d/inj=%d)" % (d2, _broke_final, _broke))
    chk(_gfrac > 0.99,
        "d(out)/d(g) 非零比例=%.3f (g 拿到梯度种子, 输入层加法直通)" % _gfrac)
    # 说明: 未破零时输出恒 0 属正常 (final_layer zero-init), 打印以免误读
    print("      (说明: 未破零时 out 恒为 0 —— final_layer zero-init 所致, 非接线问题)")
    chk(not hasattr(model, "deform_skel") or model.deform_skel is None,
        "前向后仍无 deform_skel")
except Exception as e:
    import traceback; traceback.print_exc(); ok = False

print()
print("=" * 78)
print("[4] eval 侧透传一致性 (model_io)")
print("=" * 78)
try:
    import inspect
    from src.eval import model_io
    src = inspect.getsource(model_io)
    chk('deform_skel=gi("deform_skel", 0)' in src,
        'model_io 从 ckpt/config 读 deform_skel (默认 0)')
    chk('deform_ckpt=str(g("deform_ckpt", "") or "")' in src,
        'model_io 从 config 读 deform_ckpt')
except Exception as e:
    import traceback; traceback.print_exc(); ok = False

print()
print("=" * 78)
print("核查结论:", "✅ 全部通过" if ok else "❌ 存在问题, 见上方 ✗")
print("=" * 78)
sys.exit(0 if ok else 1)
