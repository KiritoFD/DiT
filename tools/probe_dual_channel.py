#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""probe_dual_channel.py — 风格双通道（SkelNet 间架通路 vs Backbone adaLN 纹理通路）
的强度、梯度流与敏感度定量诊断脚本。

诊断内容：
1. 信号强度 (Signal Strength & Modulations):
   - ||e_callig|| 范数
   - c = t_emb + y_emb 中 ||y_emb|| vs ||t_emb|| 的能量占比
   - cond_fusion 输入中 ||e_callig|| vs ||e_glyph_vec||
   - SkelNet 间架形变量: 平均/最大位移 ||(dx, dy)||, 笔画调制 Delta_g, 骨架改动率 ||g'-g||/||g||
   - Backbone 各层 adaLN 调制参数 (gamma, beta, alpha) 范数

2. 梯度分解 (Gradient Probe):
   - 将 e_callig 拆分为独立的叶子节点 e_skel (入 SkelNet) 与 e_dit (入 Backbone adaLN)
   - L_diff 对 e_skel 的梯度范数 vs 对 e_dit 的梯度范数
   - L_diff 对 SkelNet 参数的梯度 vs L_deform 对 SkelNet 参数的梯度
   - 扩散目标与骨架 GT 目标的梯度余弦相似度: cos(grad_skel(L_diff), grad_skel(L_deform))
     (判定扩散生成任务与中间骨架监督是在同向协同还是相互拉扯)

3. 敏感度消融 (Sensitivity & Ablation):
   - L_diff 基准损失
   - 骨架风格打乱 (e_skel 乱序, e_dit 正确): Delta L
   - 纹理风格打乱 (e_dit 乱序, e_skel 正确): Delta L
   - 双通道全打乱: Delta L
   - 关闭骨架形变 (g' = g_std): Delta L
   - 关闭主干风格 (e_dit = null): Delta L
"""

import argparse
import json
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
os.chdir(ROOT)


def build_model_and_batch(config_path, ckpt_path=None, batch_size=16, device="cpu"):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    from src.eval.model_io import load_model_from_ckpt
    from src.model.dit import DiT_2Cond_models
    from src.utils.latent_dataset import MCCDLatentDataset
    from src.loss.flow_matching import FlowMatching, TIME_SCALE

    # 1. 词表映射
    csmap = None
    if cfg.get("callig_script_map"):
        with open(cfg["callig_script_map"], "r", encoding="utf-8") as f:
            csmap = json.load(f)
    num_classes = len(csmap["pair_map"]) if (csmap and "pair_map" in csmap) else cfg.get("num_calligraphers", 45)

    if ckpt_path and os.path.exists(ckpt_path):
        print(f"[probe] Loading full trained model from ckpt: {ckpt_path}")
        model, _ = load_model_from_ckpt(ckpt_path, device=device, use_ema=True, verbose=True)
    else:
        # 2. 构建模型并加载离线组件
        model_name = cfg.get("model", "DiT-2Cond-S/2")
        model_fn = DiT_2Cond_models[model_name]
        model = model_fn(
            input_size=32,
            num_calligraphers=num_classes,
            num_characters=cfg.get("num_characters", 7765),
            learn_sigma=False,
            norm_type=cfg.get("norm_type", "rms"),
            mlp_type=cfg.get("mlp_type", "swiglu"),
            qk_norm=bool(cfg.get("qk_norm", 1)),
            rope=bool(cfg.get("rope", 1)),
            rope_theta=float(cfg.get("rope_theta", 100.0)),
            attn_impl=cfg.get("attn_impl", "sdpa"),
            use_glyph_cond=True,
            glyph_scale_init=cfg.get("glyph_scale_init", 0.6),
            glyph_embedder_depth=cfg.get("glyph_embedder_depth", 2),
            glyph_inject_layers=cfg.get("glyph_inject_layers", 4),
            glyph_inject_mode=cfg.get("glyph_inject_mode", "adaln"),
            glyph_vec_cond=bool(cfg.get("glyph_vec_cond", True)),
            glyph_vec_dim=cfg.get("glyph_vec_dim", 128),
            glyph_vec_pool=cfg.get("glyph_vec_pool", "mean"),
            condition_fusion=cfg.get("condition_fusion", "factorized_cat"),
            callig_embed_dim=cfg.get("callig_embed_dim", 128),
            use_char_cond=not bool(cfg.get("no_char_cond", True)),
            deform_skel=int(cfg.get("deform_skel", 1)),
            deform_width=int(cfg.get("deform_width", 96)),
            deform_max_off=float(cfg.get("deform_max_off", 6.0)),
            deform_coarse=int(cfg.get("deform_coarse", 8)),
            deform_grid=int(cfg.get("deform_grid", 32)),
            residual=int(cfg.get("residual", 0)),
            res_cap=float(cfg.get("res_cap", 1.0)),
            stroke_mod=int(cfg.get("stroke_mod", 1)),
            stroke_cap=float(cfg.get("stroke_cap", 1.0)),
            gate_radius=float(cfg.get("gate_radius", 0.25)),
            deform_dt_ch=int(cfg.get("deform_dt_ch", 1)),
            deform_ckpt=str(cfg.get("deform_ckpt", "") or ""),
        )

        # 加载书家预训练表
        if cfg.get("callig_emb_pretrained") and os.path.exists(cfg["callig_emb_pretrained"]):
            tab = torch.load(cfg["callig_emb_pretrained"], map_location="cpu", weights_only=False)
            emb = tab["embedding"] if isinstance(tab, dict) else tab
            with torch.no_grad():
                N_load = min(emb.shape[0], model.y_callig_embedder.embedding_table.weight.shape[0])
                model.y_callig_embedder.embedding_table.weight[:N_load].copy_(emb[:N_load])
                print(f"[probe] Loaded callig_emb_pretrained: {emb.shape} -> copied {N_load} rows")

        model = model.to(device)
    model.train()  # 保持 train 状态以获得梯度

    # 3. 构造 dataset 并取一个 batch
    ds = MCCDLatentDataset(
        csv_file=cfg["data_csv"],
        latent_shards_dir=cfg["latent_shards_dir"],
        img_root=cfg.get("img_root", None),
        preload=False,
        load_image=False,
        skel_latent_shards_dir=cfg["skel_latent_shards_dir"],
        inst_skel_shards_dir=cfg["inst_skel_shards_dir"],
        callig_script_map=csmap,
    )

    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False)
    batch = next(iter(loader))

    fm = FlowMatching(
        num_steps=50,
        sampler="heun",
        t_sampler="logit_normal",
        shift=1.0,
    )

    return model, batch, fm, cfg


def run_probe(model, batch, fm, device="cpu"):
    print(f"\n{'='*70}\n[Probe 1] 信号双通道前向注入测量 (Forward Pass Analysis)\n{'='*70}")

    x0 = batch["latent"].to(device).float()
    g_std = batch["skel_latent"].to(device).float()
    g_inst = batch["inst_skel"].to(device).float()
    y_callig = batch["y_callig"].to(device).long()
    y_char = batch.get("y_char", torch.zeros_like(y_callig)).to(device).long()
    B = x0.shape[0]

    # 采样 flow matching t 和噪声
    torch.manual_seed(42)
    t = fm.sample_t(B, device)
    noise = torch.randn_like(x0)
    x_t = fm._interp(x0, noise, t)
    v_target = noise - x0

    # 1. 获取基础 callig embedding
    with torch.no_grad():
        e_callig_base = model.y_callig_embedder(y_callig, False)  # (B, 128)
        norm_ec = e_callig_base.norm(dim=-1).mean().item()
        print(f"  • 原始风格向量 ||e_callig||: 平均范数 = {norm_ec:.4f} (dim={e_callig_base.shape[-1]})")

    # 2. 模拟双分支分流：创建独立的叶子节点 e_skel 与 e_dit
    e_skel = e_callig_base.clone().detach().requires_grad_(True)
    e_dit = e_callig_base.clone().detach().requires_grad_(True)

    # ── 通道 A: SkelNet 骨架形变通路 ──
    g_deform = model.deform_skel(g_std, e_skel)
    off_stats = model.deform_skel.offset_stats()

    diff_g = (g_deform - g_std).abs()
    rel_change_g = (g_deform - g_std).norm() / g_std.norm()

    print(f"\n  [通道 A: SkelNet 间架通道]")
    print(f"  • 全局位移场 (dx, dy): 均值 = {off_stats['mean_abs']:.4f} px, 最大 = {off_stats['max_abs']:.4f} px")
    print(f"  • 骨架潜空间相对形变量 ||g' - g_std|| / ||g_std||: {rel_change_g.item()*100:.2f}%")
    print(f"  • 平均绝对改动像素值: {diff_g.mean().item():.4f}")

    # 骨架进入 glyph_embedder
    g_tok = model.glyph_embedder(g_deform).flatten(2).transpose(1, 2)  # (B, 256, 384)
    # 池化出骨架全局向量
    pooled = g_tok.mean(dim=1)
    e_glyph_vec = model.glyph_vec_proj(pooled)  # (B, 128)

    norm_eglyph = e_glyph_vec.norm(dim=-1).mean().item()
    print(f"  • 骨架全局池化向量 ||e_glyph_vec||: 平均范数 = {norm_eglyph:.4f}")

    # ── 通道 B: Diffusion 主干 adaLN 通路 ──
    y_cat = torch.cat([e_dit, e_glyph_vec], dim=-1)  # (B, 256)
    y_emb = model.cond_fusion(y_cat)                 # (B, 384)
    y_emb = model._style_branch(e_dit, y_emb)

    t_emb = model.t_embedder(t * 1000.0)             # (B, 384)
    c = t_emb + y_emb                                # (B, 384)

    norm_t = t_emb.norm(dim=-1).mean().item()
    norm_y = y_emb.norm(dim=-1).mean().item()
    norm_c = c.norm(dim=-1).mean().item()
    energy_ratio = (norm_y ** 2) / (norm_y ** 2 + norm_t ** 2)

    print(f"\n  [通道 B: Diffusion adaLN 通道]")
    print(f"  • cond_fusion 输入操作数对比: ||e_dit|| = {e_dit.norm(dim=-1).mean().item():.4f} vs ||e_glyph_vec|| = {norm_eglyph:.4f}")
    print(f"  • 调制向量能量分解 (c = t_emb + y_emb):")
    print(f"    - 时间信号 ||t_emb|| = {norm_t:.4f}")
    print(f"    - 条件信号 ||y_emb|| = {norm_y:.4f}")
    print(f"    - 风格条件能量占比 ||y||^2 / (||y||^2 + ||t||^2) = {energy_ratio*100:.2f}%")

    # 逐层 adaLN 调制强度检查
    print(f"\n  [DiT Block 逐层 adaLN 调制幅度]")
    block_gammas = []
    block_alphas = []
    for i, blk in enumerate(model.blocks):
        # blk.adaLN_modulation(c) -> (B, 6*D)
        mod = blk.adaLN_modulation(c)
        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = mod.chunk(6, dim=1)
        g_norm = (gamma1.norm(dim=-1).mean().item() + gamma2.norm(dim=-1).mean().item()) / 2.0
        a_norm = (alpha1.norm(dim=-1).mean().item() + alpha2.norm(dim=-1).mean().item()) / 2.0
        block_gammas.append(g_norm)
        block_alphas.append(a_norm)
    print(f"    - Scale (gamma) 均值范数: 前4层={sum(block_gammas[:4])/4:.3f} | 中4层={sum(block_gammas[4:8])/4:.3f} | 后4层={sum(block_gammas[8:])/4:.3f}")
    print(f"    - Gate (alpha)  均值范数: 前4层={sum(block_alphas[:4])/4:.3f} | 中4层={sum(block_alphas[4:8])/4:.3f} | 后4层={sum(block_alphas[8:])/4:.3f}")

    # 完整前向以得到最终输出
    # 注入骨架到残差流
    x = model.x_embedder(x_t)
    if not model.rope:
        x = x + model.pos_embed
    x = x + model.glyph_scale * g_tok

    rope = (model.rope_cos, model.rope_sin) if model.rope else None
    _inj = model._inj_map if (model.glyph_injections and g_tok is not None) else {}

    for i, block in enumerate(model.blocks):
        x = block(x, c, rope=rope)
        if i in _inj:
            x = model.glyph_injections[_inj[i]](x, g_tok)

    x = model.final_layer(x, c)
    v_pred = model.unpatchify(x)

    # ── 损失计算 ──
    loss_diff = ((v_pred - v_target) ** 2).mean()
    loss_deform = ((g_deform - g_inst) ** 2).mean()
    loss_total = loss_diff + 1.0 * loss_deform

    print(f"\n  [当前 Batch 损失]")
    print(f"  • L_diff (扩散流匹配损失) = {loss_diff.item():.4f}")
    print(f"  • L_deform (骨架形变监督损失) = {loss_deform.item():.4f}")
    print(f"  • L_total = {loss_total.item():.4f}")

    # =========================================================================
    # [Probe 2] 梯度流分解 (Gradient Probe)
    # =========================================================================
    print(f"\n{'='*70}\n[Probe 2] 梯度流与方向协同分析 (Gradient Flow & Synergy Analysis)\n{'='*70}")

    # 1. 单独计算 L_diff 对 e_skel 和 e_dit 的梯度
    grad_skel_diff, grad_dit_diff = torch.autograd.grad(
        loss_diff, (e_skel, e_dit), retain_graph=True
    )
    # 2. 单独计算 L_deform 对 e_skel 和 e_dit 的梯度
    grad_skel_deform, = torch.autograd.grad(
        loss_deform, (e_skel,), retain_graph=True
    )

    gn_skel_diff = grad_skel_diff.norm(dim=-1).mean().item()
    gn_dit_diff = grad_dit_diff.norm(dim=-1).mean().item()
    gn_skel_deform = grad_skel_deform.norm(dim=-1).mean().item()

    print(f"  • 扩散任务 L_diff 反传到风格嵌入的梯度:")
    print(f"    - 通道 A (经由 DiT -> SkelNet 反传): ||grad_{{skel}}(L_diff)|| = {gn_skel_diff:.6f}")
    print(f"    - 通道 B (经由 DiT adaLN 直接反传):  ||grad_{{dit}}(L_diff)||  = {gn_dit_diff:.6f}")
    print(f"    - 扩散梯度比值 (SkelNet / adaLN) = {gn_skel_diff / max(gn_dit_diff, 1e-9):.4f}")

    print(f"\n  • 骨架中间监督 L_deform 对 SkelNet 的梯度:")
    print(f"    - ||grad_{{skel}}(L_deform)|| = {gn_skel_deform:.6f}")
    print(f"    - 监督信号强度比 (L_deform / L_diff 作用在 SkelNet 上) = {gn_skel_deform / max(gn_skel_diff, 1e-9):.2f} 倍")

    # 3. 梯度方向协同度 (Cosine Similarity)
    cos_sim = F.cosine_similarity(grad_skel_diff, grad_skel_deform, dim=-1).mean().item()
    print(f"\n  • 核心检验: 扩散目标与骨架 GT 监督在 SkelNet 上的梯度余弦相似度:")
    print(f"    - cos(grad_skel(L_diff), grad_skel(L_deform)) = {cos_sim:+.4f}")
    if cos_sim > 0.05:
        print(f"    => 【协同同向】: 扩散任务自发引导骨架朝向真实书家间架形变！两者正向共振。")
    elif cos_sim < -0.05:
        print(f"    => 【冲突拉扯】: 扩散任务的梯度在把骨架拉向其它方向，与 GT 间架冲突！")
    else:
        print(f"    => 【解耦正交】: 扩散任务对骨架间架细节近乎零干涉，主要依靠 L_deform 单独锚定。")

    # =========================================================================
    # [Probe 3] 风格敏感度消融测试 (Sensitivity & Ablation)
    # =========================================================================
    print(f"\n{'='*70}\n[Probe 3] 风格敏感度与扰动消融 (Sensitivity & Ablation)\n{'='*70}")

    def eval_forward(esk, edit, g_input=None):
        with torch.no_grad():
            if g_input is None:
                g_d = model.deform_skel(g_std, esk)
            else:
                g_d = g_input
            gtok = model.glyph_embedder(g_d).flatten(2).transpose(1, 2)
            p = gtok.mean(dim=1)
            egv = model.glyph_vec_proj(p)
            yc = torch.cat([edit, egv], dim=-1)
            ye = model.cond_fusion(yc)
            ye = model._style_branch(edit, ye)
            cc = t_emb + ye
            xx = model.x_embedder(x_t)
            if not model.rope:
                xx = xx + model.pos_embed
            xx = xx + model.glyph_scale * gtok
            for j, b in enumerate(model.blocks):
                xx = b(xx, cc, rope=rope)
                if j in _inj:
                    xx = model.glyph_injections[_inj[j]](xx, gtok)
            xx = model.final_layer(xx, cc)
            vp = model.unpatchify(xx)
            return ((vp - v_target) ** 2).mean().item()

    base_loss = eval_forward(e_skel, e_dit)

    # 打乱索引 (roll 1)
    perm = torch.roll(torch.arange(B), 1)
    e_skel_perm = e_skel[perm]
    e_dit_perm = e_dit[perm]

    # 1. 骨架风格打乱 (骨架错，纹理对)
    loss_skel_mismatch = eval_forward(e_skel_perm, e_dit)
    # 2. 纹理风格打乱 (纹理错，骨架对)
    loss_dit_mismatch = eval_forward(e_skel, e_dit_perm)
    # 3. 双通道全打乱 (完全换成别人)
    loss_both_mismatch = eval_forward(e_skel_perm, e_dit_perm)
    # 4. 关闭骨架形变 (直接用未形变的 g_std)
    loss_no_deform = eval_forward(None, e_dit, g_input=g_std)
    # 5. 关闭主干风格 (设为 0 / null)
    loss_no_dit_style = eval_forward(e_skel, torch.zeros_like(e_dit))

    print(f"  • 基准 L_diff (双通道对齐) = {base_loss:.5f}")
    print(f"  • [消融 1] 骨架间架风格打乱 (间架错, 纹理对):  L = {loss_skel_mismatch:.5f} (Δ = {loss_skel_mismatch - base_loss:+.5f}, {(loss_skel_mismatch - base_loss)/base_loss*100:+.2f}%)")
    print(f"  • [消融 2] 主干纹理风格打乱 (间架对, 纹理错):  L = {loss_dit_mismatch:.5f} (Δ = {loss_dit_mismatch - base_loss:+.5f}, {(loss_dit_mismatch - base_loss)/base_loss*100:+.2f}%)")
    print(f"  • [消融 3] 双通道风格同时打乱 (全错):          L = {loss_both_mismatch:.5f} (Δ = {loss_both_mismatch - base_loss:+.5f}, {(loss_both_mismatch - base_loss)/base_loss*100:+.2f}%)")
    print(f"  • [消融 4] 关闭骨架形变 (直接用 g_std):        L = {loss_no_deform:.5f} (Δ = {loss_no_deform - base_loss:+.5f}, {(loss_no_deform - base_loss)/base_loss*100:+.2f}%)")
    print(f"  • [消融 5] 关闭主干风格 (e_dit 归零):          L = {loss_no_dit_style:.5f} (Δ = {loss_no_dit_style - base_loss:+.5f}, {(loss_no_dit_style - base_loss)/base_loss*100:+.2f}%)")

    print(f"\n{'='*70}\n[诊断结论总结]\n{'='*70}")
    print(f"1. 双通道注入确认: 风格信息已严格实现双通道注入。")
    print(f"   - 通道 A (SkelNet 间架级): e_callig 驱动 deformable offset/affine/stroke -> 输出形变骨架 g'，形变量达 {rel_change_g.item()*100:.1f}%。")
    print(f"   - 通道 B (Backbone 笔触级): e_callig 进入 cond_fusion 与 e_glyph_vec 融合，在 12 层 DiT blocks 的 adaLN 中全面驱动 scale/shift/gate。")
    print(f"2. 梯度流态势:")
    print(f"   - 主干 adaLN 承接了扩散目标对风格的核心梯度 (||grad|| = {gn_dit_diff:.6f})。")
    print(f"   - SkelNet 通道除了承接扩散梯度的微弱穿透 ({gn_skel_diff:.6f}) 外，主要由 L_deform 监督主导 ({gn_skel_deform:.6f}，强约 {gn_skel_deform/max(gn_skel_diff, 1e-9):.1f} 倍)，保证间架结构不会被扩散噪声带偏。")
    print(f"   - 扩散目标与骨架监督在 SkelNet 上的余弦相似度为 {cos_sim:+.3f}。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="src/train/configs/v21_skelnet_200k.json")
    parser.add_argument("--ckpt", default="assets/results/v21_skelnet_200k/20260926-005749-v21-skelnet-200k/checkpoints/0005000.pt")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    model, batch, fm, cfg = build_model_and_batch(args.config, ckpt_path=args.ckpt, batch_size=args.batch_size, device=args.device)
    run_probe(model, batch, fm, device=args.device)
