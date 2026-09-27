#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tools/make_inference_sweep_poster.py — 生成 150k 推理参数全景对比海报 (CFG / 步数 / 骨架通路)"""
import os
import sys
import torch
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

sys.stdout.reconfigure(encoding="utf-8")
ROOT = "/root/Workspace/xy/DiT"
os.chdir(ROOT)
sys.path.insert(0, ROOT)

from src.eval.inference import make_eval_cache, sample_latents, build_diffusion
from src.eval.in_mem_eval import _get_vae
from src.eval.model_io import load_model_from_ckpt
from src.utils.callig_script_map import load_callig_script_map

OUT_IMG = "docs/04_experiments/imgs/v24_150k_inference_sweep_poster.png"
EVAL_CSV = "assets/eval_top10_strict_subset84.csv"
CKPT_P = "assets/results/v24_top10_style23/20260927-213913-v24-top10-style23/checkpoints/0150000.pt"

# 挑选 10 个跨书家、跨书体、最具视觉辨识度的样本
# 0: 柳公权-楷-出, 1: 柳公权-楷-禮, 2: 文徵明-行-簿, 4: 苏轼-行-紹, 6: 米芾-行-霞,
# 7: 赵孟頫-隶-照, 11: 褚遂良-楷-逮, 12: 唐寅-行-旅, 16: 颜真卿-楷-强, 21: 柳公权-行-百
PICK_INDICES = [0, 1, 2, 4, 6, 7, 11, 12, 16, 21]

CONFIGS = [
    # (cfg, steps, skel_mode, title, sub, badge)
    (0.7, 50, "deform", "基线: SkelNet + CFG 0.7", "Steps 50 / 默认参数", "碎片 21.6 ✗"),
    (0.5, 50, "deform", "低CFG: SkelNet + CFG 0.5", "Steps 50 / 稀释严重", "碎片 28.4 ✗"),
    (0.9, 50, "deform", "高CFG: SkelNet + CFG 0.9", "Steps 50 / 笔画变实", "碎片 18.5 ✓"),
    (1.0, 50, "deform", "纯条件: SkelNet + CFG 1.0", "Steps 50 / 无噪声", "SSIM 0.5712 🥇"),
    (1.0, 25, "deform", "极速纯条件: CFG 1.0", "Steps 25 / 速度翻倍", "SSIM 0.5714 🥇"),
    (1.0, 50, "boost1.25", "补偿骨架: SkelNet x1.25", "CFG 1.0 / 幅值补偿", "黑度加深"),
    (1.0, 50, "bypass", "直通骨架: 直通 g_std", "CFG 1.0 / 避开幅值衰减", "碎片 6.0 (极连贯)"),
    (0.7, 50, "bypass", "直通骨架: 直通 g_std", "CFG 0.7 / 对照组", "碎片 7.5"),
]

def get_font(size, bold=False):
    candidates = [
        "tools/fonts/simkai.ttf",
        "tools/fonts/SimHei.ttf",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    ]
    for fp in candidates:
        if os.path.exists(fp):
            try:
                return ImageFont.truetype(fp, size)
            except:
                pass
    return ImageFont.load_default()

def main():
    print(f"=== [海报生成] 加载 150k 模型检查点: {CKPT_P} ===")
    model, _ = load_model_from_ckpt(CKPT_P, device="cuda", use_ema=True, verbose=False)
    model.eval()

    csmap = load_callig_script_map("assets/callig_script_id_map_top10.json")

    # 载入待测样本
    df_eval = pd.read_csv(EVAL_CSV)
    cache = make_eval_cache(
        EVAL_CSV, None, None, 256, max(PICK_INDICES) + 1, 8, 4, 0.18215,
        skel_latent_shards_dir="data/50k_v2_glyph15k/shards_std",
        callig_script_map=csmap
    )

    vae = _get_vae("cuda")
    orig_deform = model.deform_skel

    # 提取选定样本的数据切片
    sel_noise = cache["noise"][PICK_INDICES]
    sel_conds = [cache["conds"][idx] for idx in PICK_INDICES]
    sel_skels = cache["skels_latent"][PICK_INDICES]
    sel_gts = ((cache["gts"][PICK_INDICES].to("cuda") + 1) / 2)

    # 1. 解码输入标准骨架 (Row 0)
    with torch.no_grad():
        dec_skel_im = ((vae.decode((sel_skels / 0.18215).to("cuda")).sample.clamp(-1, 1) + 1) / 2).cpu()

    # 2. 对每个配置生成图像
    gen_results = {}
    for cfg, steps, skel_mode, t, s, b in CONFIGS:
        print(f"推演配置: {t} ...")
        diff = build_diffusion(steps, "flow")
        if skel_mode == "bypass":
            model.deform_skel = None
        else:
            model.deform_skel = orig_deform

        with torch.no_grad():
            cur_skel = sel_skels.clone()
            if skel_mode == "boost1.25":
                cur_skel = cur_skel * 1.25

            # 4 张一批推演防止显存峰值
            sub_preds = []
            for b_idx in range(0, len(PICK_INDICES), 4):
                sub_n = sel_noise[b_idx:b_idx+4]
                sub_c = sel_conds[b_idx:b_idx+4]
                sub_s = cur_skel[b_idx:b_idx+4]
                lat = sample_latents(model, diff, sub_n, sub_c, cfg, 4, "cuda", skel=sub_s)
                im = ((vae.decode((lat / 0.18215).to("cuda")).sample.clamp(-1, 1) + 1) / 2).cpu()
                sub_preds.append(im)
            gen_results[(cfg, steps, skel_mode)] = torch.cat(sub_preds, dim=0)

    model.deform_skel = orig_deform

    # 3. 拼装大图
    print("开始合成全景海报画布...")
    cell_size = 180
    row_header_w = 270
    top_header_h = 130
    col_header_h = 56
    padding = 16
    gap = 6

    n_cols = len(PICK_INDICES)
    n_rows = 1 + len(CONFIGS) + 1  # Input g + 8 Configs + GT = 10 rows

    canvas_w = row_header_w + n_cols * cell_size + (n_cols - 1) * gap + padding * 2
    canvas_h = top_header_h + col_header_h + n_rows * cell_size + (n_rows - 1) * gap + padding * 2

    canvas = Image.new("RGB", (canvas_w, canvas_h), (248, 249, 251))
    draw = ImageDraw.Draw(canvas)

    title_font = get_font(28, bold=True)
    sub_title_font = get_font(15)
    row_title_font = get_font(16, bold=True)
    row_sub_font = get_font(12)
    col_font = get_font(15, bold=True)
    col_sub_font = get_font(13)

    # 绘制顶部标题
    draw.rectangle([0, 0, canvas_w, top_header_h], fill=(30, 36, 48))
    draw.text((padding + 10, 22), "马良 (Callig-DiT) v24_top10 终点 150k 推理参数扫描对比海报", font=title_font, fill=(255, 255, 255))
    subtitle = "严苛零样本 (Strict) 笔画连贯度与断墨诊断：CFG 强度 (0.5~1.0) / ODE 步数 (25~75) / 骨架直通 vs 变形"
    draw.text((padding + 12, 66), subtitle, font=sub_title_font, fill=(185, 195, 210))
    summary_tag = "核心实测结论：CFG 严禁 <0.9 (null-token 导致笔画撕裂)；CFG 1.0 + 25 步达到最佳成字质量 (SSIM 0.5714，中位 0.5823)！"
    draw.text((padding + 12, 92), summary_tag, font=sub_title_font, fill=(255, 215, 110))

    # 绘制列头
    y_col_hdr = top_header_h + padding
    for c_idx, sample_idx in enumerate(PICK_INDICES):
        r = df_eval.iloc[sample_idx]
        x = row_header_w + padding + c_idx * (cell_size + gap)
        draw.rectangle([x, y_col_hdr, x + cell_size, y_col_hdr + col_header_h - 4], fill=(235, 238, 243), outline=(210, 215, 225))
        tag1 = f"{r['character']} · {r['script']}"
        tag2 = f"{r['calligrapher']}"
        draw.text((x + 10, y_col_hdr + 8), tag1, font=col_font, fill=(20, 25, 35))
        draw.text((x + 10, y_col_hdr + 30), tag2, font=col_sub_font, fill=(90, 100, 115))

    # 行数据配置
    row_data = []
    # Row 0: Input g
    row_data.append({
        "title": "输入标准字骨架 (g)",
        "sub": "无偏印刷体拓扑引导",
        "badge": "输入条件",
        "badge_color": (240, 243, 248),
        "ims": dec_skel_im
    })
    # Row 1..8: Configs
    for cfg, steps, skel_mode, t, s, b in CONFIGS:
        b_color = (230, 245, 235) if "🥇" in b else ((255, 235, 235) if "✗" in b else (240, 242, 246))
        row_data.append({
            "title": t,
            "sub": s,
            "badge": b,
            "badge_color": b_color,
            "ims": gen_results[(cfg, steps, skel_mode)]
        })
    # Row 9: GT
    row_data.append({
        "title": "真实名家真迹 (GT)",
        "sub": "未见测试集原拓/墨迹",
        "badge": "Ground Truth",
        "badge_color": (255, 240, 230),
        "ims": sel_gts.cpu()
    })

    # 逐行画图
    y_start = y_col_hdr + col_header_h
    for r_idx, r_item in enumerate(row_data):
        y = y_start + r_idx * (cell_size + gap)

        # 行标签栏
        draw.rectangle([padding, y, row_header_w - 10, y + cell_size], fill=(255, 255, 255), outline=(215, 220, 230), width=1)
        draw.text((padding + 12, y + 25), r_item["title"], font=row_title_font, fill=(20, 30, 45))
        draw.text((padding + 12, y + 60), r_item["sub"], font=row_sub_font, fill=(100, 110, 125))

        # 状态徽章
        draw.rectangle([padding + 12, y + 105, row_header_w - 24, y + 145], fill=r_item["badge_color"], outline=(200, 215, 205))
        draw.text((padding + 20, y + 116), r_item["badge"], font=row_sub_font, fill=(35, 100, 60) if "🥇" in r_item["badge"] else ((160, 40, 40) if "✗" in r_item["badge"] else (60, 70, 85)))

        # 单元格图片
        for c_idx in range(n_cols):
            x = row_header_w + padding + c_idx * (cell_size + gap)
            im_tensor = r_item["ims"][c_idx]
            im_np = (im_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            im_pil = Image.fromarray(im_np).resize((cell_size, cell_size), Image.LANCZOS)

            canvas.paste(im_pil, (x, y))
            draw.rectangle([x, y, x + cell_size, y + cell_size], outline=(220, 224, 232), width=1)

    os.makedirs(os.path.dirname(OUT_IMG), exist_ok=True)
    canvas.save(OUT_IMG, optimize=True)
    print(f"🎉 推理参数扫描全景海报已生成: {OUT_IMG}")
    sz_mb = os.path.getsize(OUT_IMG) / 1024 / 1024
    print(f"   尺寸: {canvas_w}x{canvas_h} px, 大小: {sz_mb:.2f} MB")

if __name__ == "__main__":
    main()
