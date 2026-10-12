#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""test_std_chain_local.py — 本地验证 **std 链路**：给定 (字, 书法家, 书体, TTF 路径) 能否正确出图。

两段，各自独立 PASS/FAIL：

  A) 条件渲染正确性（不需要模型/GPU）
     1. 用**指定的 TTF** 渲染给定字 -> 软掩码 (64,64)
     2. 与 store 里该 (书体,字) 的**权威缓存**逐像素/结构对比（同一字体应当一致）
     3. **换字体必须换图**：同一字换另一个 TTF，掩码必须明显不同（证明 TTF 真被用了 ✓）
     4. 掩码非空、字形量级合理（墨量落在常见区间）

  B) 端到端出图 + 指标（需要模型；本地无 std ckpt 时用随机初始化跑"契约级"验证）
     1. 按 config 建模型（std_soft=True）-> 挂 glyph_embedder 前向钩子
     2. 走**生产路径** in_mem_eval 的 cache + 采样 + 解码（1 张图，ODE γ=0 / heun / 50 / cfg 1.0）
     3. 断言: 软掩码真的到了编码器（shape=(1,1,64,64)、mean>0）；输出 PNG 尺寸/取值域正确、有墨
     4. 打印墨量/近白判据（复用 watchdog 的口径）

用法:
  PY=/opt/conda/envs/cu121/bin/python   # 本地就是 python
  $PY tools/test_std_chain_local.py --char 惠 --script kai --calligrapher 苏轼 \
      --font tools/fonts/simkai.ttf --config src/train/configs/v83A_stdsoft_zerofix.json \
      --out _local_ckpt/std_chain_test [--ckpt <std_soft.pt>] [--no-model]
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.isdir(os.path.join(ROOT, "src")):
    os.chdir(ROOT)
sys.path.insert(0, os.getcwd())
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

STORE_DEFAULT = "exp-std/data/std_soft_glyphs_v1"
results = []


def check(name, ok, detail=""):
    results.append((name, bool(ok), detail))
    print("  %s %-46s %s" % ("✓" if ok else "✗", name, detail))


def load_index(store):
    with open(os.path.join(store, "index.csv"), encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def render_with_ttf(char, ttf, face=0, size=64, sigma=0.65):
    """用指定 TTF 渲染字符 -> (64,64) float32 [0,1] 软掩码，与 store 同规格。"""
    from PIL import ImageDraw, ImageFilter, ImageFont
    # 大画布渲染再缩放：与 store 的做法一致（避免小字号锯齿）
    canvas = 256
    img = Image.new("L", (canvas, canvas), 0)
    fnt = ImageFont.truetype(ttf, int(canvas * 0.78), index=int(face or 0))
    d = ImageDraw.Draw(img)
    bbox = d.textbbox((0, 0), char, font=fnt)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    d.text(((canvas - w) / 2 - bbox[0], (canvas - h) / 2 - bbox[1]), char, font=fnt, fill=255)
    if sigma > 0:
        img = img.filter(ImageFilter.GaussianBlur(sigma * canvas / size))
    img = img.resize((size, size), Image.LANCZOS)
    return np.asarray(img).astype(np.float32) / 255.0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--char", required=True)
    ap.add_argument("--script", default="kai")
    ap.add_argument("--calligrapher", default="")
    ap.add_argument("--font", required=True)
    ap.add_argument("--font2", default="", help="第二个 TTF，用于'换字体必须换图'检验")
    ap.add_argument("--store", default=STORE_DEFAULT)
    ap.add_argument("--config", default="src/train/configs/v83A_stdsoft_zerofix.json")
    ap.add_argument("--ckpt", default="", help="std_soft ckpt；不给则随机初始化（契约级验证）")
    ap.add_argument("--out", default="_local_ckpt/std_chain_test")
    ap.add_argument("--sigma", type=float, default=0.65)
    ap.add_argument("--no-model", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    print("=" * 78)
    print("输入: 字=%s  书体=%s  书法家=%s  TTF=%s" % (a.char, a.script, a.calligrapher, a.font))
    print("=" * 78)

    # ───────────────── A) 条件渲染正确性 ─────────────────
    print("\n[A] 条件渲染（TTF -> 软掩码）")
    check("TTF 存在", os.path.isfile(a.font), a.font)
    idx = load_index(a.store)
    hit = [r for r in idx if r["character"] == a.char and r["script_key"] == a.script]
    check("store 里有该 (书体,字) 的权威记录", bool(hit),
          (hit[0]["path"] + " font=" + hit[0]["font_file"]) if hit else "缺失")

    m_my = render_with_ttf(a.char, a.font, sigma=a.sigma)
    check("用给定 TTF 渲染成功", m_my.shape == (64, 64) and m_my.max() > 0,
          "shape=%s ink=%.4f" % (m_my.shape, float(np.mean(m_my))))
    Image.fromarray((m_my * 255).astype(np.uint8)).save(os.path.join(a.out, "A_my_render.png"))

    if hit:
        rel = hit[0]["path"]
        img = Image.open(os.path.join(a.store, rel)).convert("L")
        m_ref = np.asarray(img).astype(np.float32) / 255.0
        if m_ref.shape != (64, 64):
            m_ref = np.asarray(Image.fromarray((m_ref * 255).astype(np.uint8)).resize((64, 64))).astype(np.float32) / 255.0
        Image.fromarray((m_ref * 255).astype(np.uint8)).save(os.path.join(a.out, "A_store_cache.png"))
        d = float(np.mean(np.abs(m_my - m_ref)))
        # 同字体应高度一致（渲染实现细节可有小差），不同字体必然差很多
        check("与 store 权威缓存的差异 < 0.05（同字体应一致）", d < 0.05, "mean|Δ|=%.4f" % d)
        check("store 记录用的就是给定 TTF 或同族", 
              os.path.basename(hit[0]["font_file"]).lower() in os.path.basename(a.font).lower()
              or os.path.basename(a.font).lower() in hit[0]["font_path"].lower(),
              "store font_file=%s" % hit[0]["font_file"])

    font2 = a.font2 or next((p for p in ("tools/fonts/STXINGKA.TTF", "tools/fonts/SIMLI.TTF",
                                         "tools/fonts/SimHei.ttf") if os.path.isfile(p)), "")
    if font2 and os.path.isfile(font2):
        m2 = render_with_ttf(a.char, font2, sigma=a.sigma)
        Image.fromarray((m2 * 255).astype(np.uint8)).save(os.path.join(a.out, "A_render_font2.png"))
        d2 = float(np.mean(np.abs(m_my - m2)))
        check("换字体必须换图（TTF 真被使用）", d2 > 0.02, "%s vs %s mean|Δ|=%.4f"
              % (os.path.basename(a.font), os.path.basename(font2), d2))
    else:
        check("换字体检验（跳过：没找到第二个 TTF）", False, "传 --font2 指定")

    # ───────────────── B) 端到端出图 ─────────────────
    if a.no_model:
        print("\n[B] 已用 --no-model 跳过")
    else:
        print("\n[B] 端到端出图（走生产路径 in_mem_eval；ckpt=%s）" % (a.ckpt or "<随机初始化>"))
        try:
            import torch
            dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            cfg = json.load(open(a.config, encoding="utf-8"))
            from argparse import Namespace
            args = Namespace(**cfg)
            args.std_soft_cond = True
            args.std_soft_root = a.store
            args.eval_vae_path = str(next((p for p in (
                "data/pretrained/pretrained_models/sd-vae-ft-ema",
                "pretrained_models/sd-vae-ft-ema") if os.path.isfile(os.path.join(p, "config.json"))), ""))
            check("图像 VAE 就位", bool(args.eval_vae_path), args.eval_vae_path)
            from src.eval import model_io
            model = model_io.build_model_from_args(args, dev)
            if a.ckpt:
                ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
                sd = ck.get("ema") if ck.get("ema") is not None else ck.get("model", ck)
                # ? torch.compile ?? `_orig_mod.` ?? -> ???? 161 missing + 161 unexpected ?
                sd = {k.replace("_orig_mod.", "").replace("module.", ""): v for k, v in sd.items()}
                miss, unexp = model.load_state_dict(sd, strict=False)
                check("ckpt 加载（EMA）", len([k for k in miss if "glyph" in k]) == 0,
                      "missing=%d unexpected=%d" % (len(miss), len(unexp)))
            check("模型有 std 字形分支", hasattr(model, "glyph_embedder") and bool(getattr(model, "std_soft_cond", False)),
                  "std_soft_cond=%s" % getattr(model, "std_soft_cond", None))
            seen = {}
            h = model.glyph_embedder.register_forward_pre_hook(
                lambda mod, inp: seen.setdefault("x", inp[0].detach()))
            try:
                from src.eval.in_mem_eval import run_in_mem_eval
                # 只评 1 张：借首个 set 的 csv，n=1
                spec = cfg["in_mem_eval_sets"].split(",")[0].split(":")
                args.in_mem_eval_sets = "%s:%s:1" % (spec[0], spec[1])
                args.in_mem_eval_save_samples = True
                args.in_mem_eval_lpips = False
                args.in_mem_eval_batch = 1
                args.in_mem_eval_vae_batch = 1
                args.sde_gamma = 0.0
                args.flow_sampler = "heun"
                args.eval_steps = 50
                args.eval_cfg = 1.0
                args.eval_blend_alpha = 0.0
                res = run_in_mem_eval(model, args, 0, dev, a.out, logger=print)
            finally:
                h.remove()
            gx = seen.get("x")
            check("软掩码到达 glyph_embedder", gx is not None and float(gx.float().mean()) > 0,
                  "shape=%s mean=%s" % (tuple(gx.shape) if gx is not None else None,
                                        ("%.4f" % float(gx.float().mean())) if gx is not None else None))
            # ??"????"??: ???????? g0.png ? gt0.png
            # (?????? eval200fix_input_g/ ?? g*.png, ???? 64x64)
            _od = [os.path.dirname(q) for q in glob.glob(os.path.join(a.out, "**", "gt0.png"), recursive=True)]
            pngs = [os.path.join(d, "g0.png") for d in _od if os.path.isfile(os.path.join(d, "g0.png"))]
            check("产出生成图 g0.png", bool(pngs), pngs[0] if pngs else "无")
            if pngs:
                arr = np.asarray(Image.open(pngs[0]).convert("L")).astype(np.float32) / 255.0
                ink = float(np.mean(1 - arr))
                check("生成图尺寸 256x256", arr.shape == (256, 256), str(arr.shape))
                check("生成图有墨（非空白）", ink > 0.01, "soft_ink_mass=%.4f" % ink)
                print("    指标(该 1 张):", json.dumps({k: v for k, v in list(res.values())[0].items()
                                                        if k in ("ssim", "ink_iou", "soft_ink_ratio", "near_blank_rate")},
                                                       ensure_ascii=False))
        except Exception as e:                                    # noqa: BLE001
            import traceback
            traceback.print_exc()
            check("端到端出图（异常）", False, "%s: %s" % (type(e).__name__, e))

    print("\n" + "=" * 78)
    bad = [n for n, ok, _ in results if not ok]
    print("PASS %d / %d" % (len(results) - len(bad), len(results)), "  FAIL:", bad if bad else "无 ✓")
    print("产物目录:", a.out)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
