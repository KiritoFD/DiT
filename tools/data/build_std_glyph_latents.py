# -*- coding: utf-8 -*-
"""
build_std_glyph_latents.py — 字泛化管线: 标准字库 → 渲染 256×256 → VAE encode → latent 字典.

输出 (与 src/utils/glyph_latent.py 的 GlyphLatentLookup 兼容的目录约定):
  std_glyph_latent_v2/
    manifest.json                # "font/U+XXXXX" -> 相对路径 (float16 (4,32,32) npy)
    <font>/U+XXXXX.npy
    coverage_report.json         # 每字体: 渲染数/缺字表
    preview/                     # 渲染 PNG 抽样 (供人工检查)

字体书体映射 (font key -> 文件):
  kai_gb   simkai.ttf   楷体 GB2312      -> script 楷(0)
  kai_st   STKAITI.TTF  华文楷体         -> script 楷(0)
  wei_st   STXINWEI.TTF 华文新魏         -> script 楷(0) 变体
  xing_st  STXINGKA.TTF 华文行楷         -> script 行(3)
  li_gb    SIMLI.TTF    隶变             -> script 隶(4)
  li_st    STLITI.TTF   华文隶书         -> script 隶(4)

渲染归一 (对齐数据集构图): 白底黑字, 墨迹 bbox 等比缩放放入
--box-frac×256 的正方形内, 居中 (数据集实测 ink h≈0.81, w≈0.93, 居中).

用法 (本地 GPU):
  python tools/build_std_glyph_latents.py \
      --charsets tools/charsets/std_8105.txt tools/charsets/midclean_chars_kai.txt \
                 tools/charsets/midclean_chars_xing.txt tools/charsets/midclean_chars_li.txt \
      --vae pretrained_models/sd-vae-ft-ema \
      --out std_glyph_latent_v2
断点续跑: 已存在的 npy 直接跳过 (--overwrite 强制重做).
"""
import os
import sys
import json
import glob
import argparse

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.stdout.reconfigure(encoding="utf-8")

FONTS = {  # key -> (相对 C:/Windows/Fonts 的文件名, 说明)
    "kai_gb": ("simkai.ttf", "楷体 GB2312"),
    "kai_st": ("STKAITI.TTF", "华文楷体"),
    "wei_st": ("STXINWEI.TTF", "华文新魏"),
    "xing_st": ("STXINGKA.TTF", "华文行楷"),
    "li_gb": ("SIMLI.TTF", "隶变"),
    "li_st": ("STLITI.TTF", "华文隶书"),
}
FONT_DIR = r"C:\Windows\Fonts"


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--charsets", nargs="+", required=True,
                    help="字符表 txt (每行一个字); 去重并集为渲染字符集")
    ap.add_argument("--fonts", nargs="*", default=list(FONTS.keys()))
    ap.add_argument("--font-dir", default=FONT_DIR)
    ap.add_argument("--vae", default="pretrained_models/sd-vae-ft-ema")
    ap.add_argument("--out", default="std_glyph_latent_v2")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--box-frac", type=float, default=0.88,
                    help="墨迹 bbox 目标边长占画布比例")
    ap.add_argument("--render-threads", type=int, default=16)
    ap.add_argument("--encode-batch", type=int, default=8,
                    help="VAE encode batch; 8GB 卡建议 8 (encode 是显存尖峰)")
    ap.add_argument("--mem-cap-gb", type=float, default=4.5,
                    help="PyTorch 显存分配上限 (GB); 总显存 = 本值 + 其他应用基线")
    ap.add_argument("--preview-n", type=int, default=24)
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args()


def load_charset(paths):
    chars = []
    seen = set()
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for line in f:
                c = line.strip()
                if c and c not in seen:
                    seen.add(c)
                    chars.append(c)
    return chars


def render_glyph(ch, font_path, size, box_frac):
    """渲染单字 -> (size,size) uint8 白底黑字, 墨迹 bbox 归一化居中。失败返回 None."""
    try:
        font = ImageFont.truetype(font_path, size)
    except Exception:
        return None
    # 先大图渲染再裁 bbox, 避免小字号 hinting 失真
    img = Image.new("L", (size * 2, size * 2), 255)
    draw = ImageDraw.Draw(img)
    try:
        draw.text((size // 2, size // 2), ch, font=font, fill=0, anchor="mm")
    except Exception:
        return None
    a = np.asarray(img)
    ink = a < 250
    if not ink.any():
        return None
    ys, xs = np.where(ink)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    crop = img.crop((x0, y0, x1, y1))
    # 等比缩放到目标 box
    target = box_frac * size
    h, w = crop.size[1], crop.size[0]
    s = target / max(h, w)
    tw, th = max(int(w * s), 1), max(int(h * s), 1)
    crop = crop.resize((tw, th), Image.LANCZOS)
    canvas = Image.new("L", (size, size), 255)
    canvas.paste(crop, ((size - tw) // 2, (size - th) // 2))
    return np.asarray(canvas)


def ukey(ch):
    return f"U+{ord(ch):05X}"


def main():
    args = parse_args()
    chars = load_charset(args.charsets)
    print(f"[charset] {len(chars)} uniq chars "
          f"({os.path.basename(args.charsets[0])} 等 {len(args.charsets)} 个字表)")

    from diffusers.models import AutoencoderKL
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        torch.cuda.set_per_process_memory_fraction(min(args.mem_cap_gb / total_gb, 1.0))
        print(f"[mem] cap {args.mem_cap_gb:.1f}G / {total_gb:.1f}G total")
    vae = AutoencoderKL.from_pretrained(args.vae).to(device).eval()
    print(f"[vae] {args.vae} on {device}")

    report = {}
    for fkey in args.fonts:
        fname, desc = FONTS[fkey]
        fpath = os.path.join(args.font_dir, fname)
        fdir = os.path.join(args.out, fkey)
        os.makedirs(fdir, exist_ok=True)
        os.makedirs(os.path.join(args.out, "preview"), exist_ok=True)

        # ---- 渲染 (跳过已有 npy 的字, 断点续跑) ----
        todo, skipped = [], 0
        for ch in chars:
            outp = os.path.join(fdir, f"{ukey(ch)}.npy")
            if os.path.exists(outp) and not args.overwrite:
                skipped += 1
                continue
            todo.append(ch)
        rendered, missing = [], []
        pngs = []
        for i, ch in enumerate(todo):
            arr = render_glyph(ch, fpath, args.size, args.box_frac)
            if arr is None:
                missing.append(ch)
                continue
            rendered.append((ch, arr))
            if len(pngs) < args.preview_n:
                pngs.append((ch, arr))
            if (i + 1) % 1000 == 0:
                print(f"[{fkey}] rendered {i + 1}/{len(todo)} "
                      f"(missing {len(missing)})", flush=True)

        # ---- GPU encode ----
        n_ok = skipped
        with torch.no_grad():
            for i in range(0, len(rendered), args.encode_batch):
                batch = rendered[i:i + args.encode_batch]
                x = torch.from_numpy(
                    np.stack([a for _, a in batch])).unsqueeze(1).repeat(1, 3, 1, 1)
                x = (x.float() / 255.0 * 2 - 1).to(device)
                lat = vae.encode(x).latent_dist.mode() * 0.18215
                lat = lat.float().cpu().numpy().astype(np.float16)
                for j, (ch, _) in enumerate(batch):
                    np.save(os.path.join(fdir, f"{ukey(ch)}.npy"), lat[j])
                n_ok += len(batch)
                del x, lat
                if device.type == "cuda":
                    torch.cuda.empty_cache()  # batch 间释放, 防碎片累积 OOM
                if (i // args.encode_batch) % 20 == 0:
                    print(f"[{fkey}] encoded {n_ok}/{len(chars)}", flush=True)

        # ---- preview PNG ----
        for ch, arr in pngs:
            Image.fromarray(arr).save(
                os.path.join(args.out, "preview", f"{fkey}_{ukey(ch)}.png"))

        report[fkey] = {
            "desc": desc, "file": fname,
            "ok": n_ok, "rendered_now": len(rendered), "missing": len(missing),
            "missing_chars": "".join(missing[:200]),
        }
        print(f"[{fkey}] DONE ok={n_ok} missing={len(missing)} "
              f"({desc})", flush=True)

    # ---- manifest ----
    manifest = {}
    for fkey in args.fonts:
        for p in sorted(glob.glob(os.path.join(args.out, fkey, "U+*.npy"))):
            rel = os.path.relpath(p, args.out).replace("\\", "/")
            manifest[f"{fkey}/{os.path.splitext(os.path.basename(p))[0]}"] = rel
    with open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=0)
    report["_total_manifest"] = len(manifest)
    with open(os.path.join(args.out, "coverage_report.json"), "w",
              encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[done] manifest {len(manifest)} latents -> {args.out}")


if __name__ == "__main__":
    main()
