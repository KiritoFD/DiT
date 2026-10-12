#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""probe_glyph_encoder.py — 条件字形编码器的偏置/内容体检（参数化，任意机器/任意 run 可用）

为什么需要它（v81 白化事故的机制）:
  std_soft 的 CNN 编码器前两层 conv 带 bias ⇒ 对**全零输入**有非零响应,
  该响应随训练从 19.9 涨到 72.1（3.6×）, 内容/背景比从 2.48 掉到 1.13,
  最终"空白画布特征"压过真实字形结构 ⇒ decode 出白纸。
修复: 注入量改为 **enc(g) - enc(0)**（不 detach）⇒ 偏置被差分抵消。

本工具同时报三组量, 用于**判断修复是否生效**:
  A. ||enc(0)||            —— 裸偏置响应大小（会随训练涨, 本身不是 bug 只要它被差分掉）
  B. ||enc(g)-enc(0)||     —— **真正被注入的内容信号**（修复后应平稳/上升, 不应衰减到 0）
  C. A/B 比值（背景/内容）—— v81 从 0.40 升到 0.88（内容被淹没）; 修复后应保持低位平稳 ✓

用法:
  python tools/analysis/probe_glyph_encoder.py \
      --ckpt-dir checkpoints --config src/train/configs/v83A_stdsoft_zerofix.json \
      --store exp-std/data/std_soft_glyphs_v1 --device cpu --steps 20000,25000,30000
"""
import argparse
import glob
import json
import os
import sys

import torch

sys.path.insert(0, ".")
if hasattr(sys.stdout, "reconfigure"):          # Python 3.7+；3.6 上没有
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt-dir", required=True, help="含 00NNNNN.pt 的目录")
ap.add_argument("--config", required=True, help="训练/评测用的 config json")
ap.add_argument("--store", default="", help="std_soft 字形 store 目录（含 index.csv）")
ap.add_argument("--device", default="cpu", help="cpu / cuda:0（GPU 被训练占满时用 cpu）")
ap.add_argument("--steps", default="", help="只测这些 step, 逗号分隔, 如 20000,25000; 空=全部")
ap.add_argument("--n", type=int, default=8, help="取多少个字形求平均")
ap.add_argument("--ckpt-glob", default="*.pt")
a = ap.parse_args()
dev = torch.device(a.device)

from src.eval import model_io  # noqa: E402

cfg = json.load(open(a.config, encoding="utf-8"))


class _A:
    pass


args = _A()
for k, v in cfg.items():
    setattr(args, k, v)

# ── 取真实字形掩码 ──
store_dir = a.store or getattr(args, "std_soft_store", "") or getattr(args, "std_soft_dir", "")
g_real = None
if store_dir and os.path.isfile(os.path.join(store_dir, "index.csv")):
    import csv
    from src.utils.std_soft_glyph import StdSoftGlyphStore
    rows = list(csv.DictReader(open(os.path.join(store_dir, "index.csv"), encoding="utf-8-sig")))
    st = StdSoftGlyphStore(store_dir, is_train=False, preload=False)
    tens, used = [], []
    for r in rows:
        if len(tens) >= a.n:
            break
        try:
            t = st.get(int(r.get("script_id") or r.get("script") or 0), r["character"])
        except Exception:
            continue
        if t is None:
            continue
        t = t if t.dim() == 3 else t.unsqueeze(0)      # (1,64,64)
        tens.append(t)
        used.append(r["character"])
    if tens:
        g_real = torch.stack(tens).float().to(dev)      # (N,1,64,64)
        # 用 % 格式化: 避免老解释器/嵌套引号的 f-string 兼容问题
        _ex = "".join(used[:8])
        print("  真字形 %s 墨量均值=%.4f 例=%s" % (tuple(g_real.shape), float(g_real.mean()), _ex))
    else:
        print("  取真字形失败: store 里没有可用条目")
else:
    print(f"  未找到 store({store_dir or '未指定'}), 只测 ||enc(0)||")

steps = [s.strip() for s in a.steps.split(",") if s.strip()]
files = sorted(glob.glob(os.path.join(a.ckpt_dir, a.ckpt_glob)))
if steps:
    files = [f for f in files if any(s in os.path.basename(f) for s in steps)]
print(f"  待测 ckpt: {len(files)} 个  device={a.device}\n")

hdr = ("  %8s | %10s | %17s | %9s | %10s | %10s"
       % ("step", "||enc0||/N", "||encg-enc0||/N", "背景/内容", "||encg||/N", "rms(enc0)"))
print(hdr)
print("  " + "-" * (len(hdr) + 2))
for f in files:
    tag = os.path.basename(f).replace(".pt", "")
    try:
        # ★ 实测(2026-10-12): 该 ckpt 的 args 里没有 std_soft 标志 -> load_model_from_ckpt
        #   按 ckpt args 自建的模型**缺字形编码器** -> strict=True 直接 Unexpected key ✗。
        #   故改为: 按**传入 config** 建模型 + strict=False 加载, 并打印 glyph 相关缺/多键**自证**。
        m = model_io.build_model_from_args(args, dev)
        _ck = torch.load(f, map_location="cpu", weights_only=False)
        _sd = _ck.get("ema") if _ck.get("ema") is not None else _ck.get("model", _ck)
        _sd = {k.replace("_orig_mod.", ""): v for k, v in _sd.items()}
        _miss, _unexp = m.load_state_dict(_sd, strict=False)
        _gm = [k for k in _miss if "glyph" in k.lower()]
        _gu = [k for k in _unexp if "glyph" in k.lower()]
        print("      load: missing=%d unexpected=%d | glyph: missing=%d unexpected=%d"
              % (len(_miss), len(_unexp), len(_gm), len(_gu)))
        del _ck, _sd
        m.to(dev).eval()
    except Exception as e:
        print(f"  {tag:>8} | 加载失败 {type(e).__name__}: {str(e)[:60]}")
        continue
    emb = next((mod for n, mod in m.named_modules()
                if "embed" in n.lower() and any(t in n.lower() for t in ("glyph", "soft", "skel"))), None)
    if emb is None:
        print(f"  {tag:>8} | 未找到 glyph embedder")
        continue
    try:
        with torch.no_grad():
            # ★ 口径必须一致（2026-10-12 修 bug）: 旧版拿 **1 个全零样本** 的范数去除
            #   **4 个真实样本** 的范数 -> 比值系统性偏离 2×(√4), 且与 v81 旧脚本(两边都是
            #   4 样本)口径不同 -> 跨版本对比无效 ✗。现在: f0 与 fr 用**同一 batch 大小**,
            #   同时给 per-sample 范数与逐元素 RMS, 便于与别的会话/工具对齐 ✓
            _nb = max(1, min(4, int(getattr(g_real, "shape", [1])[0]))) if g_real is not None else 1
            f0 = emb(torch.zeros(_nb, 1, 64, 64, device=dev)).float()
            n0_tot = float(f0.norm())
            n0_ps = n0_tot / (_nb ** 0.5)                       # per-sample 范数
            n0_rms = n0_tot / (f0.numel() ** 0.5)               # 逐元素 RMS（跨工具对齐用）
            if g_real is not None:
                fr = emb(g_real[:_nb]).float()
                nc_ps = float((fr - f0).norm()) / (_nb ** 0.5)
                ng_ps = float(fr.norm()) / (_nb ** 0.5)
                ratio = n0_ps / max(nc_ps, 1e-9)                # ★ 同口径 per-sample/per-sample
                print("  %s | %10.4f | %17.4f | %9.4f | %10.4f | rms0=%.4f"
                      % (tag, n0_ps, nc_ps, ratio, ng_ps, n0_rms))
            else:
                print("  %s | %10.4f | %17s | %9s | %10s | rms0=%.4f"
                      % (tag, n0_ps, "-", "-", "-", n0_rms))
    except Exception as e:
        print("  %s | 前向失败 %s: %s" % (tag, type(e).__name__, str(e)[:60]))

print("""
  判读:
   * **背景/内容** 随训练**单调上升** ⇒ 内容被偏置淹没（v81 事故特征: 0.40 → 0.88）✗
   * 修复(注入 enc(g)-enc(0))生效时 ⇒ 该比值应**低位平稳** ✓（即使 ||enc(0)|| 仍在涨也无害, 因为被差分掉）
   * 若 内容项 ||enc(g)-enc(0)|| 本身**衰减到接近 0** ⇒ 编码器在丢弃字形结构, 属另一种失效 ✗
""")
