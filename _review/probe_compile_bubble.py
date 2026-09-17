"""独立探针: torch.compile 是否/多大程度上制造显存空洞。

## 为什么不用 train.py 起探针
上次用 `train.py --global-batch-size 16` 复现启动流程，**卡死在 preload** ——
REPA 的 DINO 缓存是 `mode=pinned`（29.6GiB 独占的 page-locked 主机内存），
训练已占一份，第二份申请会阻塞。

## 本探针
只复现 **建模型 → compile → 前向/反向/opt.step**，用随机张量，
**不加载数据集、不加载 REPA 缓存、不加载 VAE**。
显存占用 ~1-2G，可以安全地跑在正在训练的任务旁边。

## 要回答的问题
1. compile 到底制不制造空洞？
2. 空洞是**批无关的固定量**（某块固定 buffer）还是**随 batch 线性放大**？
3. `empty_cache()` 能回收多少？

用法: PYTHONPATH=<repo> python probe_compile_bubble.py [batch ...]
"""
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
from src.model import DiT_2Cond_models  # noqa: E402

COMMON = dict(
    num_calligraphers=52, num_characters=7765, condition_fusion="factorized_cat",
    callig_embed_dim=128, char_embed_dim=384, use_char_cond=False,
    use_glyph_cond=True, glyph_scale_init=0.6, glyph_embedder_depth=2,
    glyph_inject_layers=4, glyph_inject_mode="xattn", glyph_vec_cond=True,
    glyph_vec_dim=128, norm_type="rms", mlp_type="swiglu", qk_norm=1, rope=1,
    attn_impl="sdpa", input_size=32, in_channels=4, image_channels=4,
    learn_sigma=False,
)


def rss(tag):
    torch.cuda.synchronize()
    r = torch.cuda.memory_reserved() / 2 ** 30
    a = torch.cuda.memory_allocated() / 2 ** 30
    p = torch.cuda.max_memory_reserved() / 2 ** 30
    print(f"  {tag:<32} reserved={r:7.3f}G  allocated={a:7.3f}G  "
          f"空洞={r - a:7.3f}G  peak={p:7.3f}G", flush=True)
    return r, a, p


def run(B, use_compile=True):
    print(f"\n===== batch={B}  compile={use_compile} =====", flush=True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(0)
    rss("0 起点")

    m = DiT_2Cond_models["DiT-2Cond-S/2"](**COMMON)
    rss("1 建模型 (cpu)")
    m = m.cuda().train()
    rss("2 .cuda()")
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-4)
    rss("3 optimizer")

    if use_compile:
        m = torch.compile(m, mode="default")
        rss("4 compile 注入 (尚未执行)")

    x = torch.randn(B, 4, 32, 32, device="cuda")
    t = torch.rand(B, device="cuda")
    yc = torch.randint(0, 52, (B,), device="cuda")
    yh = torch.zeros(B, dtype=torch.long, device="cuda")
    g = torch.randn(B, 4, 32, 32, device="cuda")
    rss("5 输入就绪")

    out = m(x, t, yc, yh, g=g)
    rss("6 首次 forward" + (" (含编译)" if use_compile else ""))
    out.float().pow(2).mean().backward()
    rss("7 首次 backward")
    opt.step()
    opt.zero_grad(set_to_none=True)
    rss("8 首次 opt.step")

    for _ in range(5):
        out = m(x, t, yc, yh, g=g)
        out.float().pow(2).mean().backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
    r0, a0, _ = rss("9 稳态 5 步后")

    torch.cuda.empty_cache()
    rss("10 empty_cache 后")

    del m, opt, x, t, yc, yh, g, out
    torch.cuda.empty_cache()
    rss("11 释放后")
    return r0 - a0


if __name__ == "__main__":
    batches = [int(a) for a in sys.argv[1:]] or [16, 64, 128]
    print(f"  GPU: {torch.cuda.get_device_name(0)}  "
          f"总显存 {torch.cuda.get_device_properties(0).total_memory / 2 ** 30:.1f}G")
    for B in batches:
        run(B, use_compile=True)
        run(B, use_compile=False)
