# -*- coding: utf-8 -*-
"""cpu_sampler.py — 为 CPU 执行性质专门写的 Heun+CFG 采样器.

与 GPU 路径 (FlowMatching.ddim_sample_loop, sampler=heun) **数学严格等价**,
差异全部是 CPU 执行性质驱动的性能选择:

  1. fp32 无 autocast —— Zen2/EPYC 无 AVX512-BF16/AMX, bf16 走软件上转反而慢.
  2. Heun 两 stage 分开评估 —— FlowMatching.heun_batch=True 每步处理 6B 行
     (v1: 2B + v_cat: 4B), 多出的 50% 行数只为摊 GPU kernel launch; CPU 是
     GEMM compute-bound, 纯亏. 逐 stage 每步 4B 行.
  3. CFG 沿 batch 拼批 —— forward_with_cfg 内部 cat 成 2B 行, GEMM M 大,
     每层 ~30MB 权重 tile 的带宽开销按行摊薄 (B 越大越接近 compute-bound).
  4. 预分配输出缓冲 / t 网格一次生成 / 无 empty_cache / 无 host<->device 往返.

时间网格与 shift 映射逐行复刻 FlowMatching._schedule:
    s = linspace(1, 0, steps+1); t = shift*s / (1+(shift-1)*s)  (shift=1 -> 均匀)
"""
import torch as th

TIME_SCALE = 1000.0


@th.no_grad()
def heun_sample_cpu(model, noise, conds, cfg_scale, batch, skel=None, seed=0,
                    steps=50, shift=1.0, log_fn=None, cond_key="cond"):
    """CPU Heun 采样 (每步 2 次速度评估, CFG 在 forward_with_cfg 内拼批).

    model  : ControlNetDiT / 带 forward_with_cfg 的包装模型 (eval 模式, CPU)
    noise  : (N,C,H,W) 固定噪声 (CPU)
    conds  : [(callig_id, glyph_id)] * N
    skel   : (N,4,32,32) 骨架 latent (None = base 臂)
    返回   : (N,C,H,W) fp32 CPU
    """
    n = noise.shape[0]
    lc, ls = noise.shape[1], noise.shape[2]
    out = th.zeros(n, lc, ls, ls, dtype=th.float32)
    th.manual_seed(seed)  # 采样无随机分量 (noise 固定), 保持与 sample_latents 签名一致

    s = th.linspace(1.0, 0.0, steps + 1, dtype=th.float64)
    ts = (shift * s / (1.0 + (shift - 1.0) * s)).tolist() if shift != 1.0 else s.tolist()
    use_cfg = bool(cfg_scale) and cfg_scale > 0

    for i0 in range(0, n, batch):
        i1 = min(i0 + batch, n)
        b = i1 - i0
        x = noise[i0:i1].clone()
        mk = dict(
            y_callig=th.tensor([c[0] for c in conds[i0:i1]], dtype=th.long),
            y_char=th.tensor([c[1] for c in conds[i0:i1]], dtype=th.long))
        if skel is not None:
            mk[cond_key] = skel[i0:i1].clone()

        for k in range(steps):
            t_i, t_next = ts[k], ts[k + 1]
            dt = t_next - t_i
            t1 = th.full((b,), t_i)
            v1 = _velocity(model, x, t1, mk, cfg_scale, use_cfg)
            x_e = x + dt * v1
            t2 = th.full((b,), t_next)
            v2 = _velocity(model, x_e, t2, mk, cfg_scale, use_cfg)
            x = x + dt * 0.5 * (v1 + v2)

        out[i0:i1] = x
        if log_fn is not None:
            log_fn(f"    chunk [{i0}:{i1}] done")
    return out


def _velocity(model, x, t, mk, cfg_scale, use_cfg):
    """一次速度场评估 (含 CFG 拼批 / learn_sigma 通道裁剪), 与 FlowMatching._v 一致."""
    if use_cfg:
        v = model.forward_with_cfg(x, t * TIME_SCALE, cfg_scale=cfg_scale, **mk)
    else:
        v = model(x, t * TIME_SCALE, **mk)
    if isinstance(v, tuple):
        v = v[0]
    return v
