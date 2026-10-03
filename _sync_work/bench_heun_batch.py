"""实测 heun_batch True/False 的墙钟差异 (小 batch 以挤进 GPU 剩余显存)。
预期: batched 多跑 50% 的 forward (6B vs 4B 行)。
"""
import os, sys, time
import torch as th

os.chdir("/root/Workspace/xy/DiT")
sys.path.insert(0, "/root/Workspace/xy/DiT")
from src.model import DiT_2Cond_models
from src.loss.flow_matching import FlowMatching

dev = "cuda"
th.manual_seed(0)
m = DiT_2Cond_models["DiT-2Cond-S/2"](
    num_calligraphers=64, num_characters=7765, condition_fusion="factorized_cat",
    callig_embed_dim=128, char_embed_dim=384, use_char_cond=False,
    use_glyph_cond=True, glyph_scale_init=0.6, glyph_embedder_depth=2,
    glyph_inject_layers=4, glyph_inject_mode="adaln", glyph_vec_cond=True,
    glyph_vec_dim=128, norm_type="rms", mlp_type="swiglu", qk_norm=1, rope=1,
    attn_impl="sdpa", input_size=32, in_channels=4, image_channels=4,
    learn_sigma=False, use_checkpoint=False,
).to(dev).eval()
print(f"params {sum(p.numel() for p in m.parameters())/1e6:.2f}M")

B, C, S = 8, 4, 32
x = th.randn(B, C, S, S, device=dev)
yc = th.randint(0, 64, (B,), device=dev)
yh = th.zeros(B, dtype=th.long, device=dev)
g = th.randn(B, 4, S, S, device=dev)
kw = dict(y_callig=yc, y_char=yh, g=g)


def make_cfg_fn(cfg):
    def fn(xx, tt, **kk):
        return m.forward_with_cfg(xx, tt, cfg_scale=cfg, **kk)
    return fn


STEPS = 20
for hb in (True, False):
    d = FlowMatching(num_steps=STEPS, sampler="heun", heun_batch=hb, shift=1.0)
    # warmup
    with th.no_grad(), th.autocast("cuda", dtype=th.bfloat16):
        d.ddim_sample_loop(make_cfg_fn(0.7), (B, C, S, S), x, clip_denoised=False,
                           model_kwargs=kw, device=dev)
    th.cuda.synchronize()
    t0 = time.time()
    for _ in range(3):
        with th.no_grad(), th.autocast("cuda", dtype=th.bfloat16):
            d.ddim_sample_loop(make_cfg_fn(0.7), (B, C, S, S), x,
                               clip_denoised=False, model_kwargs=kw, device=dev)
    th.cuda.synchronize()
    dt = (time.time() - t0) / 3
    print(f"  heun_batch={str(hb):<5}  {dt*1000:8.1f} ms / {STEPS} 步   "
          f"({dt/STEPS*1000:.2f} ms/步)")
