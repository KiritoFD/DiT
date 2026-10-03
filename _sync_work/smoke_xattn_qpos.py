"""冒烟: xattn 的 q_pos 开关是否真的生效 + model_io 能否正确重建。

⚠⚠ 关键前提: 注入模块的 `out_proj` 是 **zero-init** -> 初始输出恒为 0。
   不打破它，`q_pos` 改什么都不影响模型输出 —— 会得到一个**假阴性**。
   这正是 docs/system/70 §C.7 记录的坑（"测试脚本本身要先做 sanity check"）。
   所以下面用 `unzero()` 把 out_proj 随机化，让注入真的起作用。
"""
import os
import sys
from argparse import Namespace

import torch

os.chdir("/root/Workspace/xy/DiT")
sys.path.insert(0, "/root/Workspace/xy/DiT")
from src.model import DiT_2Cond_models  # noqa: E402
from src.eval.model_io import build_model_from_args  # noqa: E402

COMMON = dict(
    num_calligraphers=64, num_characters=7765, condition_fusion="factorized_cat",
    callig_embed_dim=128, char_embed_dim=384, use_char_cond=False,
    use_glyph_cond=True, glyph_scale_init=0.6, glyph_embedder_depth=2,
    glyph_inject_layers=4, glyph_inject_mode="xattn", glyph_vec_cond=True,
    glyph_vec_dim=128, norm_type="rms", mlp_type="swiglu", qk_norm=1, rope=1,
    attn_impl="sdpa", input_size=32, in_channels=4, image_channels=4,
    learn_sigma=False,
)


def unzero(m):
    """打破**所有** zero-init，否则开关效果被吞掉。

    ⚠ 有**两处** zero-init，必须都打破:
      ① 注入模块的 out_proj      -> 不打破则注入输出恒为 0
      ② final_layer.linear       -> 不打破则**最终输出恒为 0**，前面所有差异全被吞掉
    只打破 ① 会得到一个假阴性（本文件第一版就是这么错的，与 doc70 §C.7 同类）。
    """
    with torch.no_grad():
        for inj in (getattr(m, "glyph_injections", None) or []):
            inj.out_proj.weight.normal_(0, 0.02)
            inj.out_proj.bias.normal_(0, 0.02)
        fl = m.final_layer.linear
        fl.weight.normal_(0, 0.02)
        fl.bias.normal_(0, 0.02)


outs = {}
for qp in (False, True):
    torch.manual_seed(1234)                     # 两个模型用同一份随机权重
    m = DiT_2Cond_models["DiT-2Cond-S/2"](xattn_q_pos=qp, **COMMON).eval()
    unzero(m)
    print(f"  xattn q_pos={qp}: params={sum(p.numel() for p in m.parameters()):,} "
          f"  inj[0].q_pos={m.glyph_injections[0].q_pos}")
    x = torch.randn(2, 4, 32, 32)
    t = torch.rand(2)
    yc = torch.randint(0, 64, (2,))
    yh = torch.zeros(2, dtype=torch.long)
    g = torch.randn(2, 4, 32, 32)
    with torch.no_grad():
        outs[qp] = m(x, t, yc, yh, g=g)

d = (outs[False] - outs[True]).abs().max().item()
print(f"  q_pos 是否改变行为: max|diff| = {d:.3e}  -> "
      + ("是（开关生效）" if d > 1e-6 else "**否 —— 开关没生效!**"))

# model_io 能否正确重建（含 xattn_q_pos）
a = Namespace(model="DiT-2Cond-S/2", **COMMON)
a.vae_downscale = 8
a.num_calligraphers = 64
a.num_characters = 7765
a.cond_drop_all_prob = 0.1
a.cond_drop_one_prob = 0.0
a.no_char_cond = True
a.w_glyph_cond = 1
a.xattn_q_pos = True
m2 = build_model_from_args(a, "cpu")
ok = bool(m2.xattn_q_pos) and bool(m2.glyph_injections[0].q_pos)
print(f"  model_io 重建后 xattn_q_pos={m2.xattn_q_pos} "
      f"inj[0].q_pos={m2.glyph_injections[0].q_pos} -> {'OK' if ok else '**漏传!**'}")
