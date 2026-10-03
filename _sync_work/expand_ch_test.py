"""验证 4ch -> 12ch 的通道扩展是否**无损**。

扩展只涉及 3 个张量：
  ① x_embedder.proj.weight      [h, 4, p, p]  -> [h, 12, p, p]   拷前 4 个输入通道
  ② final_layer.linear.weight   [(HWpp)*4, h] -> [(HWpp)*12, h]  **按 patch 交错**拷贝
  ③ final_layer.linear.bias     [(HWpp)*4]    -> [(HWpp)*12]     **按 patch 交错**拷贝
（x_embedder.proj.bias 形状 [h] 不变）

⚠ 关键：final_layer 的输出布局是 `reshape(N,h,w,p,p,C)` —— **通道是最内层**，
所以通道不是"前 4 行"，而是每个 patch 位置内连续的 4 个。必须 reshape 成 (HWpp, C, ...) 再拷。

新通道全部**零初始化** -> 第 0 步 12ch 模型在前 4 通道上的输出应与 4ch 模型**完全相同**。
"""
import os
import sys

import torch

os.chdir("/root/Workspace/xy/DiT")
sys.path.insert(0, "/root/Workspace/xy/DiT")
from src.model import DiT_2Cond_models  # noqa: E402

COMMON = dict(
    num_calligraphers=64, num_characters=7765, condition_fusion="factorized_cat",
    callig_embed_dim=128, char_embed_dim=384, use_char_cond=False,
    use_glyph_cond=True, glyph_scale_init=0.6, glyph_embedder_depth=2,
    glyph_inject_layers=4, glyph_inject_mode="adaln", glyph_vec_cond=True,
    glyph_vec_dim=128, norm_type="rms", mlp_type="swiglu", qk_norm=1, rope=1,
    attn_impl="sdpa", input_size=32, learn_sigma=False,
)


def build(in_ch, img_ch):
    return DiT_2Cond_models["DiT-2Cond-S/2"](
        in_channels=in_ch, image_channels=img_ch, **COMMON)


def expand_4ch_to_12ch(m4, m12, n_aux=2):
    """把 4ch 模型的权重灌进 12ch 模型（新通道零初始化）。"""
    old_in = m4.in_channels          # 4
    new_in = m12.in_channels         # 12
    assert new_in == old_in + 4 * n_aux

    sd4 = m4.state_dict()
    sd12 = m12.state_dict()
    done = []
    with torch.no_grad():
        for k, v_new in sd12.items():
            v_old = sd4[k]
            if tuple(v_old.shape) == tuple(v_new.shape):
                v_new.copy_(v_old)                      # 其余全部原样搬
                continue

            if k.endswith("x_embedder.proj.weight"):
                # [h, C, p, p]：C 在 dim=1，直接拷前 old_in 个输入通道
                v_new[:, :old_in].copy_(v_old)
                done.append(f"{k} {tuple(v_old.shape)}->{tuple(v_new.shape)} (输入通道)")

            elif k.endswith("final_layer.linear.weight"):
                # [(HWpp)*C, h]：reshape 成 (HWpp, C, h)，C 在中间
                HWpp = v_old.shape[0] // old_in
                v_new.view(HWpp, new_in, -1)[:, :old_in].copy_(
                    v_old.view(HWpp, old_in, -1))
                done.append(f"{k} {tuple(v_old.shape)}->{tuple(v_new.shape)} (输出通道, 按patch交错)")

            elif k.endswith("final_layer.linear.bias"):
                # [(HWpp)*C]：同理
                HWpp = v_old.shape[0] // old_in
                v_new.view(HWpp, new_in)[:, :old_in].copy_(
                    v_old.view(HWpp, old_in))
                done.append(f"{k} {tuple(v_old.shape)}->{tuple(v_new.shape)} (输出bias, 按patch交错)")

            else:
                raise RuntimeError(f"未预期的形状差异: {k} {tuple(v_old.shape)} -> {tuple(v_new.shape)}")
    return done


if __name__ == "__main__":
    torch.manual_seed(0)
    m4 = build(4, 4).eval()
    m12 = build(12, 4).eval()
    print("  形状差异处理:")
    for d in expand_4ch_to_12ch(m4, m12, n_aux=2):
        print("   ", d)

    B = 3
    x4 = torch.randn(B, 4, 32, 32)
    aux = torch.randn(B, 8, 32, 32)
    x12 = torch.cat([x4, aux], dim=1)          # 前 4 通道与 4ch 的输入相同
    t = torch.rand(B)
    yc = torch.randint(0, 64, (B,))
    yh = torch.zeros(B, dtype=torch.long)
    g = torch.randn(B, 4, 32, 32)

    with torch.no_grad():
        o4 = m4(x4, t, yc, yh, g=g)
        o12 = m12(x12, t, yc, yh, g=g)

    d = (o4 - o12[:, :4]).abs().max().item()
    print()
    print(f"  4ch 输出 vs 12ch 输出前 4 通道: max|diff| = {d:.3e}")
    print(f"  新通道输出是否为零: {float(o12[:, 4:].abs().max()):.3e}")
    print()
    print("  **无损扩展成立**" if d < 1e-5 else "  **不成立（有差异）**")
