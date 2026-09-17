"""验证 null_embed 在**真实加载顺序**下不会丢。

真实顺序（train.py）:
  建模型 -> freeze_table()（创建 null_embed）-> load_state_dict(ckpt)
所以加载时 null_embed 已存在，ckpt 里的值能正确灌进去。

本脚本对比两种顺序，证明"顺序对了就没问题、顺序错了会静默丢"。
"""
import glob
import os
import sys

import torch

os.chdir("/root/Workspace/xy/DiT")
sys.path.insert(0, "/root/Workspace/xy/DiT")
from src.model import DiT_2Cond_models  # noqa: E402

ck_path = sorted(glob.glob("assets/results/v13_base_50k/*/checkpoints/*.pt"))[-1]
ck = torch.load(ck_path, map_location="cpu", weights_only=False)
sd = {(k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
      for k, v in ck["delta"].items()}
print(f"  ckpt: {os.path.basename(ck_path)}")
print(f"  ckpt 里有 null_embed? {'y_callig_embedder.null_embed' in sd}")
if "y_callig_embedder.null_embed" in sd:
    ref = sd["y_callig_embedder.null_embed"]
    print(f"    形状 {tuple(ref.shape)}  abs.mean={float(ref.abs().mean()):.6f}")

C = dict(
    num_calligraphers=45, num_characters=7765, condition_fusion="factorized_cat",
    callig_embed_dim=128, char_embed_dim=384, use_char_cond=False,
    use_glyph_cond=True, glyph_scale_init=0.6, glyph_embedder_depth=2,
    glyph_inject_layers=4, glyph_inject_mode="adaln", glyph_vec_cond=True,
    glyph_vec_dim=128, norm_type="rms", mlp_type="swiglu", qk_norm=1, rope=1,
    attn_impl="sdpa", input_size=32, in_channels=4, image_channels=4,
    learn_sigma=False,
)


def build_and_maybe_freeze(freeze):
    torch.manual_seed(0)
    m = DiT_2Cond_models["DiT-2Cond-S/2"](**C)
    if freeze:
        m.y_callig_embedder.freeze_table()
    return m


print()
print("  === 顺序 A: freeze_table() -> load（真实顺序）===")
m = build_and_maybe_freeze(True)
has = "y_callig_embedder.null_embed" in m.state_dict()
print(f"    加载前模型里有 null_embed? {has}")
miss, unexp = m.load_state_dict(sd, strict=False)
print(f"    missing={len(miss)} unexpected={len(unexp)}")
got = m.y_callig_embedder.null_embed
same = bool(torch.equal(got.detach(), sd["y_callig_embedder.null_embed"]))
print(f"    加载后 null_embed == ckpt 的值? **{same}**")
print(f"    requires_grad={got.requires_grad} (应为 True)")

print()
print("  === 顺序 B: load -> freeze_table()（错误顺序，会覆盖成随机）===")
m2 = build_and_maybe_freeze(False)
miss2, unexp2 = m2.load_state_dict(sd, strict=False)
print(f"    missing={len(miss2)} unexpected={len(unexp2)}")
print(f"    ckpt 里的 null_embed 被丢了? "
      f"{'y_callig_embedder.null_embed' in unexp2}")
m2.y_callig_embedder.freeze_table()
got2 = m2.y_callig_embedder.null_embed
same2 = bool(torch.equal(got2.detach(), sd["y_callig_embedder.null_embed"]))
print(f"    之后再 freeze_table() -> == ckpt 的值? **{same2}**  (False = 被随机覆盖)")

print()
print("  -> " + ("**真实顺序正确，null_embed 不会丢** ✓" if same
                 else "**真丢了，需修** ✗"))
