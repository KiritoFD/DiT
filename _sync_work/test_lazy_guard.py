"""验证 materialize_lazy_params 护栏：不调 freeze_table 也不会丢 null_embed。"""
import glob
import os
import sys

import torch

os.chdir("/root/Workspace/xy/DiT")
sys.path.insert(0, "/root/Workspace/xy/DiT")
from src.model import DiT_2Cond_models  # noqa: E402
from src.utils.channel_expand import materialize_lazy_params  # noqa: E402

ck = torch.load(sorted(glob.glob("assets/results/v13_base_50k/*/checkpoints/*.pt"))[-1],
                map_location="cpu", weights_only=False)
sd = {(k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
      for k, v in ck["delta"].items()}

C = dict(
    num_calligraphers=45, num_characters=7765, condition_fusion="factorized_cat",
    callig_embed_dim=128, char_embed_dim=384, use_char_cond=False,
    use_glyph_cond=True, glyph_scale_init=0.6, glyph_embedder_depth=2,
    glyph_inject_layers=4, glyph_inject_mode="adaln", glyph_vec_cond=True,
    glyph_vec_dim=128, norm_type="rms", mlp_type="swiglu", qk_norm=1, rope=1,
    attn_impl="sdpa", input_size=32, in_channels=4, image_channels=4,
    learn_sigma=False,
)

torch.manual_seed(0)
m = DiT_2Cond_models["DiT-2Cond-S/2"](**C)
print("  freeze_table 之前模型有 null_embed? "
      f"{'y_callig_embedder.null_embed' in m.state_dict()}")

added = materialize_lazy_params(m, sd)
print(f"  materialize 补出: {added}")
print("  补出后模型有 null_embed? "
      f"{'y_callig_embedder.null_embed' in m.state_dict()}")

miss, unexp = m.load_state_dict(sd, strict=False)
print(f"  load: missing={len(miss)}  unexpected={len(unexp)}")
if unexp:
    print(f"    unexpected: {unexp[:5]}")

same = bool(torch.equal(m.y_callig_embedder.null_embed.detach(),
                        sd["y_callig_embedder.null_embed"]))
print(f"  null_embed == ckpt 的值? **{same}**")
print("  -> " + ("**护栏生效：不调 freeze_table 也不会丢** ✓" if same
                 else "**仍有问题** ✗"))
