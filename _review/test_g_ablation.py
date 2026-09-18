"""决定性实验：模型到底有没有在用 g（标准字形）？

做法：同一批样本跑三次采样
  A) 正常 g
  B) g 换成**别的样本的 g**（打乱）
  C) g 全零
比较 A 与 B/C 的输出差异：
  - 差异大 -> 模型确实在用 g（内容条件有效），瓶颈在别处
  - 差异小 -> **模型忽略 g**，瓶颈就是条件注入机制（adaln 不够）
"""
import glob
import os
import sys

import torch

os.chdir("/root/Workspace/xy/DiT")
sys.path.insert(0, "/root/Workspace/xy/DiT")
from src.train.cli import parse_args  # noqa: E402
from src.eval.inference import make_eval_cache  # noqa: E402
from src.utils.callig_map import load_callig_id_map  # noqa: E402

sys.argv = ["x", "--config", "src/train/configs/v13_12ch_post.json"]
args = parse_args()
CMAP, _ = load_callig_id_map("assets/callig_id_map_50k.json")

cache = make_eval_cache("assets/eval_v13_strict.csv", None, None, 256, 8, 8, 4,
                        0.18215, skel_latent_shards_dir="data/50k/shards_std",
                        callig_id_map=CMAP)
print(f"  cache: noise {tuple(cache['noise'].shape)} "
      f"g {tuple(cache['skels_latent'].shape)}")

# 只建 4ch 的 base 模型（要评的 ckpt 是 base 155k）
import importlib
mod = importlib.import_module("src.train.train")
ck = torch.load(sorted(glob.glob(
    "assets/results/v13_base_50k/*/checkpoints/0155000.pt"))[-1],
    map_location="cpu", weights_only=False)
sd = ck.get("ema") or ck.get("delta")
sd = {(k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
      for k, v in sd.items()}
print(f"  权重 {len(sd)} 张量")

# 用 train.py 的建模函数太重；这里直接用 args 建
from src.model import DiT_2Cond_models  # noqa: E402
model = DiT_2Cond_models[args.model](
    input_size=32, num_calligraphers=args.num_calligraphers,
    num_characters=args.num_characters, use_checkpoint=False,
    learn_sigma=False, condition_fusion=args.condition_fusion,
    callig_embed_dim=args.callig_embed_dim, char_embed_dim=args.char_embed_dim,
    glyph_vec_cond=getattr(args, "glyph_vec_cond", False),
    glyph_vec_dim=int(getattr(args, "glyph_vec_dim", 128)),
    glyph_vec_pool=getattr(args, "glyph_vec_pool", "mean"),
    cond_drop_all_prob=0.0, cond_drop_one_prob=0.0,
    use_glyph_cond=True, glyph_scale_init=0.6,
    glyph_embedder_depth=2, glyph_inject_layers=4,
    glyph_inject_mode="adaln", norm_type="rms", mlp_type="swiglu",
    qk_norm=1, rope=1, attn_impl="sdpa",
    num_heads=getattr(args, "num_heads", 6),
    in_channels=4, image_channels=4,
)
miss, unexp = model.load_state_dict(sd, strict=False)
print(f"  load: missing={len(miss)} unexpected={len(unexp)}")
model = model.cuda().eval()

from src.loss import create_diffusion_or_flow, flow_kwargs_from  # noqa: E402
diff = create_diffusion_or_flow(**flow_kwargs_from(args))

from src.eval.inference import sample_latents  # noqa: E402
dev = torch.device("cuda")
nz, conds, g = cache["noise"], cache["conds"], cache["skels_latent"]

with torch.no_grad():
    A = sample_latents(model, diff, nz, conds, 0.7, 4, dev, skel=g, seed=0)
    g_perm = g.flip(0)
    B = sample_latents(model, diff, nz, conds, 0.7, 4, dev, skel=g_perm, seed=0)
    C = sample_latents(model, diff, nz, conds, 0.7, 4, dev,
                       skel=torch.zeros_like(g), seed=0)

dAB = (A - B).abs().mean().item()
dAC = (A - C).abs().mean().item()
scale = A.abs().mean().item()
print()
print(f"  输出尺度 (|A| 均值)      = {scale:.4f}")
print(f"  A vs B（g 打乱）差异     = {dAB:.4f}   ({dAB/scale*100:.1f}% of scale)")
print(f"  A vs C（g 归零）差异     = {dAC:.4f}   ({dAC/scale*100:.1f}% of scale)")
print()
print("  -> " + ("**模型在用 g** ✓ 瓶颈在别处" if dAB / scale > 0.15
                 else "**模型几乎忽略 g** ✗ 瓶颈 = 字形注入机制"))
