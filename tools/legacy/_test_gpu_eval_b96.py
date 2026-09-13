"""Standalone test: build model from latest ckpt, run GPU eval with batch=96, check VRAM."""
import os, sys, time, json, glob
os.environ["XFORMERS_DISABLED"] = "1"
import torch
import numpy as np

sys.path.insert(0, "/root/Workspace/xy/DiT")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.chdir("/root/Workspace/xy/DiT")

from models import DiT_2Cond_models
from in_process_eval import prepare_eval_cache, run_gpu_eval, load_eval_vae

# DINO injection
def _inject_dino(model, args):
    emb_path = getattr(args, "char_dino_embeddings", None)
    idx_path = getattr(args, "char_dino_index", None)
    if not emb_path or not idx_path:
        return
    emb = np.load(emb_path)
    with open(idx_path, encoding="utf-8") as f:
        idx_data = json.load(f)
    glyphs = idx_data.get("glyphs", idx_data)
    table = model.y_char_embedder.embedding_table.weight
    NUM_CH = 7026
    injected = 0
    with torch.no_grad():
        for gi, (sid, cid) in enumerate(glyphs):
            gid = int(sid) * NUM_CH + int(cid)
            if 0 <= gid < table.shape[0] and gi < emb.shape[0]:
                e = emb[gi]
                e = e / (np.linalg.norm(e) + 1e-8)
                table.data[gid] = torch.from_numpy(e).float()
                injected += 1
    print(f"[dino-init] injected {injected} glyph embeddings")

# Find latest ckpt
marker = "5script/results/s10_b4_grey_clear/_active_ckpt_dir.txt"
with open(marker) as f:
    ckpt_dir = f.read().strip()
ckpt_dir = os.path.join("/root/Workspace/xy/DiT", ckpt_dir)
print(f"ckpt_dir: {ckpt_dir}")

ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "*.pt")))
if not ckpts:
    print("NO CKPTS"); sys.exit(1)
ckpt_path = ckpts[-1]
print(f"latest ckpt: {ckpt_path}")

ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
train_args = ckpt["args"]
print(f"model: {train_args.model}, image_size: {train_args.image_size}")

# Build model
device = "cuda"
from diffusion import create_diffusion
latent_size = train_args.image_size // train_args.vae_downscale
latent_channels = int(train_args.latent_channels)
model = DiT_2Cond_models[train_args.model](
    input_size=latent_size,
    in_channels=latent_channels,
    num_calligraphers=train_args.num_calligraphers,
    num_characters=train_args.num_characters,
    condition_fusion=train_args.condition_fusion,
    callig_embed_dim=int(train_args.callig_embed_dim),
    char_embed_dim=int(train_args.char_embed_dim),
    learn_sigma=True,
    cond_drop_all_prob=float(getattr(train_args, "cond_drop_all_prob", 0.05)),
    cond_drop_one_prob=float(getattr(train_args, "cond_drop_one_prob", 0.25)),
    skel_head_enabled=getattr(train_args, "w_skel_head", 0) > 0,
    use_glyph_cond=getattr(train_args, "w_glyph_cond", False),
    glyph_scale_init=float(getattr(train_args, "glyph_scale_init", 0.4)),
).to(device).eval()

# Load EMA weights
ema = ckpt.get("ema", ckpt.get("model", ckpt))
missing, unexpected = model.load_state_dict(ema, strict=False)
print(f"loaded EMA: missing={len(missing)}, unexpected={len(unexpected)}")

# Inject DINO
_inject_dino(model, train_args)

print(f"\n=== GPU mem after model load: {torch.cuda.memory_allocated()/1024**3:.2f}G ===")

# Build eval cache
img_root = getattr(train_args, "img_root", "") or ""
eval_csv = getattr(train_args, "eval_csv", "5script/eval500_clean.csv")
eval_n = int(getattr(train_args, "eval_n", 455))
cache = prepare_eval_cache(
    eval_csv, img_root, train_args.image_size, eval_n,
    train_args.vae_downscale, latent_channels, float(train_args.vae_scaling_factor))

# Load VAE
vae = load_eval_vae(train_args, device)
print(f"=== GPU mem after VAE load: {torch.cuda.memory_allocated()/1024**3:.2f}G ===")

# Test with dit_batch=240, vae_batch=32
print("\n=== Testing dit_batch=240, vae_batch=32 ===")
step = int(ckpt.get("train_steps", 0))
t0 = time.time()
run_gpu_eval(model, train_args, cache, step, ckpt_dir, device,
             dit_batch=240, vae_batch=32, ddim_steps=50, cfg_scale=4.0)
elapsed = time.time() - t0
print(f"=== dit=240/vae=32: {elapsed:.1f}s, peak VRAM: {torch.cuda.max_memory_allocated()/1024**3:.2f}G ===")

# Check saved files
step_dir = os.path.join(ckpt_dir, "eval_samples", f"step{step:07d}")
n_files = len(glob.glob(os.path.join(step_dir, "sample*.png")))
print(f"saved: {n_files} sample images + {len(glob.glob(os.path.join(step_dir, 'gt*.png')))} gt images")
