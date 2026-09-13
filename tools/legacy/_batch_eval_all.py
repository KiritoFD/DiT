"""Batch eval all existing ckpts: DiT DDIM (bf16, batch=240) → VAE decode (fp32, batch=32) → save PNGs."""
import os, sys, time, json, glob
os.environ["XFORMERS_DISABLED"] = "1"
import torch
import numpy as np

sys.path.insert(0, "/root/Workspace/xy/DiT")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.chdir("/root/Workspace/xy/DiT")

from models import DiT_2Cond_models
from in_process_eval import prepare_eval_cache, run_gpu_eval, load_eval_vae

def _inject_dino(model, args):
    emb_path = getattr(args, "char_dino_embeddings", None)
    idx_path = getattr(args, "char_dino_index", None)
    if not emb_path or not idx_path: return
    emb = np.load(emb_path)
    with open(idx_path, encoding="utf-8") as f:
        idx_data = json.load(f)
    glyphs = idx_data.get("glyphs", idx_data)
    table = model.y_char_embedder.embedding_table.weight
    NUM_CH = 7026
    with torch.no_grad():
        for gi, (sid, cid) in enumerate(glyphs):
            gid = int(sid) * NUM_CH + int(cid)
            if 0 <= gid < table.shape[0] and gi < emb.shape[0]:
                e = emb[gi]
                e = e / (np.linalg.norm(e) + 1e-8)
                table.data[gid] = torch.from_numpy(e).float()

# Setup
marker = "5script/results/s10_b4_grey_clear/_active_ckpt_dir.txt"
with open(marker) as f:
    ckpt_dir = f.read().strip()
ckpt_dir = os.path.join("/root/Workspace/xy/DiT", ckpt_dir)
print(f"ckpt_dir: {ckpt_dir}")

ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "*.pt")))
print(f"ckpts: {len(ckpts)}")

# Load first ckpt to get train_args
first = torch.load(ckpts[0], map_location="cpu", weights_only=False)
train_args = first["args"]
ckpt = first

device = "cuda"
latent_size = train_args.image_size // train_args.vae_downscale
latent_channels = int(train_args.latent_channels)
model = DiT_2Cond_models[train_args.model](
    input_size=latent_size, in_channels=latent_channels,
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
_inject_dino(model, train_args)
print(f"model loaded, VRAM: {torch.cuda.memory_allocated()/1024**3:.2f}G")

# Build eval cache
img_root = getattr(train_args, "img_root", "") or ""
cache = prepare_eval_cache(
    getattr(train_args, "eval_csv", "5script/eval500_clean.csv"),
    img_root, train_args.image_size, int(getattr(train_args, "eval_n", 455)),
    train_args.vae_downscale, latent_channels, float(train_args.vae_scaling_factor))

# Load VAE once
vae = load_eval_vae(train_args, device)

# Eval each ckpt
for ckpt_path in ckpts:
    name = os.path.basename(ckpt_path)
    step = int(name.replace(".pt", ""))
    print(f"\n{'='*60}")
    print(f"evaluating {name} (step {step})")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ema = ckpt.get("ema", ckpt.get("model", ckpt))
    missing, unexpected = model.load_state_dict(ema, strict=False)
    print(f"loaded EMA: missing={len(missing)}, unexpected={len(unexpected)}")

    run_gpu_eval(model, train_args, cache, step, ckpt_dir, device,
                 dit_batch=240, vae_batch=32, ddim_steps=50, cfg_scale=4.0)
    del ckpt, ema
    torch.cuda.empty_cache()

print(f"\nAll done! Peak VRAM: {torch.cuda.max_memory_allocated()/1024**3:.2f}G")
