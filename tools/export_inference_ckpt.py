import torch
import os

ckpt_path = "/home/ds/Workspace/moyi/results/moyi_top10_rf/checkpoints/moyi_0025000.pt"
out_dir = "/home/ds/Workspace/moyi/results/moyi_top10_rf/checkpoints"

print(f"Loading {ckpt_path}...")
ckpt = torch.load(ckpt_path, map_location="cpu")

# 1. Weights only (model + ema + args + step, no optimizer state)
weights_path = os.path.join(out_dir, "moyi_0025000_weights.pt")
torch.save({
    "step": ckpt["step"],
    "model": ckpt["model"],
    "ema": ckpt["ema"],
    "args": ckpt.get("args", {}),
}, weights_path)
print(f"✓ Saved weights only: {weights_path} ({os.path.getsize(weights_path) / (1024**2):.1f} MB)")

# 2. EMA float16 weights (lightweight inference package)
ema_fp16 = {k: v.half() for k, v in ckpt["ema"].items()}
ema_path = os.path.join(out_dir, "moyi_0025000_ema_fp16.pt")
torch.save({
    "step": ckpt["step"],
    "ema": ema_fp16,
    "args": ckpt.get("args", {}),
}, ema_path)
print(f"✓ Saved EMA fp16 weights: {ema_path} ({os.path.getsize(ema_path) / (1024**2):.1f} MB)")
