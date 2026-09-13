import torch, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ckpt = torch.load("/root/Workspace/xy/DiT/5script/results/s8_klf4_clean_dino/20260823-234546-s8-klf4-clean-dino/checkpoints/0105000.pt", map_location="cpu", weights_only=False)
print("keys:", list(ckpt.keys()))
print("train_steps:", ckpt.get("train_steps"))
print("has ema:", "ema" in ckpt)
print("has opt:", "opt" in ckpt)
