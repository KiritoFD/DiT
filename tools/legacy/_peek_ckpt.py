import torch, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ckpt = torch.load("/root/Workspace/xy/DiT/5script/results/s10_b4_grey_clear/20260824-140208-s10-b4-grey-clear/checkpoints/0001000.pt",
                  map_location="cpu", weights_only=False)
ema = ckpt.get("ema", ckpt.get("model", ckpt))
# check final_layer shape
for k in ema:
    if "final_layer" in k or "x_embedder" in k:
        print(k, ema[k].shape)
        if "final_layer" in k:
            break
