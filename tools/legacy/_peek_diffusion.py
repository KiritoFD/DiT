import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, "/root/Workspace/xy/DiT")
from diffusion import create_diffusion
d = create_diffusion("50")
print("type:", type(d).__name__)
print("attrs:", [a for a in dir(d) if not a.startswith("_") and "time" in a.lower() or "step" in a.lower() or "alpha" in a.lower() or "beta" in a.lower()])
print("num_timesteps:", d.num_timesteps)
print("alphas_cumprod shape:", d.alphas_cumprod.shape if hasattr(d, "alphas_cumprod") else "N/A")
# check timesteps
if hasattr(d, "timesteps"):
    print("timesteps:", d.timesteps[:5], "...", d.timesteps[-5:])
else:
    # Maybe it's in model_kwargs or we need to generate
    import torch
    ts = torch.linspace(0, d.num_timesteps - 1, 50).long().flip(0)
    print("manual timesteps:", ts[:5].tolist(), "...", ts[-5:].tolist())
