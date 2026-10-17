import sys
import os
import torch
from torch.utils.flop_counter import FlopCounterMode

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.unet.io import Config, build

def main():
    cfg = Config(channels=(192, 384, 576), num_res_blocks=2)
    model = build(cfg, initialize_tables=True).cuda()
    model.eval()
    
    b = 2
    layout = torch.channels_last
    model = model.to(memory_format=layout)
    x = torch.randn(b, 4, 32, 32, device="cuda").to(memory_format=layout)
    t = torch.tensor([500.0] * b, device="cuda")
    w = torch.zeros(b, dtype=torch.long, device="cuda")
    s = torch.zeros(b, dtype=torch.long, device="cuda")
    g = torch.rand(b, 1, 64, 64, device="cuda")
    
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model Parameters: {params:.2f} M")
    
    with FlopCounterMode(display=False) as flop_mode:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(x, t, w, g=g, y_script=s)
        fwd_flops = flop_mode.get_total_flops()
    
    fwd_per_sample = fwd_flops / b
    # In standard deep learning (conv / transformer), backward FLOPs is ~2x forward FLOPs (data grad + weight grad)
    # Total step FLOPs per sample = 3x forward FLOPs
    bwd_per_sample = 2.0 * fwd_per_sample
    total_step_per_sample = 3.0 * fwd_per_sample
    
    print(f"Exact Forward FLOPs per image: {fwd_per_sample / 1e9:.4f} GFLOPs")
    print(f"Exact Backward FLOPs per image: {bwd_per_sample / 1e9:.4f} GFLOPs")
    print(f"Exact Total (Fwd+Bwd) FLOPs per image: {total_step_per_sample / 1e9:.4f} GFLOPs")

if __name__ == "__main__":
    main()
