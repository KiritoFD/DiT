import sys
import os
import time
import json
import torch
import torch.nn as nn
from torch.nn import functional as F

sys.path.insert(0, "/home/ds/Workspace/DiT")
from src.unet.blocks import norm, timestep_embedding, Attention, ResBlock
from src.loss.flow_matching import FlowMatching

class ScalableLatentUNet(nn.Module):
    def __init__(self, channels=(192, 384, 576), num_res_blocks=2, num_writers=10, num_scripts=3):
        super().__init__()
        self.channels = channels
        self.num_res_blocks = num_res_blocks
        self.num_writers = num_writers
        self.num_scripts = num_scripts
        self.writer = nn.Embedding(num_writers + 1, 32)
        self.script = nn.Embedding(num_scripts + 1, 16)
        self.time_proj = nn.Linear(128, 512)
        self.writer_proj = nn.Linear(32, 512)
        self.script_proj = nn.Linear(16, 512)
        self.condition = nn.Sequential(nn.SiLU(), nn.Linear(512, 512))
        
        self.std_stem = nn.Sequential(nn.Conv2d(1, 64, 3, stride=2, padding=1), nn.SiLU(), nn.Conv2d(64, 64, 3, padding=1))
        self.std_down = nn.ModuleList([
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.Conv2d(128, 192, 3, stride=2, padding=1)
        ])
        self.std_proj = nn.ModuleList([nn.Conv2d(s, c, 1) for s, c in zip((64, 128, 192), channels)])
        
        self.input = nn.Conv2d(4, channels[0], 3, padding=1)
        
        # Encoders
        self.enc_blocks = nn.ModuleList()
        self.enc_attns = nn.ModuleList()
        self.down = nn.ModuleList()
        
        for i, c in enumerate(channels):
            stage_blocks = nn.ModuleList([ResBlock(c, c) for _ in range(num_res_blocks)])
            self.enc_blocks.append(stage_blocks)
            self.enc_attns.append(Attention(c) if i > 0 else nn.Identity())
            if i < len(channels) - 1:
                self.down.append(nn.Conv2d(channels[i], channels[i+1], 3, stride=2, padding=1))
                
        # Middle
        self.middle1 = ResBlock(channels[-1], channels[-1])
        self.middle_attn = Attention(channels[-1])
        self.middle2 = ResBlock(channels[-1], channels[-1])
        
        # Decoders
        self.dec_blocks = nn.ModuleList()
        self.dec_attns = nn.ModuleList()
        self.up = nn.ModuleList()
        for i, c in enumerate(channels):
            stage_blocks = nn.ModuleList([ResBlock(c*2 if j == 0 else c, c) for j in range(num_res_blocks)])
            self.dec_blocks.append(stage_blocks)
            self.dec_attns.append(Attention(c) if i > 0 else nn.Identity())
            if i > 0:
                self.up.append(nn.Conv2d(channels[i], channels[i-1], 3, padding=1))
                
        self.output = nn.Sequential(norm(channels[0]), nn.SiLU(), nn.Conv2d(channels[0], 4, 3, padding=1))
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def forward(self, x, t, y_callig, g=None, y_script=None):
        c = self.condition(self.time_proj(timestep_embedding(t)) +
                           self.writer_proj(self.writer(y_callig)) +
                           self.script_proj(self.script(y_script)))
        glyph = self.std_stem(g)
        h = self.input(x)
        skips = []
        for i, stage_blocks in enumerate(self.enc_blocks):
            if i > 0:
                glyph = F.silu(self.std_down[i-1](glyph))
            h = h + self.std_proj[i](glyph)
            for block in stage_blocks:
                h = block(h, c)
            h = self.enc_attns[i](h)
            skips.append(h)
            if i < len(self.channels) - 1:
                h = self.down[i](h)
                
        h = self.middle2(self.middle_attn(self.middle1(h, c)), c)
        
        for i in reversed(range(len(self.channels))):
            h = torch.cat([h, skips[i]], dim=1)
            for block in self.dec_blocks[i]:
                h = block(h, c)
            h = self.dec_attns[i](h)
            if i > 0:
                h = self.up[i-1](F.interpolate(h, scale_factor=2, mode="nearest"))
        return self.output(h)

def test_model(channels, num_res_blocks, batch, compile_model=True):
    model = ScalableLatentUNet(channels=channels, num_res_blocks=num_res_blocks).cuda().to(memory_format=torch.channels_last)
    params = sum(p.numel() for p in model.parameters()) / 1e6
    opt = torch.optim.AdamW(model.parameters(), lr=5e-5, fused=True)
    process = FlowMatching(num_steps=50, sampler="heun")
    
    forward_fn = torch.compile(model, mode="default") if compile_model else model
    
    layout = torch.channels_last
    z = torch.randn(batch, 4, 32, 32, device="cuda").to(memory_format=layout)
    g = torch.rand(batch, 1, 64, 64, device="cuda")
    w = torch.randint(0, 10, (batch,), device="cuda")
    s = torch.randint(0, 3, (batch,), device="cuda")
    
    # Warmup
    for _ in range(3 if compile_model else 2):
        opt.zero_grad(set_to_none=True)
        t = process.sample_t(batch, z.device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = forward_fn(z, t * 1000.0, w, g=g, y_script=s)
            loss = F.mse_loss(pred, z)
        loss.backward()
        opt.step()
        
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    
    t0 = time.perf_counter()
    steps = 10
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        t = process.sample_t(batch, z.device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = forward_fn(z, t * 1000.0, w, g=g, y_script=s)
            loss = F.mse_loss(pred, z)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    
    peak_alloc = torch.cuda.max_memory_allocated() / (1024**3)
    peak_res = torch.cuda.max_memory_reserved() / (1024**3)
    step_rate = steps / elapsed
    img_rate = step_rate * batch
    
    return {
        "params_M": params,
        "batch": batch,
        "compile": compile_model,
        "step_s": step_rate,
        "img_s": img_rate,
        "peak_alloc_GiB": peak_alloc,
        "peak_res_GiB": peak_res,
    }

def main():
    print("Testing Scalable U-Net models on RTX 4090 (48GB) to find config eating ~40GB VRAM...")
    
    configs = [
        ((192, 384, 576), 2, "72M_DiT24L_aligned"),
        ((192, 384, 576), 3, "93M_Deep_scaled"),
        ((256, 512, 768), 2, "125M_DiTB_aligned"),
    ]
    
    for ch, n_res, tag in configs:
        m = ScalableLatentUNet(channels=ch, num_res_blocks=n_res)
        p_m = sum(p.numel() for p in m.parameters()) / 1e6
        print("\n" + "="*60)
        print("Model: %s | Params: %.2fM | Channels: %s | ResBlocks: %d" % (tag, p_m, str(ch), n_res))
        print("="*60)
        for b in [384, 768, 1152, 1536, 1920, 2304]:
            try:
                torch.cuda.empty_cache()
                res = test_model(ch, n_res, b, compile_model=True)
                print("  Batch=%4d | %.2f st/s | %.0f img/s | Peak Alloc: %.2f GiB | Reserved: %.2f GiB" % (
                    res["batch"], res["step_s"], res["img_s"], res["peak_alloc_GiB"], res["peak_res_GiB"]
                ), flush=True)
                if res["peak_alloc_GiB"] >= 36.0 or res["peak_res_GiB"] >= 42.0:
                    print("  >>> TARGET REACHED: VRAM eaten ~40GB at Batch=%d <<<" % b)
                    break
            except torch.cuda.OutOfMemoryError:
                print("  Batch=%4d -> OOM!" % b)
                break
            except Exception as e:
                import traceback
                print("  Batch=%4d -> Error: %s" % (b, str(e)))
                traceback.print_exc()
                break

if __name__ == "__main__":
    main()
