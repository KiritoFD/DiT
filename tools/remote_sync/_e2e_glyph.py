# -*- coding: utf-8 -*-
"""远程端到端验证: 楷隶 dataset 取到标准字形 latent g + 模型(XL甲2) forward 能吃 g。
不打完整训练, 只验证链路 + 显存。"""
import os, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.environ["XFORMERS_DISABLED"] = "1"
import torch
from torch.utils.data import DataLoader

from latent_dataset import MCCDLatentDataset
from models import DiT_2Cond_models
from lora import inject_lora
from download import find_model

def main():
    device = "cuda"
    print("[e2e] 构建楷隶 dataset ...")
    ds = MCCDLatentDataset(
        csv_file="kailishu_train.csv",
        latent_shards_dir="final_latents",
        img_root="final_imgs_256",
        image_size=256, load_canny=False, load_skel=False, load_image=True,
        preload=False, structure_size=256, use_glyph_cond=True)
    loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)
    b = next(iter(loader))
    print(f"[e2e] batch keys: {list(b.keys())}")
    print(f"[e2e] latent: {tuple(b['latent'].shape)}, g(glyph): {tuple(b['g'].shape)}")
    assert tuple(b['g'].shape) == (4,4,32,32), "g should be (B,4,32,32)"
    assert b['g'].abs().sum().item() > 0, "g should be non-zero (std glyph latent)"
    print(f"[e2e] g 非零, 语义值 mean={b['g'].mean().item():.3f} std={b['g'].std().item():.3f}")

    print("[e2e] 构建 DiT-2Cond-XL + glyph_cond ...")
    model = DiT_2Cond_models["DiT-2Cond-XL/2"](
        input_size=32, num_calligraphers=1011, num_characters=35130,
        use_checkpoint=False, condition_fusion="xl_highdim",
        callig_embed_dim=384, char_embed_dim=768,
        cond_drop_all_prob=0.05, cond_drop_one_prob=0.25,
        use_glyph_cond=True, glyph_scale_init=0.4)
    pre = find_model("pretrained_models/DiT-XL-2-256x256.pt")
    pre2 = {k:v for k,v in pre.items()
            if not k.startswith(("y_embedder","cond_fusion","callig_proj","char_proj",
                                 "y_callig","y_char","skel_head","glyph_embedder"))}
    model.load_state_dict(pre2, strict=False)
    inject_lora(model, r=16, lora_alpha=16, target="all")
    model = model.to(device).train()
    tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[e2e] trainable={tr/1e6:.1f}M glyph_scale={model.glyph_scale.item()}")

    x = b['latent'].to(device); g = b['g'].to(device)
    print(f"[e2e] latent dtype={x.dtype}, g dtype={g.dtype}, g shape={tuple(g.shape)}")
    t = torch.randint(0,1000,(x.shape[0],),device=device)
    yc = b['y_callig'].to(device)
    yg = b['y_char'].to(device)
    # 与训练一致: 用 bf16 autocast(autocast 会统一 conv 的 half/float)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(x.float(), t, yc, yg, g=g.float())
    print(f"[e2e] forward+g out: {tuple(out.shape)}")
    # backward
    loss = (out[:,:4]**2).mean()
    loss.backward()
    ge = model.glyph_embedder.weight.grad
    print(f"[e2e] glyph_embedder.grad abs sum: {float(ge.abs().sum().item()) if ge is not None else None}")
    print("[e2e] ALL PASS")

if __name__=="__main__":
    main()
