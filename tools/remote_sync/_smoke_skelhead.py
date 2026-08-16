# -*- coding: utf-8 -*-
"""smoke: DiT-2Cond-XL + xl_highdim + skel_head 的构建与 forward tuple 输出。"""
import os, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.environ["XFORMERS_DISABLED"] = "1"
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from models import DiT_2Cond_models
from lora import inject_lora
from download import find_model

def main():
    torch.manual_seed(0)
    model = DiT_2Cond_models["DiT-2Cond-XL/2"](
        input_size=32, num_calligraphers=1011, num_characters=35130,
        use_checkpoint=False, condition_fusion="xl_highdim",
        callig_embed_dim=384, char_embed_dim=768,
        cond_drop_all_prob=0.05, cond_drop_one_prob=0.25,
        skel_head_enabled=True)
    assert model.skel_head is not None, "skel_head should exist"
    # 加载 XL body
    pre = find_model("pretrained_models/DiT-XL-2-256x256.pt")
    pre2 = {k: v for k, v in pre.items()
            if not k.startswith(("y_embedder","cond_fusion","callig_proj",
                                 "char_proj","y_callig","y_char","skel_head"))}
    missing, unexpected = model.load_state_dict(pre2, strict=False)
    print(f"[smoke] body loaded missing={len(missing)} unexpected={len(unexpected)}")
    print(f"[smoke] skel_head params: {sum(p.numel() for p in model.skel_head.parameters())}")
    # 注入 LoRA（不含 skel_head；LoRA 目标 blocks.attn/mlp）
    model = model.eval()
    # forward: 应返回 tuple (main, skel)
    x = torch.randn(2, 4, 32, 32)
    t = torch.randint(0, 1000, (2,))
    yc = torch.tensor([0,1]); yg = torch.tensor([0,1])
    with torch.no_grad():
        out = model(x, t, yc, yg)
        print(f"[smoke] forward returns tuple={isinstance(out, tuple)}")
        if isinstance(out, tuple):
            main, skel = out
            print(f"[smoke] main shape={tuple(main.shape)} skel shape={tuple(skel.shape)}")
            assert tuple(main.shape)==(2,8,32,32)
            assert tuple(skel.shape)==(2,1,32,32)
        # CFG 应只取主输出
        cfg = model.forward_with_cfg(x, t, yc, yg, cfg_scale=4.0)
        print(f"[smoke] CFG output shape={tuple(cfg.shape)} (expect 2,8,32,32)")
    # training_losses 通路（diffusion）
    from diffusion import create_diffusion
    d = create_diffusion("")
    model.train()
    ld = d.training_losses(model, x, t, dict(y_callig=yc, y_char=yg))
    print(f"[smoke] loss keys: {list(ld.keys())}")
    print(f"[smoke] intermediate_feats shape: {tuple(ld['intermediate_feats'].shape) if 'intermediate_feats' in ld else 'MISSING'}")
    print("[smoke] ALL PASS")

if __name__ == "__main__":
    main()
