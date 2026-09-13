# -*- coding: utf-8 -*-
"""V3-A 二因子 glyph 模型 smoke test：构建、forward、4-way mask、CFG、参数量。"""
import torch
from models import DiT_2Cond_models

def main():
    torch.manual_seed(0)
    model = DiT_2Cond_models["DiT-2Cond-S/2"](
        input_size=32, num_calligraphers=1011, num_characters=35130,
        use_checkpoint=False, condition_fusion="factorized_add",
        callig_embed_dim=128, char_embed_dim=192,
        cond_drop_all_prob=0.10, cond_drop_one_prob=0.30)
    model.train()
    total = sum(p.numel() for p in model.parameters())
    print(f"[smoke] trainable params: {total:,}")
    assert hasattr(model, "callig_proj") and model.cond_fusion is None

    # --- forward + 4-way mask 统计 ---
    N = 512
    yc = torch.randint(0, 1011, (N,))
    yg = torch.randint(0, 35130, (N,))
    x = torch.randn(N, 4, 32, 32)
    t = torch.randint(0, 1000, (N,))
    out = model(x, t, yc, yg)
    assert out.shape == (N, 8, 32, 32), out.shape

    # mask 计数：r<0.10 uncond；0.10..0.40 drop_one（which 0/1 各半）
    import torch as _t
    r = _t.rand(N)
    drop_all = r < 0.10
    drop_one = (r >= 0.10) & (r < 0.40)
    which = _t.randint(0, 2, (N,))
    drop_glyph = drop_one & (which == 1)
    drop_callig = drop_one & (which == 0)
    exp_uncond = drop_all.sum().item() / N
    exp_callig_only = drop_callig.sum().item() / N
    exp_glyph_only = drop_glyph.sum().item() / N
    exp_full = (~drop_all & ~drop_one).sum().item() / N
    print(f"[smoke] expected mask rates: full={exp_full:.3f} callig_only={exp_callig_only:.3f} "
          f"glyph_only={exp_glyph_only:.3f} uncond={exp_uncond:.3f}")

    # --- CFG（batch>1 正确性） ---
    model.eval()
    xb = torch.randn(8, 4, 32, 32)
    tb = torch.randint(0, 1000, (8,))
    ycb = torch.randint(0, 1011, (8,))
    ygb = torch.randint(0, 35130, (8,))
    out1 = model.forward_with_cfg(xb, tb, ycb, ygb, cfg_scale=4.0)
    out2 = model.forward_with_cfg(xb[:1], tb[:1], ycb[:1], ygb[:1], cfg_scale=4.0)
    assert out1.shape == (8, 8, 32, 32)
    print(f"[smoke] CFG batch8 vs batch1 max diff: {abs(out1[0]-out2[0]).max().item():.2e}")
    assert abs(out1[0] - out2[0]).max().item() < 1e-3, "CFG batch inconsistency"

    # --- 零初始化的 marginal 行为：drop 后 embedding 用 num_classes 行 ---
    model.train()
    yc_dropped = torch.full((4,), model.y_callig_embedder.num_classes)
    yg_full = torch.tensor([0, 1, 2, 3])
    out_g = model(x[:4], t[:4], yc_dropped, yg_full)
    assert out_g.shape == (4, 8, 32, 32)
    print("[smoke] glyph-only forward OK")
    yc_full = torch.tensor([0, 1, 2, 3])
    yg_dropped = torch.full((4,), model.y_char_embedder.num_classes)
    out_a = model(x[:4], t[:4], yc_full, yg_dropped)
    assert out_a.shape == (4, 8, 32, 32)
    print("[smoke] callig-only forward OK")
    print("[smoke] ALL PASS")

if __name__ == "__main__":
    main()
