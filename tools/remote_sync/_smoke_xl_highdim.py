# -*- coding: utf-8 -*-
"""xl_highdim 模型 smoke：验证条件维度 1152、cond_fusion 存在、能加载 XL body、前向+CFG。"""
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
        cond_drop_all_prob=0.10, cond_drop_one_prob=0.30)
    total = sum(p.numel() for p in model.parameters())
    # 检查条件头维度
    dc = model.y_callig_embedder.embedding_table.weight.shape[1]
    dg = model.y_char_embedder.embedding_table.weight.shape[1]
    hdim = model.x_embedder.proj.weight.shape[0]
    print(f"[smoke-xl] total={total/1e6:.1f}M hidden={hdim} callig_dim={dc} glyph_dim={dg} (dc+dg={dc+dg})")
    assert hdim == 1152, "XL hidden should be 1152"
    assert model.cond_fusion is not None, "xl_highdim needs cond_fusion"
    # cond_fusion 输出维度
    last = model.cond_fusion[-1]
    print(f"[smoke-xl] cond_fusion out dim = {last.out_features}")

    # 加载 XL body（过滤条件头）
    pre = find_model("pretrained_models/DiT-XL-2-256x256.pt")
    pre2 = {k: v for k, v in pre.items()
            if not k.startswith(("y_embedder", "cond_fusion", "callig_proj",
                                 "char_proj", "y_callig", "y_char"))}
    missing, unexpected = model.load_state_dict(pre2, strict=False)
    print(f"[smoke-xl] body loaded: missing={len(missing)} unexpected={len(unexpected)}")
    # missing 应仅为条件头相关

    # 注入 LoRA
    inject_lora(model, r=16, lora_alpha=16, target="all")
    model = model.eval()

    # forward
    x = torch.randn(2, 4, 32, 32)
    t = torch.randint(0, 1000, (2,))
    yc = torch.tensor([0, 1])
    yg = torch.tensor([0, 1])
    with torch.no_grad():
        out = model(x, t, yc, yg)
        print(f"[smoke-xl] forward out shape: {tuple(out.shape)} (expect (2,8,32,32))")
        assert out.shape == (2, 8, 32, 32)
        outg = model(x, t, torch.full((2,), model.y_callig_embedder.num_classes), yg)
        outa = model(x, t, yc, torch.full((2,), model.y_char_embedder.num_classes))
        outf = model.forward_with_cfg(x, t, yc, yg, cfg_scale=4.0)
        print(f"[smoke-xl] glyph-only / callig-only / CFG all OK, shapes: "
              f"{tuple(outg.shape)} {tuple(outa.shape)} {tuple(outf.shape)}")
    print("[smoke-xl] ALL PASS")

if __name__ == "__main__":
    main()
