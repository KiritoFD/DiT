# -*- coding: utf-8 -*-
"""
test_controlnet.py — 本地测试 ControlNet 算法正确性.

验证:
  1. ControlNetDiT 可正常构建和 forward (不依赖远程/GPU)
  2. zero-init: cond=None 和 cond=随机图 初始输出应完全相同 (完美 warm-start)
  3. ctrl_encoder 参数 trainable, 主模型参数 frozen
  4. forward_with_cfg 形状正确
  5. checkpoint 保存/加载 round-trip

运行: python tools/controlnet/test_controlnet.py
"""
import os
os.environ["XFORMERS_DISABLED"] = "1"
import sys
_s = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src")
if _s not in sys.path:
    sys.path.insert(0, _s)
import torch
import torch.nn as nn
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from controlnet_dit import ControlNetDiT, ControlConditionEncoder, DiTBlockSimple

from models import DiT_2Cond_models


def test_construction():
    """测试 ControlNetDiT 构建."""
    print("=== Test 1: Construction ===")
    main = DiT_2Cond_models["DiT-2Cond-S/2"](
        num_calligraphers=100, num_characters=1000,
        condition_fusion="factorized_add",
        callig_embed_dim=128, char_embed_dim=256,
        cond_drop_all_prob=0.05, cond_drop_one_prob=0.25,
        use_checkpoint=False, learn_sigma=True)
    ctrl = ControlNetDiT(main, cond_in_channels=1, train_ctrl_only=True)

    # Check trainable
    trainable = sum(p.numel() for p in ctrl.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in ctrl.parameters() if not p.requires_grad)
    print(f"  trainable: {trainable:,} | frozen: {frozen:,}")
    assert trainable > 0, "ctrl_encoder should have trainable params"
    assert frozen > 0, "main model should be frozen"

    # Check main is frozen
    for name, p in ctrl.main.named_parameters():
        assert not p.requires_grad, f"main param {name} should be frozen"
    # Check ctrl_encoder is trainable
    for name, p in ctrl.ctrl_encoder.named_parameters():
        assert p.requires_grad, f"ctrl param {name} should be trainable"
    print("  PASSED: main frozen, ctrl trainable")


def test_zero_init_warm_start():
    """zero-init: cond=None 和 cond=随机图 初始输出应完全相同."""
    print("=== Test 2: Zero-init warm-start ===")
    torch.manual_seed(42)
    main = DiT_2Cond_models["DiT-2Cond-S/2"](
        num_calligraphers=100, num_characters=1000,
        condition_fusion="factorized_add",
        callig_embed_dim=128, char_embed_dim=256,
        cond_drop_all_prob=0.05, cond_drop_one_prob=0.25,
        use_checkpoint=False, learn_sigma=True)
    ctrl = ControlNetDiT(main, cond_in_channels=1, train_ctrl_only=True)
    ctrl.eval()

    B = 2
    x = torch.randn(B, 4, 32, 32)
    t = torch.randint(0, 1000, (B,))
    y_callig = torch.randint(0, 100, (B,))
    y_char = torch.randint(0, 1000, (B,))

    with torch.no_grad():
        out_no_cond = ctrl(x, t, y_callig, y_char, cond=None)
        skel = torch.rand(B, 1, 256, 256).round()
        out_with_cond = ctrl(x, t, y_callig, y_char, cond=skel)

    diff = (out_no_cond - out_with_cond).abs().max().item()
    print(f"  max diff (no_cond vs cond): {diff:.2e}")
    assert diff < 1e-5, f"zero-init should give identical outputs, got diff={diff}"
    print("  PASSED: zero-init perfect warm-start")


def test_forward_shapes():
    """forward 和 forward_with_cfg 输出形状."""
    print("=== Test 3: Forward shapes ===")
    main = DiT_2Cond_models["DiT-2Cond-S/2"](
        num_calligraphers=100, num_characters=1000,
        condition_fusion="factorized_add",
        callig_embed_dim=128, char_embed_dim=256,
        cond_drop_all_prob=0.05, cond_drop_one_prob=0.25,
        use_checkpoint=False, learn_sigma=True)
    ctrl = ControlNetDiT(main, cond_in_channels=1, train_ctrl_only=True)
    ctrl.eval()

    B = 4
    x = torch.randn(B, 4, 32, 32)
    t = torch.randint(0, 1000, (B,))
    y_callig = torch.randint(0, 100, (B,))
    y_char = torch.randint(0, 1000, (B,))
    skel = torch.rand(B, 1, 256, 256).round()

    with torch.no_grad():
        out = ctrl(x, t, y_callig, y_char, cond=skel)
    assert out.shape == (B, 8, 32, 32), f"forward shape {out.shape}, expected (B,8,32,32)"
    print(f"  forward: {out.shape} OK")

    with torch.no_grad():
        out_cfg = ctrl.forward_with_cfg(x, t, y_callig, y_char, cfg_scale=4.0, cond=skel)
    assert out_cfg.shape == (B, 8, 32, 32), f"cfg shape {out_cfg.shape}"
    print(f"  forward_with_cfg: {out_cfg.shape} OK")


def test_gradient_flow():
    """梯度只流到 ctrl_encoder, 不流到主模型.

    关键: 训练时主模型加载了预训练 ckpt, final_layer 已非零.
    测试中我们手动模拟 'loaded' 状态 (打破 final_layer 的 zero-init),
    否则 fresh 模型的 final_layer=0 会阻断所有梯度.
    """
    print("=== Test 4: Gradient flow ===")
    main = DiT_2Cond_models["DiT-2Cond-S/2"](
        num_calligraphers=100, num_characters=1000,
        condition_fusion="factorized_add",
        callig_embed_dim=128, char_embed_dim=256,
        cond_drop_all_prob=0.05, cond_drop_one_prob=0.25,
        use_checkpoint=False, learn_sigma=True)
    ctrl = ControlNetDiT(main, cond_in_channels=1, train_ctrl_only=True)
    ctrl.train()

    # Simulate "loaded pretrained checkpoint": break the zero-init on final_layer
    # (a fresh model has final_layer=0 from initialize_weights, which blocks ALL
    # gradient. After loading the 195k ckpt, final_layer is non-zero.)
    with torch.no_grad():
        fl = ctrl.main.final_layer
        fl.linear.weight.normal_(0, 0.01)
        fl.adaLN_modulation[-1].weight.normal_(0, 0.01)
        fl.adaLN_modulation[-1].bias.normal_(0, 0.01)

    B = 2
    x = torch.randn(B, 4, 32, 32)
    t = torch.randint(0, 1000, (B,))
    y_callig = torch.randint(0, 100, (B,))
    y_char = torch.randint(0, 1000, (B,))
    skel = torch.rand(B, 1, 256, 256).round()

    out = ctrl(x, t, y_callig, y_char, cond=skel)
    loss = out.mean()
    loss.backward()

    # With final_layer non-zero (loaded ckpt), gradient flows to ctrl_encoder.
    # At zero-init out_projs: d(out)/d(ctrl_feat)=proj_w=0, so ctrl_blocks get
    # NO grad. But out_projs.weight gets grad = ctrl_feat * d(out)/d(injected),
    # and out_projs.bias gets grad = d(out)/d(injected). After first optimizer
    # step, proj_w becomes non-zero, and THEN ctrl_blocks start getting grad.
    proj_params = [p for n, p in ctrl.ctrl_encoder.named_parameters() if "out_projs" in n]
    proj_grad = sum(p.grad.norm().item() for p in proj_params if p.grad is not None)
    print(f"  out_projs grad norm: {proj_grad:.6f} (non-zero = learning seed)")
    assert proj_grad > 0, "out_projs should get gradient (the learning seed)"

    block_params = [p for n, p in ctrl.ctrl_encoder.named_parameters() if "ctrl_blocks" in n]
    block_grad = sum(p.grad.norm().item() for p in block_params if p.grad is not None)
    print(f"  ctrl_blocks grad norm: {block_grad:.6f} (expect ~0 at zero-init proj)")

    # Main model should have NO gradients (frozen)
    main_grad = sum(p.grad.norm().item() for p in ctrl.main.parameters() if p.grad is not None)
    print(f"  main model grad norm: {main_grad:.6f} (expect 0, frozen)")
    assert main_grad == 0, "main model should have no gradients"
    print("  PASSED: gradient flow verified (zero-init semantics)")


def test_checkpoint_roundtrip():
    """checkpoint 保存/加载 round-trip."""
    print("=== Test 5: Checkpoint round-trip ===")
    import tempfile
    main = DiT_2Cond_models["DiT-2Cond-S/2"](
        num_calligraphers=100, num_characters=1000,
        condition_fusion="factorized_add",
        callig_embed_dim=128, char_embed_dim=256,
        cond_drop_all_prob=0.05, cond_drop_one_prob=0.25,
        use_checkpoint=False, learn_sigma=True)
    ctrl1 = ControlNetDiT(main, cond_in_channels=1, train_ctrl_only=True)

    # Perturb ctrl_encoder weights (simulate training)
    with torch.no_grad():
        for p in ctrl1.ctrl_encoder.parameters():
            p.add_(torch.randn_like(p) * 0.01)

    # Save only ctrl keys
    ck = {"ctrl": {k: v.detach().cpu() for k, v in ctrl1.state_dict().items()
                   if k.startswith("ctrl_encoder")}}

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        torch.save(ck, f.name)
        ck_path = f.name

    # Load into fresh ctrl
    main2 = DiT_2Cond_models["DiT-2Cond-S/2"](
        num_calligraphers=100, num_characters=1000,
        condition_fusion="factorized_add",
        callig_embed_dim=128, char_embed_dim=256,
        cond_drop_all_prob=0.05, cond_drop_one_prob=0.25,
        use_checkpoint=False, learn_sigma=True)
    ctrl2 = ControlNetDiT(main2, cond_in_channels=1, train_ctrl_only=True)
    ctrl_keys = {k: v for k, v in ck["ctrl"].items() if k.startswith("ctrl_encoder")}
    ctrl2.load_state_dict(ctrl_keys, strict=False)

    # Verify ctrl_encoder weights match
    for (n1, p1), (n2, p2) in zip(ctrl1.ctrl_encoder.named_parameters(),
                                   ctrl2.ctrl_encoder.named_parameters()):
        assert torch.allclose(p1, p2), f"mismatch in {n1}"
    print("  PASSED: checkpoint round-trip")
    os.unlink(ck_path)


def test_cfg_correctness():
    """CFG: cond=skel 始终提供, callig/char 有/无. At zero-init + cfg=1, should match base."""
    print("=== Test 6: CFG correctness ===")
    main = DiT_2Cond_models["DiT-2Cond-S/2"](
        num_calligraphers=100, num_characters=1000,
        condition_fusion="factorized_add",
        callig_embed_dim=128, char_embed_dim=256,
        cond_drop_all_prob=0.05, cond_drop_one_prob=0.25,
        use_checkpoint=False, learn_sigma=True)
    # Simulate loaded ckpt (non-zero final_layer)
    with torch.no_grad():
        fl = main.final_layer
        fl.linear.weight.normal_(0, 0.01)
        fl.adaLN_modulation[-1].weight.normal_(0, 0.01)
    ctrl = ControlNetDiT(main, cond_in_channels=1, train_ctrl_only=True)
    ctrl.eval()

    B = 2
    x = torch.randn(B, 4, 32, 32)
    t = torch.randint(0, 1000, (B,))
    y_callig = torch.randint(0, 100, (B,))
    y_char = torch.randint(0, 1000, (B,))
    skel = torch.rand(B, 1, 256, 256).round()

    with torch.no_grad():
        out_cfg = ctrl.forward_with_cfg(x, t, y_callig, y_char, cfg_scale=1.0, cond=skel)

    # At zero-init and cfg_scale=1.0, output should equal base model forward_with_cfg
    # (since ctrl injection is 0)
    with torch.no_grad():
        out_base = main.forward_with_cfg(x, t, y_callig, y_char, cfg_scale=1.0)

    diff = (out_cfg - out_base).abs().max().item()
    print(f"  max diff (ctrl cfg=1 vs base cfg=1): {diff:.2e}")
    assert diff < 1e-5, f"at zero-init + cfg=1, should match base, got diff={diff}"
    print("  PASSED: CFG correctness at zero-init")


if __name__ == "__main__":
    print("ControlNet Local Tests (CPU only)\n")
    test_construction()
    test_zero_init_warm_start()
    test_forward_shapes()
    test_gradient_flow()
    test_checkpoint_roundtrip()
    test_cfg_correctness()
    print("\n[OK] All tests passed!")
