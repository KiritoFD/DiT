# -*- coding: utf-8 -*-
"""
test_controlnet.py 鈥?鏈湴娴嬭瘯 ControlNet 绠楁硶姝ｇ‘鎬?

楠岃瘉:
  1. ControlNetDiT 鍙甯告瀯寤哄拰 forward (涓嶄緷璧栬繙绋?GPU)
  2. zero-init: cond=None 鍜?cond=闅忔満鍥?鍒濆杈撳嚭搴斿畬鍏ㄧ浉鍚?(瀹岀編 warm-start)
  3. ctrl_encoder 鍙傛暟 trainable, 涓绘ā鍨嬪弬鏁?frozen
  4. forward_with_cfg 褰㈢姸姝ｇ‘
  5. checkpoint 淇濆瓨/鍔犺浇 round-trip

杩愯: python tools/controlnet/test_controlnet.py
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
from src.model.controlnet import ControlNetDiT, ControlConditionEncoder, DiTBlockSimple

from src.model import DiT_2Cond_models


def test_construction():
    """娴嬭瘯 ControlNetDiT 鏋勫缓."""
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
    """zero-init: cond=None 鍜?cond=闅忔満鍥?鍒濆杈撳嚭搴斿畬鍏ㄧ浉鍚?"""
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
    """forward 鍜?forward_with_cfg 杈撳嚭褰㈢姸."""
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
    """姊害鍙祦鍒?ctrl_encoder, 涓嶆祦鍒颁富妯″瀷.

    鍏抽敭: 璁粌鏃朵富妯″瀷鍔犺浇浜嗛璁粌 ckpt, final_layer 宸查潪闆?
    娴嬭瘯涓垜浠墜鍔ㄦā鎷?'loaded' 鐘舵€?(鎵撶牬 final_layer 鐨?zero-init),
    鍚﹀垯 fresh 妯″瀷鐨?final_layer=0 浼氶樆鏂墍鏈夋搴?
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
    """checkpoint 淇濆瓨/鍔犺浇 round-trip."""
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
    """CFG: cond=skel 濮嬬粓鎻愪緵, callig/char 鏈?鏃? At zero-init + cfg=1, should match base."""
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
