"""Unit regression test for U-Net DiC switches."""
import torch
from src.unet.model import LatentUNet
from src.unet.io import Config, build

def test_equivalence():
    torch.manual_seed(42)
    # 1. Base model
    cfg_base = Config(channels=(192, 384, 576), num_res_blocks=2, max_steps=1000)
    model_base = build(cfg_base)
    model_base.eval()

    # 2. Model with residual_cond_gate
    cfg_gate = Config(channels=(192, 384, 576), num_res_blocks=2, max_steps=1000, residual_cond_gate=True)
    model_gate = build(cfg_gate)
    model_gate.eval()

    # 3. Model with stage_specific_cond (adapter mode)
    cfg_stage = Config(channels=(192, 384, 576), num_res_blocks=2, max_steps=1000,
                       stage_specific_cond=True, stage_cond_mode='adapter')
    model_stage = build(cfg_stage)
    model_stage.eval()

    # Copy base weights to gate and stage models
    base_state = model_base.state_dict()
    model_gate.load_state_dict(base_state, strict=False)
    model_stage.load_state_dict(base_state, strict=False)

    x = torch.randn(2, 4, 32, 32)
    t = torch.tensor([0.2, 0.8])
    y_callig = torch.tensor([1, 2])
    y_script = torch.tensor([0, 1])
    g = torch.randn(2, 1, 64, 64)

    with torch.no_grad():
        out_base = model_base(x, t, y_callig, g=g, y_script=y_script)
        out_gate = model_gate(x, t, y_callig, g=g, y_script=y_script)
        out_stage = model_stage(x, t, y_callig, g=g, y_script=y_script)

    diff_gate = (out_gate - out_base).abs().max().item()
    diff_stage = (out_stage - out_base).abs().max().item()
    print(f"Max diff (base vs residual_cond_gate): {diff_gate:.8e}")
    print(f"Max diff (base vs stage_cond adapter): {diff_stage:.8e}")
    assert diff_gate == 0.0, f"Expected 0.0 diff, got {diff_gate}"
    assert diff_stage == 0.0, f"Expected 0.0 diff, got {diff_stage}"

    # 4. Independent stage condition model
    cfg_indep = Config(channels=(192, 384, 576), num_res_blocks=2, max_steps=1000,
                       stage_specific_cond=True, stage_cond_mode='independent')
    model_indep = build(cfg_indep)
    model_indep.eval()
    with torch.no_grad():
        out_indep = model_indep(x, t, y_callig, g=g, y_script=y_script)
    assert out_indep.shape == (2, 4, 32, 32)

    # 5. GELU activation
    cfg_gelu = Config(channels=(192, 384, 576), num_res_blocks=2, max_steps=1000, act_fn='gelu')
    model_gelu = build(cfg_gelu)
    model_gelu.eval()
    with torch.no_grad():
        out_gelu = model_gelu(x, t, y_callig, g=g, y_script=y_script)
    assert out_gelu.shape == (2, 4, 32, 32)

    # Parameter counts
    def param_count(m): return sum(p.numel() for p in m.parameters())
    print(f"Base param count: {param_count(model_base):,}")
    print(f"+ residual_cond_gate: {param_count(model_gate):,} (+{param_count(model_gate)-param_count(model_base):,})")
    print(f"+ stage_cond (adapter): {param_count(model_stage):,} (+{param_count(model_stage)-param_count(model_base):,})")
    print(f"+ stage_cond (independent): {param_count(model_indep):,} (+{param_count(model_indep)-param_count(model_base):,})")
    print("ALL TESTS PASSED!")

if __name__ == '__main__':
    test_equivalence()
