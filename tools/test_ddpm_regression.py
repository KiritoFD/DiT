"""DDPM regression + model build smoke test (local GPU or CPU).

Verifies:
1. create_diffusion_or_flow('ddpm') returns a working SpacedDiffusion
   (training_losses on a tiny model with int timesteps).
2. DiT-2Cond-S/2 model builds and runs a forward pass with cond kwargs.
3. Flow forward path via model with t*TIME_SCALE works with the real model.
"""
import os
os.environ["XFORMERS_DISABLED"] = "1"
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

from diffusion import create_diffusion_or_flow
from models import DiT_2Cond_models


def test_ddpm_training_losses():
    d = create_diffusion_or_flow('', 'ddpm')  # learn_sigma=True: model outputs 2C
    assert not getattr(d, 'is_flow', False), "ddpm should not be flow"
    B, C, H, W = 2, 4, 8, 8

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(C * H * W, 2 * C * H * W)
        def forward(self, x, t, **kw):
            return self.lin(x.flatten(1)).view(x.shape[0], 2 * C, H, W)

    net = Net()
    x0 = torch.randn(B, C, H, W)
    t = torch.randint(0, 1000, (B,))
    out = d.training_losses(net, x0, t)
    assert "loss" in out and out["loss"].shape == (B,)
    print(f"[ok] DDPM training_losses works, loss mean={out['loss'].mean().item():.4f}")


def test_model_forward_ddpm_and_flow():
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(0)
    model = DiT_2Cond_models['DiT-2Cond-S/2'](
        num_calligraphers=1011, num_characters=35130, condition_fusion='factorized_add',
        callig_embed_dim=128, char_embed_dim=384,
    ).to(dev).eval()

    npar = sum(p.numel() for p in model.parameters())
    print(f"[ok] DiT-2Cond-S/2 params: {npar/1e6:.1f}M")

    B = 2
    x = torch.randn(B, 4, 32, 32, device=dev)
    y_callig = torch.randint(0, 1011, (B,), device=dev)
    y_char = torch.randint(0, 35130, (B,), device=dev)

    with torch.no_grad():
        # DDPM convention: t in [0, 1000), model outputs 2C (learn_sigma)
        t_ddpm = torch.randint(0, 1000, (B,), device=dev).float()
        out = model(x, t_ddpm, y_callig=y_callig, y_char=y_char)
        if isinstance(out, tuple):
            out = out[0]
        assert out.shape == (B, 8, 32, 32), f"ddpm out {out.shape} != (B,2C,32,32)"
        print(f"[ok] DDPM forward: out {tuple(out.shape)} (2C learn_sigma)")

        # Flow convention: t in [0,1] scaled by 1000
        t_flow = torch.rand(B, device=dev) * 1000.0
        out2 = model(x, t_flow, y_callig=y_callig, y_char=y_char)
        if isinstance(out2, tuple):
            out2 = out2[0]
        assert out2.shape == (B, 8, 32, 32)
        print(f"[ok] Flow forward (t*1000): out {tuple(out2.shape)} (2C)")


def test_flow_euler_with_real_model():
    """End-to-end: build FlowMatching + real DiT, run Euler sampling."""
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(1)
    model = DiT_2Cond_models['DiT-2Cond-S/2'](
        num_calligraphers=1011, num_characters=35130, condition_fusion='factorized_add',
        callig_embed_dim=128, char_embed_dim=384,
    ).to(dev).eval()
    fm = create_diffusion_or_flow('8', 'flow')  # 8 Euler steps

    B = 2
    z = torch.randn(B, 4, 32, 32, device=dev)
    y_callig = torch.randint(0, 1011, (B,), device=dev)
    y_char = torch.randint(0, 35130, (B,), device=dev)
    mk = dict(y_callig=y_callig, y_char=y_char)

    with torch.no_grad():
        samples = fm.ddim_sample_loop(model, z.shape, z, model_kwargs=mk, device=dev)
    assert samples.shape == (B, 4, 32, 32), f"flow samples {samples.shape} != (B,C,H,W)"
    print(f"[ok] Flow Euler 8-step with real DiT: out {tuple(samples.shape)}, "
          f"finite={torch.isfinite(samples).all().item()}, "
          f"norm={samples.float().norm().item():.3f}")


if __name__ == "__main__":
    test_ddpm_training_losses()
    test_model_forward_ddpm_and_flow()
    test_flow_euler_with_real_model()
    print("\nALL REGRESSION TESTS PASSED")
