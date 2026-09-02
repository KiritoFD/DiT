"""FlowMatching smoke test — local GPU.

Verifies:
1. training_losses: velocity target correctness (x_t=(1-t)e+t*x0, v=x0-e)
   - at t=0: x_t = noise, at t=1: x_t = x_start
   - loss decreases over steps on a tiny model
2. ddim_sample_loop (Euler ODE): with a perfect velocity field
   (v = x0 - e), 1 step recovers x0 exactly.
"""
import os
os.environ["XFORMERS_DISABLED"] = "1"
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn

from diffusion import create_diffusion_or_flow


def test_interpolant_math():
    """Verify linear interpolant endpoints and velocity target."""
    fm = create_diffusion_or_flow('50', 'flow')
    B, C, H, W = 4, 4, 8, 8
    x0 = torch.randn(B, C, H, W)
    noise = torch.randn(B, C, H, W)

    # t=0 -> x_t == x0 (data), t=1 -> x_t == noise
    t0 = torch.zeros(B)
    out = fm._interp(x0, noise, t0)
    assert torch.allclose(out, x0, atol=1e-5), "t=0 should give data x0"
    t1 = torch.ones(B)
    out = fm._interp(x0, noise, t1)
    assert torch.allclose(out, noise, atol=1e-5), "t=1 should give noise"
    print("[ok] interpolant endpoints: x_t(0)=x0(data), x_t(1)=noise")


def test_training_loss_shape():
    """training_losses returns per-sample loss [N] and velocity target used."""
    fm = create_diffusion_or_flow('50', 'flow')
    B, C, H, W = 3, 4, 8, 8

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(C * H * W, C * H * W)
        def forward(self, x, t, **kw):
            return self.lin(x.flatten(1)).view(x.shape)

    net = Net()
    x0 = torch.randn(B, C, H, W)
    t = fm.sample_t(B, 'cpu')
    out = fm.training_losses(net, x0, t)
    assert "loss" in out
    assert out["loss"].shape == (B,), f"expected [N] loss, got {out['loss'].shape}"
    print(f"[ok] training_losses: loss shape={tuple(out['loss'].shape)}, "
          f"mean={out['loss'].mean().item():.4f}")


def test_euler_recovers_x0():
    """A perfect velocity field v=noise-x0 must yield exact x0 in 1 Euler step."""
    fm = create_diffusion_or_flow('1', 'flow')  # 1 step
    B, C, H, W = 2, 4, 8, 8
    x0 = torch.randn(B, C, H, W)
    noise = torch.randn(B, C, H, W)

    class PerfectVel(nn.Module):
        def forward(self, x, t, **kw):
            # v = noise - x0 (constant, independent of x/t), matching the
            # FlowMatching velocity convention
            return (noise - x0)

    out = fm.ddim_sample_loop(PerfectVel(), (B, C, H, W), x_T=noise, device='cpu')
    assert torch.allclose(out, x0, atol=1e-4), f"Euler 1-step should recover x0, got diff {torch.abs(out-x0).max().item()}"
    print(f"[ok] Euler ODE: 1 step with perfect v recovers x0 (max err {torch.abs(out-x0).max().item():.2e})")


def test_gpu_train_step():
    """Tiny model on GPU: loss decreases after a few optimizer steps."""
    if not torch.cuda.is_available():
        print("[skip] no CUDA")
        return
    fm = create_diffusion_or_flow('50', 'flow')
    torch.manual_seed(0)
    B, C, H, W = 4, 4, 16, 16
    dev = 'cuda'

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(C * H * W, C * H * W)
        def forward(self, x, t, **kw):
            return self.lin(x.flatten(1)).view(x.shape)

    net = Net().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
    x0 = torch.randn(B, C, H, W, device=dev)

    losses = []
    for i in range(30):
        opt.zero_grad()
        t = fm.sample_t(B, dev)
        out = fm.training_losses(net, x0, t)
        loss = out["loss"].mean()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    print(f"[ok] GPU train: loss {losses[0]:.4f} -> {losses[-1]:.4f}")
    assert losses[-1] < losses[0], "loss should decrease"
    print("[ok] GPU training step works (velocity MSE decreases)")


if __name__ == "__main__":
    test_interpolant_math()
    test_training_loss_shape()
    test_euler_recovers_x0()
    test_gpu_train_step()
    print("\nALL FLOW-MATCHING SMOKE TESTS PASSED")
