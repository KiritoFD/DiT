# -*- coding: utf-8 -*-
"""验证 pred_xstart 修复：flow 下能否返回、数值是否正确、是否带梯度。"""
import os, sys, torch
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.loss.flow_matching import FlowMatching

fm = FlowMatching(num_steps=25, t_sampler="logit_normal", t_mean=0.0,
                  t_std=1.0, shift=1.0, sampler="heun", heun_batch=True)


class M(torch.nn.Module):
    """可控替身：输出 = v_pred。用 scale 参数化，便于验证反推公式。

    注意：flow matching 的模型输出是**速度 v**，不是 x_t。
    早先版本让替身返回 x*w（那是 x_t 而非 v），导致反推结果与解析式
    对不上 —— 是测试替身写错了，不是实现有问题。
    """
    def __init__(self, scale=0.0):
        super().__init__()
        self.w = torch.nn.Parameter(torch.tensor(float(scale)))

    def forward(self, x, t, **kw):
        # 输出 = w * ones_like(x)，即一个可控的常数速度场
        return torch.ones_like(x) * self.w


torch.manual_seed(0)
m = M()
B, C, H, W = 4, 4, 32, 32
x0 = torch.randn(B, C, H, W)
t = fm.sample_t(B, torch.device("cpu"))
noise = torch.randn_like(x0)
x_t = fm._interp(x0, noise, t)

print("=" * 70)
print("1) 默认不返回（零开销）")
print("=" * 70)
terms = fm.training_losses(m, x0, t, noise=noise)
print(f"  keys = {sorted(terms.keys())}   -> 与修复前一致，无额外开销")

print("\n" + "=" * 70)
print("2) return_pred_xstart=True")
print("=" * 70)
terms = fm.training_losses(m, x0, t, noise=noise, return_pred_xstart=True)
print(f"  keys = {sorted(terms.keys())}")
px = terms["pred_xstart"]
print(f"  pred_xstart shape = {tuple(px.shape)}")

print("\n" + "=" * 70)
print("3) 数值正确性")
print("=" * 70)
# (a) 反推公式本身：直线路径下 x0 = x_t - t*v 是恒等式
x0_manual = x_t - t.reshape(-1, 1, 1, 1) * (noise - x0)
print(f"  (a) 恒等式 x_t - t*(noise-x0) == x0，误差: "
      f"{(x0_manual - x0).abs().max().item():.3e}")

# (b) 实现：替身输出常数 v_pred = w*1，则 pred_xstart 应 == x_t - t*w
m2 = M(scale=0.7)
terms2 = fm.training_losses(m2, x0, t, noise=noise, return_pred_xstart=True)
px2 = terms2["pred_xstart"]
expect = x_t - t.reshape(-1, 1, 1, 1) * 0.7
print(f"  (b) 替身输出常速度 0.7 时，pred_xstart vs (x_t - t*0.7) 误差: "
      f"{(px2 - expect).abs().max().item():.3e}")

# (c) 若模型恰好预测出真实速度，则 pred_xstart 应精确还原 x0
class TrueV(torch.nn.Module):
    def __init__(self, v):
        super().__init__()
        self.v = v
    def forward(self, x, t, **kw):
        return self.v


terms3 = fm.training_losses(TrueV(noise - x0), x0, t, noise=noise,
                            return_pred_xstart=True)
print(f"  (c) 模型输出真实速度时，pred_xstart vs 真实 x0 误差: "
      f"{(terms3['pred_xstart'] - x0).abs().max().item():.3e}  <- 关键")

print("\n" + "=" * 70)
print("4) 梯度是否回传到模型参数（w_std_mid 依赖这点）")
print("=" * 70)
loss = terms["loss"].mean() + 0.5 * terms["pred_xstart"].pow(2).mean()
m.zero_grad(set_to_none=True)
loss.backward()
print(f"  d(loss)/d(model.w) = {m.w.grad.item():.6e}")
print(f"  pred_xstart.requires_grad = {px.requires_grad}")
if m.w.grad is not None and m.w.grad.abs().item() > 0:
    print("  OK: pred_xstart 参与计算图，梯度可回传到 DiT")
else:
    print("  FAIL: 梯度未回传")

print("\n" + "=" * 70)
print("5) 结论")
print("=" * 70)
print("  修复前: flow 模式 loss_dict 只有 ['loss']，")
print("          train.py 的 loss_dict.get('pred_xstart', None) 恒为 None，")
print("          导致 w_std_mid / w_latent_skel / w_latent_canny / latent_struct")
print("          全部静默失效（不报错、loss 正常下降、但从未生效）。")
print("  修复后: 仅在需要时返回，数值正确且带梯度。")
