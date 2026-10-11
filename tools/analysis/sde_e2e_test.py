# -*- coding: utf-8 -*-
"""sde_e2e_test.py — 端到端单测：用**完美的线性速度场**跑真实采样代码，验证终点方差

为什么需要它: sde_oracle_test3 验证的是**数学**（方差 ODE）；本测试验证的是**代码路径**
（FlowMatching.ddim_sample_loop + SDE 分支），可捕捉 t_i/t_next 映射、噪声标度、Heun 平均等实现错误。

构造: x0~N(0,s0^2), eps~N(0,1)  =>  x_t=(1-t)x0+t eps, D(t)=(1-t)^2 s0^2+t^2
      完美速度场 v(x,t) = c(t) x,  c(t)=(t-(1-t)s0^2)/D(t)
      => 从 x_T~N(0,I) 出发积分到 t=0, 终点方差应为 s0^2
"""
import os
import sys

import torch as th

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.loss.flow_matching import FlowMatching  # noqa: E402

NS = 20_000   # 方差估计 SE≈sqrt(2/N)≈1.4%, 足以识别 25% 级别的偏差


class LinearField(th.nn.Module):
    """完美速度场 v(x,t) = c(t) x（c 依赖 s0）。

    ⚠ 采样器把 t 乘以 TIME_SCALE(=1000) 后才喂给模型（`_v` 内 `t_batch * TIME_SCALE`），
      所以这里必须先除回去得到归一化时间 u∈[0,1] —— 首次实现漏了这一步, 导致假模型速度≈0,
      测出来与 s0 无关（这正是端到端单测要抓的实现细节）。
    """

    def __init__(self, s0):
        super().__init__()
        self.s0 = float(s0)

    def forward(self, x, t, **kw):
        u = t / 1000.0                                   # 还原归一化时间
        D = (1.0 - u) ** 2 * self.s0 ** 2 + u ** 2
        c = (u - (1.0 - u) * self.s0 ** 2) / D
        return c.view(-1, *([1] * (x.dim() - 1))) * x


def run(s0, gamma, steps=50, seed=0):
    th.manual_seed(seed)
    fm = FlowMatching(num_steps=steps, sampler="heun", sde_gamma=gamma)
    model = LinearField(s0)
    shape = (NS, 4, 32, 32)
    x_T = th.randn(*shape)
    with th.no_grad():
        out = fm.ddim_sample_loop(model, shape, x_T=x_T, clip_denoised=False,
                                  model_kwargs={}, device=th.device("cpu"))
    return float(out.var())


print(f"  端到端（真实采样代码），N={NS:,} 样本, 50 步 Heun")
print(f"  {'s0':>5} | {'γ':>5} | {'理论 s0²':>9} | {'实测终点方差':>12} | 判定")
for s0 in (1.0, 0.2):
    for g in (0.0, 0.5):
        th_v = s0 ** 2
        got = run(s0, g)
        rel = abs(got - th_v) / max(th_v, 1e-9)
        print(f"  {s0:>5.2f} | {g:>5.2f} | {th_v:>9.4f} | {got:>12.4f} | {'✓' if rel < 0.05 else '✗ 偏差 %.1f%%' % (rel*100)}")
print("""
  判读: γ=0 必须精确（纯 ODE, Heun 已收敛）；γ=0.5 若也在 5% 内
        -> **修正后的 SDE 代码路径保持边缘分布** ✓（旧实现在 s0=1.0/γ=0.5 会掉到 ≈0.75）
""")
