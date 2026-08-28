"""Flow Matching — linear interpolant + 可配置 ODE 求解器（Euler / Heun-RK2）。

Drop-in replacement for GaussianDiffusion (same interface surface used by
train.py / in_process_eval.py / inference.py):

    * ``sample_t(n, device)``    -> t in [0, 1)，采样分布可配置
    * ``training_losses(model, x_start, t, model_kwargs, noise)``
        -> dict with "loss" (velocity MSE), linear interpolant
    * ``ddim_sample_loop(model, shape, x_T, clip_denoised, model_kwargs, device)``
        -> ODE 积分 from t=1 down to t=0

Mathematical convention (SD3 / Flux style linear interpolant)
------------------------------------------------------------
    x_t = (1 - t) * x0 + t * noise        t in [0, 1]
    v   = d x_t / d t = noise - x0        (velocity target)

    - t=0 -> data, t=1 -> pure noise.
    - Training regresses the velocity field with plain MSE on the first C
      channels of the model output (DiT with learn_sigma=True emits 2C;
      the sigma channels are dropped — flow matching has no variance head).
    - Sampling integrates the ODE backward from x_1 = noise down to x_0.

Time convention
---------------
Internally the velocity network is fed ``t * TIME_SCALE`` so that the
sinusoidal TimestepEmbedder (trained with DDPM t in [0, 999]) sees the same
phase range.  ``TIME_SCALE = 1000.0`` matches the DDPM convention.

本次改造（flow free-lunch）
--------------------------
1) **t 采样分布** ``t_sampler``
   - ``uniform``       : 旧行为，U(0,1)
   - ``logit_normal``  : t = sigmoid(N(mean, std²))，把训练密度集中在中段。
     SD3 的核心发现之一 —— 均匀 t 在两端浪费了大量梯度预算，而两端的
     velocity 目标（t→0: v≈-x0；t→1: v≈ε）信息量都很低。默认开启。
   - ``cosmap``        : SD3 论文的另一选项，t = 1 - 1/(tan(πu/2)+1), u~U(0,1)

2) **求解器** ``sampler``
   - ``euler`` : 一阶，每步 1 次 NFE（旧行为）
   - ``heun``  : 二阶 Heun / RK2（trapezoidal），每步 2 次 NFE。
     **同 NFE 预算下严格优于一阶**：Heun@25 步 ≈ Euler@50 步的算力，
     但截断误差从 O(dt) 降到 O(dt²)。默认开启。
   - ``heun_batch=True`` 时，Heun 的两次评估沿 batch 维拼成一次 forward，
     显著改善 GPU 利用率（kernel launch / 显存带宽 amortize）。

3) **timestep shift** ``shift``
   采样端把均匀网格 s∈[0,1] 映射为 t = shift·s / (1 + (shift-1)·s)（SD3）。
   - shift > 1 → 步数向高噪声端（t→1）集中，适合高分辨率（布局先定）
   - shift = 1 → 退化为均匀（默认）
   - shift < 1 → 步数向低噪声端（t→0）集中，适合**细节/纹理主导**的任务

   本项目是 32×32 latent（256px 字形），笔画末端、飞白等细节在 t→0 形成，
   因此默认 shift=1.0（不做 shift）；若要尝试请把值显式写进 config 并记录，
   便于和 logit-normal 一起做 ablation。

NFE 预算
--------
``nfe`` 属性给出一次 ``ddim_sample_loop`` 的网络评估次数：
    euler: steps      heun: 2 * steps
切换求解器时请同步调整 config 里的 eval/train 步数以保持等算力对比。

Testing
-------
``tools/test_flow_matching.py`` (interpolant endpoints, loss shape, 1-step
Euler exactness, GPU training step) 和 ``tools/test_ddpm_regression.py``
(DDPM loss + real DiT forward + flow Euler with real model) 覆盖接口；
``tools/build_smoke_data.py`` + ``configs/smoke_flow_test.json`` 提供本地
端到端训练 smoke。

Reference: Lipman et al. "Flow Matching for Generative Modeling" (2022);
Esser et al. "Scaling Rectified Flow Transformers for High-Resolution Image
Synthesis" (SD3, 2024) —— logit-normal t 采样 + timestep shift。
"""

import math

import torch as th
import torch.nn.functional as F

TIME_SCALE = 1000.0  # t in [0,1] is scaled to [0,1000] before feeding the model


class FlowMatching:
    """Linear-interpolant flow matching，可配置 t 分布 / 求解器 / schedule。

    Parameters
    ----------
    num_steps : int
        ODE 积分步数（Heun 时 NFE = 2 * num_steps）。
        (Kept as ``num_timesteps`` for interface parity with GaussianDiffusion.)
    sigma_min : float
        Noise floor at t=0 (unused in the linear interpolant, kept for parity).
    use_ot : bool
        Minibatch Optimal Transport 重排（OT-CFM）。
    t_sampler : str
        "uniform" | "logit_normal" | "cosmap"
    t_mean, t_std : float
        logit_normal 的参数（SD3 用 0.0 / 1.0）。
    shift : float
        采样端 timestep shift，1.0 = 不 shift。
    sampler : str
        "euler" | "heun"
    heun_batch : bool
        Heun 的两次评估是否拼成一个 batch 做单次 forward。
    """

    def __init__(self, num_steps=50, sigma_min=1e-4, use_ot=False,
                 t_sampler="logit_normal", t_mean=0.0, t_std=1.0,
                 shift=1.0, sampler="heun", heun_batch=True):
        self.num_timesteps = int(num_steps)
        self.sigma_min = sigma_min
        self.use_ot = bool(use_ot)
        self.is_flow = True

        self.t_sampler = str(t_sampler).lower()
        if self.t_sampler not in ("uniform", "logit_normal", "cosmap"):
            raise ValueError(f"Unknown t_sampler={t_sampler!r}")
        self.t_mean = float(t_mean)
        self.t_std = float(t_std)

        self.shift = float(shift)
        if self.shift <= 0:
            raise ValueError(f"shift must be > 0, got {self.shift}")

        self.sampler = str(sampler).lower()
        if self.sampler not in ("euler", "heun"):
            raise ValueError(f"Unknown sampler={sampler!r} (expected 'euler' or 'heun')")
        self.heun_batch = bool(heun_batch)

    # ── introspection ───────────────────────────────────────────────────
    @property
    def nfe(self):
        """一次采样循环的网络前向次数（per sample）。"""
        return self.num_timesteps * (2 if self.sampler == "heun" else 1)

    def describe(self):
        return (f"FlowMatching(steps={self.num_timesteps}, sampler={self.sampler}, "
                f"nfe={self.nfe}, t_sampler={self.t_sampler}"
                + (f"(mean={self.t_mean},std={self.t_std})" if self.t_sampler == "logit_normal" else "")
                + f", shift={self.shift}, use_ot={self.use_ot})")

    # ── training ────────────────────────────────────────────────────────
    def sample_t(self, n, device):
        """按 ``t_sampler`` 采样 t ∈ (0, 1)。"""
        if self.t_sampler == "logit_normal":
            u = th.randn(n, device=device) * self.t_std + self.t_mean
            return th.sigmoid(u)
        if self.t_sampler == "cosmap":
            # SD3: t = 1 - 1 / (tan(pi/2 * u) + 1),  u ~ U(0, 1)
            u = th.rand(n, device=device).clamp_(1e-6, 1.0 - 1e-6)
            return 1.0 - 1.0 / (th.tan(u * math.pi / 2.0) + 1.0)
        return th.rand(n, device=device)

    def _interp(self, x_start, noise, t):
        """Linear interpolant: x_t = (1-t) * x_start + t * noise.  t: [N] float.

        Convention: t=0 -> data, t=1 -> pure noise (SD3/Flux linear interpolant).
        """
        return (1.0 - t[:, None, None, None]) * x_start + t[:, None, None, None] * noise

    def training_losses(self, model, x_start, t, model_kwargs=None, noise=None):
        """
        Velocity-matching loss on the linear interpolant path.

        x_t = (1 - t) * x_start + t * noise
        v   = noise - x_start        (target velocity: d x_t / d t)

        Returns ``{"loss": [N] per-sample MSE, "intermediate_feats": ...}``.
        """
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = th.randn_like(x_start)

        # ── Minibatch Optimal Transport (OT-CFM, 可配置) ─────────────────
        # 对每个 batch, 用匈牙利算法在噪声/数据 pair 上做最优重排, 使轨迹
        # 不再交叉、速度场更平滑 (Tong et al., TMLR 2024).
        # 代价: O(B^3) 匈牙利 + 一次 GPU->CPU 同步.
        if self.use_ot and x_start.shape[0] > 1:
            from scipy.optimize import linear_sum_assignment
            with th.no_grad():
                x_flat = x_start.reshape(x_start.shape[0], -1).float()
                n_flat = noise.reshape(noise.shape[0], -1).float()
                cost = th.cdist(x_flat, n_flat, p=2).pow(2)
                row_ind, col_ind = linear_sum_assignment(cost.cpu().numpy())
                noise = noise[th.from_numpy(col_ind).to(noise.device)]

        t = t.float()
        x_t = self._interp(x_start, noise, t)
        v_target = noise - x_start

        model_output = model(x_t, t * TIME_SCALE, **model_kwargs)

        terms = {}
        if isinstance(model_output, tuple):
            model_output, intermediate_feats = model_output
            terms["intermediate_feats"] = intermediate_feats

        # DiT with learn_sigma=True outputs 2C channels (mean + learned sigma).
        # Flow matching has no variance prediction: take the first C channels
        # as the velocity field.
        # NOTE: 建议训练侧直接传 learn_sigma=False，这样不会有永不收梯度的死通道。
        model_output = model_output.float()
        if model_output.shape[1] == 2 * v_target.shape[1]:
            model_output = model_output[:, : v_target.shape[1]]

        # velocity MSE (optionally L2-weighted by (1-t); plain MSE is fine)
        loss = F.mse_loss(model_output, v_target.float(), reduction="none")
        loss = loss.mean(dim=list(range(1, loss.ndim)))
        terms["loss"] = loss
        return terms

    # ── sampling schedule ───────────────────────────────────────────────
    def _schedule(self, steps, device, dtype=th.float32):
        """返回 (steps+1,) 的时间表，从 t=1 单调降到 t=0。

        s 为 [1, 0] 上均匀；shift 映射为
            t = shift * s / (1 + (shift - 1) * s)
        shift=1 时即退化为线性均匀表。
        """
        s = th.linspace(1.0, 0.0, steps + 1, device=device, dtype=dtype)
        if self.shift == 1.0:
            return s
        return self.shift * s / (1.0 + (self.shift - 1.0) * s)

    def _v(self, model, x, t_batch, model_kwargs, C):
        """一次速度场评估（含 learn_sigma 通道裁剪 / tuple 解包）。"""
        v = model(x, t_batch * TIME_SCALE, **model_kwargs)
        if isinstance(v, tuple):
            v = v[0]
        if v.shape[1] == 2 * C:
            v = v[:, :C]  # learn_sigma DiT: drop the sigma channels
        return v

    @staticmethod
    def _tile_kwargs(model_kwargs, B):
        """把 model_kwargs 里所有 batch 维 == B 的张量沿 batch 维复制一份。

        Heun 的 batched 模式会把两个 RK stage 沿 batch 维拼成 (2B, ...)，
        但 ``model_kwargs`` 里的 ``y_callig`` / ``y_char`` / ``cond`` / ``g``
        仍是 B。而 ``forward_with_cfg`` / ``ControlNetDiT.forward_with_cfg``
        都假定 ``x.shape[0] == y.shape[0]``（内部各自再 cat 一次），
        不扩展就会在 ``c = t_emb + y_emb`` 处 batch 维不匹配。
        """
        if not model_kwargs:
            return {}
        out = {}
        for k, v in model_kwargs.items():
            if th.is_tensor(v) and v.ndim > 0 and v.shape[0] == B:
                out[k] = th.cat([v, v], dim=0)
            else:
                out[k] = v          # 标量（cfg_scale 等）/ None 原样透传
        return out

    def ddim_sample_loop(
        self,
        model,
        shape,
        x_T=None,
        clip_denoised=False,
        model_kwargs=None,
        device=None,
        progress=False,
        denoised_fn=None,
        eta=0.0,
    ):
        """
        ODE 积分 from t=1 to t=0。

        Parameters match GaussianDiffusion.ddim_sample_loop for drop-in use:
            model       : callable model(x_t, t, **model_kwargs)
            shape       : (B, C, H, W) output shape
            x_T         : initial noise (default: randn(shape))
            model_kwargs: dict forwarded to model (may include cfg_scale,
                          handled inside forward_with_cfg)

        注：flow matching 没有 denoised/clip/eta 的概念，``clip_denoised`` /
        ``denoised_fn`` / ``eta`` 仅为接口兼容而被**忽略**（旧实现同样忽略）。
        """
        if model_kwargs is None:
            model_kwargs = {}
        if device is None:
            device = next(model.parameters()).device
        if x_T is None:
            x_T = th.randn(*shape, device=device)
        else:
            x_T = x_T.to(device)

        B = shape[0]
        C = shape[1]
        steps = self.num_timesteps
        ts = self._schedule(steps, device)

        x = x_T
        use_heun = (self.sampler == "heun")

        for i in range(steps):
            t_i = ts[i]
            t_next = ts[i + 1]
            dt = (t_next - t_i)                      # 负数：从 t=1 走向 t=0

            if use_heun and self.heun_batch:
                # 把两个 RK stage 沿 batch 维拼成一次 forward（更好摊销 kernel
                # launch 与显存带宽）。model 对 batch 维无状态，因此语义等价。
                # 关键：model_kwargs 里的 y_callig/y_char/cond/g 也必须一起复制，
                # 否则 CFG wrapper 的 "x.shape[0] == y.shape[0]" 假设会被打破。
                t_batch = th.full((B,), float(t_i), device=device)
                v1 = self._v(model, x, t_batch, model_kwargs, C)
                x_euler = x + dt * v1
                t2 = th.full((B,), float(t_next), device=device)
                x_cat = th.cat([x, x_euler], dim=0)
                t_cat = th.cat([t_batch, t2], dim=0)
                kw_cat = self._tile_kwargs(model_kwargs, B)
                v_cat = self._v(model, x_cat, t_cat, kw_cat, C)
                v1_, v2 = th.split(v_cat, B, dim=0)
                x = x + dt * 0.5 * (v1_ + v2)
            elif use_heun:
                t_batch = th.full((B,), float(t_i), device=device)
                v1 = self._v(model, x, t_batch, model_kwargs, C)
                x_euler = x + dt * v1
                t2 = th.full((B,), float(t_next), device=device)
                v2 = self._v(model, x_euler, t2, model_kwargs, C)
                x = x + dt * 0.5 * (v1 + v2)
            else:
                t_batch = th.full((B,), float(t_i), device=device)
                v = self._v(model, x, t_batch, model_kwargs, C)
                x = x + dt * v

        return x

    # ── interface parity helpers ────────────────────────────────────────
    def ddim_sample(self, *args, **kwargs):
        """Alias so generic callers that name the method ``ddim_sample`` work."""
        return self.ddim_sample_loop(*args, **kwargs)

    def p_sample_loop(self, *args, **kwargs):
        """Alias for GaussianDiffusion.p_sample_loop parity."""
        return self.ddim_sample_loop(*args, **kwargs)

    def sample(self, *args, **kwargs):
        """Alias（flow 语境下更自然的命名）。"""
        return self.ddim_sample_loop(*args, **kwargs)


#: FlowMatching 构造函数接受的所有参数名。
#: 用于从 argparse Namespace / config dict 里安全地筛选，避免把 ddpm 专属参数
#: （noise_schedule / learn_sigma / use_kl ...）误传进来导致 TypeError。
FLOW_PARAMS = (
    "num_steps", "sigma_min", "use_ot",
    "t_sampler", "t_mean", "t_std",
    "shift", "sampler", "heun_batch",
)

#: CLI / config 里的别名 -> FlowMatching 参数名。
#: ``flow_sampler`` 的存在是因为 argparse 的 ``--sampler`` 已被数据采样器
#: (random|factor_balanced) 占用，同名 dest 会互相覆盖。
FLOW_PARAM_ALIASES = {"flow_sampler": "sampler"}


def _resolve_flow_aliases(kwargs):
    """把别名键改写成正式的 FlowMatching 参数名（原地）。"""
    for alias, real in FLOW_PARAM_ALIASES.items():
        if alias in kwargs:
            v = kwargs.pop(alias)
            if v is not None and real not in kwargs:
                kwargs[real] = v
    return kwargs


def create_flow_matching(timestep_respacing="50", use_ot=False, **kwargs):
    """Build a FlowMatching instance.

    ``timestep_respacing`` may be an int, a str int ("50"), or a comma/space
    separated list (only the count is used here).

    额外 kwargs 透传给 FlowMatching:
        t_sampler, t_mean, t_std, shift, sampler, heun_batch

    未知 kwargs 会被丢弃并打 warning（而不是 TypeError）—— 这样
    ``create_diffusion_or_flow`` 可以无脑把整包训练配置转发过来。
    """
    if isinstance(timestep_respacing, str):
        timestep_respacing = timestep_respacing.strip()
        if timestep_respacing and timestep_respacing not in ("", "None"):
            parts = [p for p in timestep_respacing.replace(",", " ").split() if p]
            if len(parts) == 1:
                timestep_respacing = int(parts[0])
    if isinstance(timestep_respacing, (list, tuple)):
        timestep_respacing = len(timestep_respacing)
    steps = int(timestep_respacing) if timestep_respacing else 50

    kwargs = _resolve_flow_aliases(kwargs)
    kw = {k: v for k, v in kwargs.items() if k in FLOW_PARAMS and v is not None}
    dropped = sorted(k for k, v in kwargs.items()
                     if k not in FLOW_PARAMS and v is not None)
    if dropped:
        import logging
        logging.getLogger(__name__).warning(
            f"[create_flow_matching] ignoring non-flow kwargs: {dropped}")
    kw.setdefault("use_ot", use_ot)
    return FlowMatching(num_steps=steps, **kw)
