"""Flow Matching diffusion — linear interpolant + Euler ODE sampler.

Drop-in replacement for GaussianDiffusion (same interface surface used by
train.py / in_process_eval.py):

    * ``sample_t(n, device)``    -> uniform t in [0, 1)
    * ``training_losses(model, x_start, t, model_kwargs, noise)``
        -> dict with "loss" (velocity MSE), linear interpolant
    * ``ddim_sample_loop(model, shape, x_T, clip_denoised, model_kwargs, device)``
        -> Euler ODE integration from t=1 down to t=0

Mathematical convention (SD3 / Flux style linear interpolant)
------------------------------------------------------------
    x_t = (1 - t) * x0 + t * noise        t in [0, 1]
    v   = d x_t / d t = noise - x0        (velocity target)

    - t=0 -> data, t=1 -> pure noise.
    - Training regresses the velocity field with plain MSE on the first C
      channels of the model output (DiT with learn_sigma=True emits 2C;
      the sigma channels are dropped — flow matching has no variance head).
    - Sampling integrates the ODE backward: x <- x - dt * v(x_t, t),
      from x_1 = noise down to x_0 (the generated sample).

Time convention
---------------
Internally the velocity network is fed ``t * TIME_SCALE`` so that the
sinusoidal TimestepEmbedder (trained with DDPM t in [0, 999]) sees the same
phase range.  ``TIME_SCALE = 1000.0`` matches the DDPM convention.

Selection
---------
``diffusion.create_diffusion_or_flow(timestep_respacing, diffusion_type, ...)``
returns this class for ``diffusion_type="flow"`` and the classic
``SpacedDiffusion`` (DDPM/DDIM) otherwise. train.py and in_process_eval.py
both go through that factory; no other call site needs changing.

Testing
-------
``tools/test_flow_matching.py`` (interpolant endpoints, loss shape, 1-step
Euler exactness, GPU training step) and ``tools/test_ddpm_regression.py``
(DDPM loss + real DiT forward + flow Euler with real model) cover the
interface; ``tools/build_smoke_data.py`` + ``configs/smoke_flow_test.json``
give a local end-to-end training smoke run.

Reference: Lipman et al. "Flow Matching for Generative Modeling" (2022);
Stable Diffusion 3 / Flux use the same linear interpolant + velocity target.
"""

import torch as th
import torch.nn.functional as F

TIME_SCALE = 1000.0  # t in [0,1] is scaled to [0,1000] before feeding the model


class FlowMatching:
    """Linear-interpolant flow matching with an Euler ODE sampler.

    Parameters
    ----------
    num_steps : int
        Number of Euler integration steps used by ``ddim_sample_loop``.
        (Kept as ``num_timesteps`` for interface parity with GaussianDiffusion.)
    sigma_min : float
        Noise floor at t=0 (unused in the linear interpolant, kept for parity).
    """

    def __init__(self, num_steps=50, sigma_min=1e-4):
        self.num_timesteps = int(num_steps)
        self.sigma_min = sigma_min
        self.is_flow = True

    # ── training ──────────────────────────────────────────────────────────
    def sample_t(self, n, device):
        """Uniform t in [0, 1)."""
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
        model_output = model_output.float()
        if model_output.shape[1] == 2 * v_target.shape[1]:
            model_output = model_output[:, : v_target.shape[1]]

        # velocity MSE (optionally L2-weighted by (1-t); plain MSE is fine)
        loss = F.mse_loss(model_output, v_target.float(), reduction="none")
        loss = loss.mean(dim=list(range(1, loss.ndim)))
        terms["loss"] = loss
        return terms

    # ── sampling (Euler ODE) ──────────────────────────────────────────────
    def ddim_sample_loop(
        self,
        model,
        shape,
        x_T=None,
        clip_denoised=False,
        model_kwargs=None,
        device=None,
        progress=False,
    ):
        """
        Euler ODE integration from t=1 to t=0 with ``num_timesteps`` steps.

        Parameters match GaussianDiffusion.ddim_sample_loop for drop-in use:
            model       : callable model(x_t, t, **model_kwargs)
            shape       : (B, C, H, W) output shape
            x_T         : initial noise (default: randn(shape))
            model_kwargs: dict forwarded to model (may include cfg_scale,
                          handled inside forward_with_cfg)
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
        steps = self.num_timesteps
        dt = 1.0 / steps

        x = x_T
        # integrate t: 1 -> 0
        C = shape[1]
        for i in range(steps):
            t = 1.0 - i * dt
            t_batch = th.full((B,), t, device=device)
            v = model(x, t_batch * TIME_SCALE, **model_kwargs)
            if isinstance(v, tuple):
                v = v[0]
            if v.shape[1] == 2 * C:
                v = v[:, :C]  # learn_sigma DiT: drop the sigma channels
            x = x - dt * v

        return x

    # ── interface parity helpers ──────────────────────────────────────────
    def ddim_sample(self, *args, **kwargs):
        """Alias so generic callers that name the method ``ddim_sample`` work."""
        return self.ddim_sample_loop(*args, **kwargs)


def create_flow_matching(timestep_respacing="50", **kwargs):
    """Build a FlowMatching instance.

    ``timestep_respacing`` may be an int, a str int ("50"), or a comma/space
    separated list (only the count is used here).
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
    return FlowMatching(num_steps=steps, **kwargs)
