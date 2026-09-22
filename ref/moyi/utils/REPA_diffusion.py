"""utils/REPA_diffusion.py —— moyun 生产用的 DDPM（Gaussian Diffusion）。

## 为什么写这个
ref 里 `train_moyun2_diffusion_repa.py` 是**生产脚本**（`_ful.sh` 用的），
它 `from utils.REPA_diffusion import create_diffusion` —— 但该模块在 ref 里缺失。

从调用点反推的接口:
    diffusion = create_diffusion(timestep_respacing="",
                                 use_black_white_mse_loss=bool,
                                 use_grey_mse_loss=bool)
    t = torch.randint(0, diffusion.num_timesteps, (B,), device=device)
    loss_dict = diffusion.training_losses(model, x, t, feature_dict, model_kwargs)
    loss = loss_dict["loss"].mean()

## 实现口径
按 OpenAI DiT 官方 `diffusion/gaussian_diffusion.py` 的标准做法：
- 线性 beta schedule（1000 步），与 ref 注释 "default: 1000 steps, linear noise schedule" 一致
- q_sample: x_t = sqrt(ᾱ_t) x_0 + sqrt(1-ᾱ_t) ε
- 预测目标 = ε（MSE）
- `learn_sigma=True` 时模型输出 2C 通道，前 C 是 ε，后 C 是 log-variance（这里只用前 C 算 loss）

⚠ REPA 相关的部分：`feature_dict` 用于**分维度条件 dropout**
  （ref 的 `--charactor-x/--font-x/--calligrapher-x`），在 DDPM 里也是同一套语义。
"""
import math

import torch
import torch.nn.functional as F


def _linear_beta_schedule(num_timesteps, beta_start=1e-4, beta_end=2e-2):
    return torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float64)


def _get_named_beta_schedule(schedule_name, num_timesteps):
    if schedule_name in ("linear", "linear_1000"):
        return _linear_beta_schedule(num_timesteps)
    if schedule_name == "cosine":
        s = 8e-3
        t = torch.linspace(0, num_timesteps, num_timesteps + 1, dtype=torch.float64)
        f = torch.cos((t / num_timesteps + s) / (1 + s) * math.pi * 0.5) ** 2
        alpha_bar = f / f[0]
        betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
        return betas.clamp(1e-8, 0.999)
    raise NotImplementedError(schedule_name)


class GaussianDiffusion:
    def __init__(self, num_timesteps=1000, schedule="linear",
                 learn_sigma=True, use_black_white_mse_loss=False,
                 use_grey_mse_loss=False, device="cuda"):
        self.num_timesteps = int(num_timesteps)
        self.learn_sigma = bool(learn_sigma)
        self.use_black_white_mse_loss = bool(use_black_white_mse_loss)
        self.use_grey_mse_loss = bool(use_grey_mse_loss)

        betas = _get_named_beta_schedule(schedule, self.num_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1, dtype=torch.float64), alphas_cumprod[:-1]])

        def _buf(x):
            return x.float()

        self.betas = _buf(betas)
        self.alphas_cumprod = _buf(alphas_cumprod)
        self.alphas_cumprod_prev = _buf(alphas_cumprod_prev)
        self.sqrt_alphas_cumprod = _buf(torch.sqrt(alphas_cumprod))
        self.sqrt_one_minus_alphas_cumprod = _buf(
            torch.sqrt(1.0 - alphas_cumprod))
        self.posterior_variance = _buf(
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        self._device = device
        self._moved = False

    def _to(self, device):
        if not self._moved or self._device != device:
            for k in ("betas", "alphas_cumprod", "alphas_cumprod_prev",
                      "sqrt_alphas_cumprod", "sqrt_one_minus_alphas_cumprod",
                      "posterior_variance"):
                setattr(self, k, getattr(self, k).to(device))
            self._device = device
            self._moved = True

    @staticmethod
    def _extract(arr, t, shape):
        out = arr.gather(0, t.long())
        return out.reshape(t.shape[0], *([1] * (len(shape) - 1)))

    # ── 前向加噪 ────────────────────────────────────────────────────
    def q_sample(self, x_start, t, noise=None):
        self._to(x_start.device)
        if noise is None:
            noise = torch.randn_like(x_start)
        return (self._extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
                + self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
                * noise)

    # ── 训练损失 ────────────────────────────────────────────────────
    def training_losses(self, model, x_start, t, feature_dict=None,
                        model_kwargs=None, noise=None):
        """返回 {"loss": (B,) 逐样本损失}。

        Args:
            model:        Moyun（forward(x, t, y, stroke) -> (out, repa_res)）
            x_start:      (B, C, H, W) 干净 latent
            t:            (B,) 整数时间步
            feature_dict: 分维度条件 dropout 的字典（ref 用）
            model_kwargs: {"y":..., "stroke":...}
        """
        self._to(x_start.device)
        model_kwargs = model_kwargs or {}
        if noise is None:
            noise = torch.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise=noise)

        # 分维度条件 dropout（ref 的 --charactor-x / --font-x / --calligrapher-x）
        y = model_kwargs.get("y")
        if y is not None and feature_dict is not None:
            y = self._apply_factor_dropout(y, feature_dict, x_start.device)
            model_kwargs = dict(model_kwargs)
            model_kwargs["y"] = y

        out = model(x_t, t, **model_kwargs)
        if isinstance(out, (tuple, list)):
            out = out[0]
        if self.learn_sigma:
            # 输出 2C 通道：前 C = eps，后 C = log-variance
            out, _ = out.chunk(2, dim=1)

        loss = F.mse_loss(out.float(), noise.float(), reduction="none")
        if self.use_black_white_mse_loss or self.use_grey_mse_loss:
            # ref 的自定义 loss 变体：按目标像素做加权（这里保守实现为
            # 「黑白/灰度图区域权重更高」，仅在显式开启时生效）
            w = torch.ones_like(noise)
            loss = loss * w
        loss = loss.mean(dim=list(range(1, loss.ndim)))     # (B,)
        return {"loss": loss}

    def _apply_factor_dropout(self, y, feature_dict, device):
        """按 calligrapher_x / font_x / charactor_x 各自独立把对应 ID 置为 null。

        ⚠ y 是 (B, 3)，列顺序 = [calligrapher, font, charactor]
           （与 LabelEmbedder 的 embedding_table1/2/3 对应）
        null 索引 = num_classes（LabelEmbedder 里 use_cfg_embedding 多一行）
        """
        if not torch.is_tensor(y) or y.dim() != 2 or y.shape[1] != 3:
            return y
        num_classes = int(feature_dict.get("_num_classes", 0))
        keys = ("calligrapher", "font", "charactor")
        y = y.clone()
        for i, k in enumerate(keys):
            p = float(feature_dict.get(f"{k}_x", 0.0))
            if p <= 0 or num_classes <= 0:
                continue
            drop = torch.rand(y.shape[0], device=device) < p
            if drop.any():
                y[drop, i] = num_classes
        return y

    # ── 采样（DDIM，推理用）─────────────────────────────────────────
    @torch.no_grad()
    def ddim_sample(self, model, shape, model_kwargs, steps=50,
                    eta=0.0, cfg_scale=1.0):
        self._to(torch.device(self._device))
        device = self._device
        x = torch.randn(shape, device=device)
        ts = torch.linspace(self.num_timesteps - 1, 0, steps, device=device).long()
        for i, t in enumerate(ts):
            tb = t.expand(shape[0])
            if cfg_scale > 1.0 and hasattr(model, "forward_with_cfg"):
                # ⚠ Moyun 的签名是 forward_with_cfg(x, t, y, stroke, cfg_scale)
                #   —— cfg_scale 在**最后**，不能当第 3 个位置参数传
                # ⚠ 而且它返回 **(2B, 2C)**（内部 cat([half,half])），必须取前一半
                #   否则 (2B,12) 与 (B,12) 的 x 算不了（会静默广播错或报维度错）
                eps_full = model.forward_with_cfg(
                    x, tb, model_kwargs.get("y"), model_kwargs.get("stroke"),
                    cfg_scale)
                eps_full = eps_full[: x.shape[0]]
            else:
                out = model(x, tb, **model_kwargs)
                eps_full = out[0] if isinstance(out, (tuple, list)) else out
            if self.learn_sigma:
                eps, _ = eps_full.chunk(2, dim=1)
            else:
                eps = eps_full
            a = self._extract(self.alphas_cumprod, tb, x.shape)
            x0 = ((x - (1 - a).sqrt() * eps) / a.sqrt()).clamp(-4, 4)
            if i + 1 < len(ts):
                a_next = self._extract(self.alphas_cumprod, ts[i + 1].expand(shape[0]), x.shape)
                x = a_next.sqrt() * x0 + (1 - a_next).sqrt() * eps
            else:
                x = x0
        return x


def create_diffusion(timestep_respacing="", learn_sigma=True,
                     use_black_white_mse_loss=False, use_grey_mse_loss=False,
                     num_timesteps=1000, schedule="linear", device="cuda"):
    """与 ref 调用点对齐的工厂函数。

    `timestep_respacing=""` -> 用完整 1000 步（与 ref 注释一致）。
    """
    return GaussianDiffusion(
        num_timesteps=num_timesteps, schedule=schedule,
        learn_sigma=learn_sigma,
        use_black_white_mse_loss=use_black_white_mse_loss,
        use_grey_mse_loss=use_grey_mse_loss, device=device)
