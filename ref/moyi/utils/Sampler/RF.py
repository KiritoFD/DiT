"""utils/Sampler/RF.py —— Rectified Flow（条件流匹配）训练器。

## 为什么这样写
原文件在 ref 里缺失（`from utils.Sampler.RF import RF` 找不到）。
从调用点反推接口（`train_moyun2_RF.py`）:
    diffusion = RF(without_t=(args.without_t == 1))
    loss_dict = diffusion.forward(model, x, feature_dict=feature_dict, y=y, stroke=stroke)
    loss = loss_dict[0]

## Rectified Flow 的标准形式
    x_1 = 数据, x_0 ~ N(0, I)
    t ~ U(0, 1)            （或用 logit-normal，见 --t-dist）
    x_t = (1 - t) * x_0 + t * x_1
    v_target = x_1 - x_0
    loss = || v_theta(x_t, t, y) - v_target ||^2

## `without_t` 的语义
原脚本有 `--without-t 0`（默认关）。开启时表示**不给模型 t 条件**
（即模型要自己从 x_t 推断进度）—— 这里按"把 t 置零后传入"实现。
"""
import torch
import torch.nn.functional as F


class RF:
    def __init__(self, without_t=False, t_dist="uniform", t_shift=1.0,
                 noise_scale=1.0, device=None):
        """
        Args:
            without_t: True 时不把 t 传给模型（置 0）
            t_dist:    "uniform" | "logit_normal"
            t_shift:   logit-normal 的 shift（越大越偏向高噪声）
        """
        self.without_t = bool(without_t)
        self.t_dist = t_dist
        self.t_shift = float(t_shift)
        self.noise_scale = float(noise_scale)
        self.device = device

    # ── 采样 t ──────────────────────────────────────────────────────
    def sample_t(self, n, device):
        if self.t_dist == "logit_normal":
            z = torch.randn(n, device=device)
            t = torch.sigmoid(z * self.t_shift)
        else:
            t = torch.rand(n, device=device)
        # 避免 t 恰好 0/1 导致数值问题
        return t.clamp(1e-5, 1 - 1e-5)

    # ── 训练前向 ────────────────────────────────────────────────────
    def forward(self, model, x, feature_dict=None, y=None, stroke=None,
                return_dict=False):
        """
        Args:
            model: DiT（forward(x, t, y, stroke)）
            x:     (B, C, H, W) 目标 latent（12ch = image|edge|skel 拼接）
            feature_dict: moyun 用来做**分维度条件 dropout**的字典
                          {"calligrapher":..., "font":..., "charactor":...,
                           "calligrapher_x":p, "font_x":p, "charactor_x":p}
        Returns:
            (loss,)  —— 与调用点 `loss_dict[0]` 对齐
        """
        B = x.shape[0]
        device = x.device

        t = self.sample_t(B, device)
        x0 = torch.randn_like(x) * self.noise_scale
        t_b = t.view(B, 1, 1, 1)
        x_t = (1 - t_b) * x0 + t_b * x
        v_target = x - x0

        # 分维度条件 dropout（ref 的 _ful.sh: calligrapher-x 0.16, font/char 0.08）
        if y is not None and feature_dict is not None:
            y = self._apply_factor_dropout(y, feature_dict, device)

        t_in = torch.zeros_like(t) if self.without_t else t
        v_pred = model(x_t, t_in, y, stroke)

        loss = F.mse_loss(v_pred.float(), v_target.float())
        return (loss,)

    def _apply_factor_dropout(self, y, feature_dict, device):
        """y 是 (callig, font, char) 的 tuple；按 feature_dict 里的 *_x 概率各自置 null。

        ⚠ 需要模型知道 null 的索引（= num_classes）。这里假设 y 里的值已经
          是 0..num_classes-1，null 用 num_classes 表示 —— 与 moyun 的
          LabelEmbedder（use_cfg_embedding 多一行）一致。
        """
        if not isinstance(y, (tuple, list)):
            return y
        keys = ("calligrapher", "font", "charactor")
        out = []
        for i, yy in enumerate(y):
            k = keys[i] if i < len(keys) else None
            p = float(feature_dict.get(f"{k}_x", 0.0)) if k else 0.0
            if p > 0:
                # num_classes 未知 -> 用 y 的最大值 + 1 近似（训练时 y 已含 null 行）
                null_id = int(feature_dict.get(f"_{k}_num_classes", 0)) or None
                drop = torch.rand(yy.shape[0], device=device) < p
                if null_id is not None:
                    yy = torch.where(drop, torch.full_like(yy, null_id), yy)
            out.append(yy)
        return tuple(out)

    # ── 推理采样（Euler，与训练同一线性路径）────────────────────────
    @torch.no_grad()
    def sample(self, model, shape, y, stroke=None, steps=50, device="cuda",
               cfg_scale=1.0):
        """从 t=0（噪声）积分到 t=1（数据）。返回 (B, C, H, W)。"""
        x = torch.randn(shape, device=device)
        ts = torch.linspace(0, 1, steps + 1, device=device)
        for i in range(steps):
            t = ts[i].expand(shape[0])
            dt = ts[i + 1] - ts[i]
            if cfg_scale > 1.0:
                v = model.forward_with_cfg(x, t, y, stroke, cfg_scale)
            else:
                v = model(x, t, y, stroke)
            x = x + v * dt
        return x
