"""Low-memory structural supervision directly in cached VAE-latent space."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ResidualConv(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x):
        return x + self.net(x)


class LatentStructureProbe(nn.Module):
    """Small 32x32 latent-to-(Canny, skeleton) predictor.

    Train this separately on GT latents and freeze it before using its skeleton
    logits as a loss.  Keeping it frozen prevents the probe from adapting to a
    poor DiT prediction and makes the structural signal identifiable.
    """

    def __init__(self, in_channels=4, width=32, depth=2):
        super().__init__()
        self.stem = nn.Conv2d(in_channels, width, 3, padding=1)
        self.blocks = nn.Sequential(*[_ResidualConv(width) for _ in range(depth)])
        self.head = nn.Sequential(
            nn.GroupNorm(8, width), nn.SiLU(), nn.Conv2d(width, 2, 1))

    def forward(self, x):
        return self.head(self.blocks(self.stem(x)))


def downsample_structure(target, size):
    """Keep thin positive lines when mapping a 256x256 target to latent size."""
    if target is None or target.numel() == 0:
        return None
    return F.adaptive_max_pool2d(target.float(), size)


def _edge_weighted_gradient_loss(pred, target, edge_map, eps=1e-3):
    pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    true_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
    pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    true_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
    wx = torch.maximum(edge_map[:, :, :, 1:], edge_map[:, :, :, :-1])
    wy = torch.maximum(edge_map[:, :, 1:, :], edge_map[:, :, :-1, :])
    ex = ((pred_dx - true_dx).square() + eps * eps).sqrt().mean(dim=1, keepdim=True)
    ey = ((pred_dy - true_dy).square() + eps * eps).sqrt().mean(dim=1, keepdim=True)
    lx = (ex * wx).sum() / wx.sum().clamp_min(1.0)
    ly = (ey * wy).sum() / wy.sum().clamp_min(1.0)
    return 0.5 * (lx + ly)


def _balanced_bce_dice(logits, target, eps=1e-6):
    positives = target.sum()
    negatives = target.numel() - positives
    pos_weight = (negatives / positives.clamp_min(1.0)).clamp(1.0, 10.0)
    bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)
    probs = logits.sigmoid()
    dice = 1.0 - ((2.0 * (probs * target).sum() + eps)
                  / (probs.sum() + target.sum() + eps))
    return bce + dice


class LatentSkelProbe(nn.Module):
    """4ch image latent -> 4ch **instance skeleton** latent. 训练完冻结后用作结构 loss。

    与 LatentStructureProbe 的区别:
      - LatentStructureProbe 输出 2ch (canny/skel 二值 logits), 需要 **像素级** GT 骨架图
        做 target —— 而当前数据集 `canny`/`skeleton` 恒为 `torch.empty(0)`(latent_dataset.py:360),
        所以那条路在现在的 shard 流水线下**取不到 target**。
      - 本 probe 输出 4ch skeleton **latent**, target 直接取 batch 里已有的
        `skel_latent`(实例骨架 latent, 4ch, 32x32) —— 无需像素图, 与 12ch 用的是同一批数据。

    **为什么用冻结 probe 而不是可训练 head**:
      - 冻结 -> **新增可训练参数 0**。当前主要矛盾是过拟合, 这是决定性优势。
      - 可辨识性: 若 head 可训练, 它会去适应 DiT 的糟糕预测, 结构信号退化成恒等映射
        (见 LatentStructureProbe 的 docstring, 同一条理由)。
      - 梯度仍然能穿过 pred_xstart 回传到主干 —— 这才是"逼主干建立结构感知表征"的本意。
    """

    def __init__(self, in_channels=4, out_channels=4, width=64, depth=3):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.width = int(width)
        self.depth = int(depth)
        self.stem = nn.Conv2d(in_channels, width, 3, padding=1)
        self.blocks = nn.Sequential(*[_ResidualConv(width) for _ in range(depth)])
        self.head = nn.Sequential(
            nn.GroupNorm(8, width), nn.SiLU(), nn.Conv2d(width, out_channels, 1))

    def forward(self, x):
        return self.head(self.blocks(self.stem(x)))


class LatentSkelStructureLoss(nn.Module):
    """冻结 probe 版结构 loss: MSE(probe(pred_xstart), skel_latent_gt), 只在 t<=max_t 生效。

    这是"aux 是辅助"的正确形态 —— 与 12ch 的对比:

    |                | 12ch(旧, 已证有害)        | 本类(辅助)                  |
    |----------------|--------------------------|----------------------------|
    | 位置           | **扩散目标**里            | pred_xstart 上的侧挂 loss   |
    | 扩散目标       | 12ch(被污染)              | 4ch(干净)                   |
    | CFG            | 作用域被 12ch 扰乱        | 完全不受影响(image_channels=4) |
    | 可训练参数     | 主干要多学 8ch 目标       | **0**(probe 冻结)           |
    | 推理成本       | 生成 12ch 再扔 8ch        | **0**                       |
    | 梯度占比       | 等权下 aux 吃 51%         | 由 w 显式控制(建议 0.02~0.1) |

    ⚠ **target 必须是实例骨架, 不能是标准字形骨架**。doc 54 实测: 喂 GT 实例骨架 0.7326
    vs 标准骨架 0.5680(+0.16) —— 我们缺的不是"结构", 是"这个书家写的这个字的具体形态"。
    若 `skel_latent` 指到标准字形, 本 loss 就是纯重复 g, 必败。
    """

    def __init__(self, probe, max_t=0.3):
        super().__init__()
        if probe is None:
            raise ValueError("LatentSkelStructureLoss requires a pretrained frozen probe")
        self.probe = probe
        self.probe.eval()
        self.probe.requires_grad_(False)
        # flow: t in [0,1], t=0 是干净端(flow_matching.py:14 x_t=(1-t)x0+t*noise)
        # -> "只监督足够干净的时刻" = t <= max_t。
        # ⚠ 必须 float: DDPM 的口径是 0~1000, 若照抄 500 会被 int() 截成 0
        #    -> `t <= 0` 恒假 -> loss 静默恒为 0(第二次踩同一个静默失效)。
        self.max_t = float(max_t)

    def train(self, mode=True):
        super().train(mode)
        self.probe.eval()
        return self

    def forward(self, pred_xstart, skel_latent_gt, timesteps):
        """pred_xstart/skel_latent_gt: (N,4,32,32); timesteps: (N,) flow t in [0,1]."""
        zero = pred_xstart.sum() * 0.0
        if skel_latent_gt is None or skel_latent_gt.numel() == 0:
            return zero
        if skel_latent_gt.shape[1] != self.probe.out_channels:
            raise RuntimeError(
                f"skel_latent has {skel_latent_gt.shape[1]} channels but probe outputs "
                f"{self.probe.out_channels}. 检查 skel_latent_shards_dir 是否指向预期骨架。")
        active = timesteps <= self.max_t
        if not active.any():
            return zero
        # sum_over_active / N_total —— 与 LatentStructureLoss 的 `* active_fraction`
        # 口径一致, 保证不同 t 分布下 loss 量级稳定。
        active_fraction = active.float().mean()
        pred_skel = self.probe(pred_xstart[active].float())
        target = skel_latent_gt[active].float().to(pred_skel.device)
        if pred_skel.shape[-2:] != target.shape[-2:]:
            target = F.interpolate(target, size=pred_skel.shape[-2:], mode="bilinear",
                                   align_corners=False)
        return F.mse_loss(pred_skel, target) * active_fraction


class LatentStructureLoss(nn.Module):
    """Canny gradient consistency plus optional frozen-probe skeleton loss."""

    def __init__(self, probe=None, max_timestep=500):
        super().__init__()
        self.probe = probe
        # NOTE: flow matching 下 timestep ∈ [0,1]; DDPM 下 ∈ [0,1000]。必须用 float,
        # 否则 flow 阈值 (如 0.3) 会被 int() 截成 0 -> 门控恒假 (静默失效惯犯)。
        self.max_timestep = float(max_timestep)
        if self.probe is not None:
            self.probe.eval()
            self.probe.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        if self.probe is not None:
            self.probe.eval()
        return self

    def forward(self, pred_x0, target_x0, timesteps, canny=None, skeleton=None):
        active = timesteps <= self.max_timestep
        zero = pred_x0.sum() * 0.0
        if not active.any():
            return {"canny": zero, "skeleton": zero}
        active_fraction = active.float().mean()

        pred = pred_x0[active].float()
        target = target_x0[active].float()
        size = pred.shape[-2:]
        canny_small = downsample_structure(
            canny[active] if canny is not None and canny.numel() else None, size)
        skel_small = downsample_structure(
            skeleton[active] if skeleton is not None and skeleton.numel() else None, size)

        canny_loss = zero
        if canny_small is not None:
            canny_loss = _edge_weighted_gradient_loss(
                pred, target, canny_small.to(pred.device)) * active_fraction

        skeleton_loss = zero
        if skel_small is not None:
            if self.probe is None:
                raise RuntimeError("skeleton latent loss requires a pretrained frozen probe")
            skeleton_logits = self.probe(pred)[:, 1:2]
            skeleton_loss = _balanced_bce_dice(
                skeleton_logits, skel_small.to(pred.device)) * active_fraction
        return {"canny": canny_loss, "skeleton": skeleton_loss}
