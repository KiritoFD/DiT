import os

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------
# StructDecoder: latent(4,32,32) → skel/canny logit(1,256,256).
# 训练时冻结, 接在 DiT pred_xstart_latent 后面提供结构监督梯度.
# PixelShuffle 上采样 (零参数 reshape) 保证梯度完美透传.
# ----------------------------------------------------------------------------

class StructDecoder(nn.Module):
    """latent(4,32,32) → logit(1,256,256).

    多尺度 pixel-shuffle: 32→64→128→256, 每级 1×1 conv + PixelShuffle(2) + SiLU.
    PixelShuffle 是纯 reshape+permute, 梯度完美透传.
    全程可导, ~348K params (base=64, depth=6).
    支持 gradient checkpointing: 每个 stage 作为一个 checkpoint segment,
    用重算换显存 — 反向时只保留 stage 边界的激活, 中间重算.
    """
    def __init__(self, in_ch=4, base=64, depth=6, use_checkpoint=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.stem = nn.Sequential(nn.Conv2d(in_ch, base, 3, padding=1), nn.SiLU())
        blocks = []
        for _ in range(depth):
            blocks += [nn.GroupNorm(8, base), nn.SiLU(), nn.Conv2d(base, base, 3, padding=1)]
        self.body = nn.Sequential(*blocks)
        self.up1 = nn.Sequential(nn.Conv2d(base, base * 4, 1), nn.PixelShuffle(2), nn.SiLU(),
                                  nn.Conv2d(base, base, 3, padding=1), nn.SiLU())
        self.up2 = nn.Sequential(nn.Conv2d(base, base * 4, 1), nn.PixelShuffle(2), nn.SiLU(),
                                  nn.Conv2d(base, base, 3, padding=1), nn.SiLU())
        self.up3 = nn.Sequential(nn.Conv2d(base, base * 4, 1), nn.PixelShuffle(2), nn.SiLU())
        self.head = nn.Conv2d(base, 1, 1)

    def _body_forward(self, x):
        return self.body(x)

    def forward(self, x):
        f = self.stem(x)
        if self.use_checkpoint:
            f = torch.utils.checkpoint.checkpoint(self._body_forward, f, use_reentrant=False)
        else:
            f = self.body(f)
        if self.use_checkpoint:
            f = torch.utils.checkpoint.checkpoint(self.up1, f, use_reentrant=False)
            f = torch.utils.checkpoint.checkpoint(self.up2, f, use_reentrant=False)
            f = torch.utils.checkpoint.checkpoint(self.up3, f, use_reentrant=False)
        else:
            f = self.up1(f)
            f = self.up2(f)
            f = self.up3(f)
        return self.head(f)


class LatentStructLoss(nn.Module):
    """冻结 StructDecoder + BCE loss: pred_xstart_latent → skel/canny logit → BCE.

    梯度路径: BCE@256 → logit → PixelShuffle(零参数) → conv@32 → latent → DiT.
    不经过 VAE, 显存开销远小于 pixel 结构损失.
    """
    def __init__(self, decoder_ckpt_path, decoder_type="skel", pos_weight=15.0, use_checkpoint=False):
        super().__init__()
        ck = torch.load(decoder_ckpt_path, map_location="cpu", weights_only=False)
        cfg = ck.get("config", {})
        base = int(cfg.get("base", 64))
        depth = int(cfg.get("depth", 6))
        self.decoder = StructDecoder(in_ch=4, base=base, depth=depth, use_checkpoint=use_checkpoint)
        self.decoder.load_state_dict(ck["model"])
        for p in self.decoder.parameters():
            p.requires_grad = False
        self.decoder.eval()
        self.decoder_type = decoder_type
        self.register_buffer("pos_weight", torch.tensor(pos_weight))

    def forward(self, pred_xstart_latent, struct_gt):
        """pred_xstart_latent: (B,4,32,32), struct_gt: (B,1,256,256) binary."""
        logit = self.decoder(pred_xstart_latent)
        return F.binary_cross_entropy_with_logits(
            logit, struct_gt, pos_weight=self.pos_weight)

# ----------------------------------------------------------------------------
# DINOv2 teacher loading without github access.
#
# The torch.hub route (torch.hub.load("facebookresearch/dinov2", ...)) requires
# cloning the repo from github.com, which is blocked on some training boxes.
# As a drop-in alternative we build the teacher from a local transformers-style
# .safetensors checkpoint (e.g. the ModelScope export "anyforge/facebook-dinov2",
# file dinov2/dinov2-small/model.safetensors) using transformers.Dinov2Model.
# The weights are loaded directly, so no network access is needed at run time.
# ----------------------------------------------------------------------------

def _default_dino_ckpt():
    """Resolve a local DINOv2 checkpoint path.

    Priority: $DINO_WEIGHTS env var, then ./pretrained_models/dinov2_vits14_pretrain.safetensors
    (next to this file). Returns None if nothing exists.
    """
    env = os.environ.get("DINO_WEIGHTS", "")
    if env and os.path.exists(env):
        return env
    # 从本文件向上逐层查找 pretrained_models/。
    # 注意：本文件位于 src/loss/ 下，"向上两级" 得到的是 src/ 而不是项目根，
    # 早期实现正是少算了一级，导致本地 ckpt 永远找不到、静默回退到
    # torch.hub（需访问 github，在受限机器上会失败）。这里改成向上搜索，
    # 与 src/loss/ 或 src/ 两种布局都兼容。
    here = os.path.dirname(os.path.abspath(__file__))
    for _ in range(5):
        for name in ("dinov2_vits14_pretrain.safetensors",
                     "dinov2_vits14_pretrain.pth"):
            cand = os.path.join(here, "pretrained_models", name)
            if os.path.exists(cand):
                return cand
        here = os.path.dirname(here)
    return None


def _load_local_dinov2(ckpt_path):
    """Build a DINOv2 teacher from a transformers-style safetensors checkpoint.

    The ModelScope export uses the HuggingFace transformers weight layout, so we
    rebuild the network with transformers.Dinov2Model and load the state dict
    directly. Architecture (patch size 14, mlp ratio 4, image size 224) is
    inferred from the checkpoint itself, so vit-small/base/large/giant all work.
    """
    from safetensors import safe_open
    from transformers import Dinov2Config, Dinov2Model

    with safe_open(ckpt_path, framework="pt") as f:
        keys = list(f.keys())
        hidden = f.get_tensor("encoder.layer.0.attention.attention.query.weight").shape[0]
    depth = max(int(k.split(".")[2]) for k in keys if k.startswith("encoder.layer.")) + 1
    heads = {384: 6, 768: 12, 1024: 16, 1536: 24}.get(hidden, 6)
    n_reg = 1 if any("register_tokens" in k for k in keys) else 0
    config = Dinov2Config(
        image_size=224,
        patch_size=14,
        num_channels=3,
        hidden_size=hidden,
        num_hidden_layers=depth,
        num_attention_heads=heads,
        intermediate_size=4 * hidden,
        hidden_act="gelu",
        layer_norm_eps=1e-6,
        layer_scale_init_value=1.0,
        num_register_tokens=n_reg,
    )
    model = Dinov2Model(config)
    sd = {}
    with safe_open(ckpt_path, framework="pt") as f:
        for k in keys:
            sd[k] = f.get_tensor(k)

    # The ModelScope export may have been trained at a different resolution than
    # the 224x224 config above (e.g. 518x518 -> 1370 = 1 cls + 37x37 patches).
    # DINOv2 positional embeddings are designed to be interpolated; resize the
    # patch part to match the config before loading.
    target_len = config.image_size // config.patch_size
    pe = sd.get("embeddings.position_embeddings")
    if pe is not None and pe.shape[1] != 1 + target_len * target_len:
        cls = pe[:, :1]                                    # (1, 1, D)
        patch = pe[:, 1:]                                  # (1, h*w, D)
        h = w = int(round((patch.shape[1]) ** 0.5))        # source grid (37 for 518)
        patch = patch.reshape(1, h, w, -1).permute(0, 3, 1, 2)  # (1, D, h, w)
        patch = F.interpolate(patch, size=(target_len, target_len), mode="bicubic",
                              align_corners=False)
        patch = patch.permute(0, 2, 3, 1).reshape(1, target_len * target_len, -1)
        sd["embeddings.position_embeddings"] = torch.cat([cls, patch], dim=1)
        print(f"[REPALoss] resized position_embeddings {tuple(pe.shape)} -> "
              f"{tuple(sd['embeddings.position_embeddings'].shape)}")

    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        raise RuntimeError(f"DINOv2 teacher missing keys: {missing[:8]} ...")
    if unexpected:
        print(f"[REPALoss] ignoring unexpected teacher keys: {unexpected[:5]} ...")
    return model


class _TeacherWrapper(nn.Module):
    """Normalize teacher output behind a single forward_features() interface.

    - torch.hub DINOv2 returns a dict with 'x_norm_patchtokens'.
    - transformers.Dinov2Model returns a BaseModelOutput whose last_hidden_state
      is (B, 1 + num_patches, D) with the CLS token in position 0.
    Both are reduced to the (B, num_patches, D) patch-token tensor.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward_features(self, x):
        if hasattr(self.model, "forward_features"):
            out = self.model.forward_features(x)
            if isinstance(out, dict):
                return out["x_norm_patchtokens"]
            if isinstance(out, torch.Tensor):  # some torch.hub refs return raw tensor
                return out[:, 1:]
            raise TypeError(f"unexpected teacher output type: {type(out)}")
        out = self.model(x)
        return out.last_hidden_state[:, 1:]


# ============================================================================
# 像素结构损失（修复版）
#
# 旧版的两个致命问题（已删除，此处仅留说明）：
#   1) SkeletonLoss v1: 用 3x3 膨胀的 1px 骨架当 "背景掩码"，而毛笔笔画通常
#      10~30px 宽，笔画主体被误判为背景并遭 off_skel 惩罚 -> 模型被迫抹掉笔画
#      血肉。修复：SkeletonLoss 只做正向牵引（recall），绝不惩罚骨架以外的
#      墨水，笔画粗细与飞白交给扩散损失决定。
#   2) Canny 边缘损失 v1: 逐图除以最大梯度（早期模糊 x0 中噪点被放大成"最强
#      边缘"，梯度尺度失真）；并且用连续梯度场去 L1 拟合 1px 二值 Canny 线，
#      逼模型消灭抗锯齿与晕染。修复：EdgeGradientLoss 不做归一化、不做二值
#      匹配，直接对齐预测图与 GT 图的连续 Sobel 梯度幅值场（超分领域经典
#      gradient-profile 损失），保边缘锐度同时保留灰度过渡。
# ============================================================================

class EdgeGradientLoss(nn.Module):
    """边缘损失: 预测图与 GT 图的 Sobel 梯度幅值场直接匹配 (L1)。"""

    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]],
                               dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]],
                               dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def _grad(self, img):
        gray = 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]
        gray = (gray + 1.0) / 2.0
        gx = F.conv2d(gray, self.sobel_x, padding=1)
        gy = F.conv2d(gray, self.sobel_y, padding=1)
        return torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)

    def forward(self, x_pred, x_gt):
        """
        x_pred: (B, 3, H, W) 预测图 [-1, 1]
        x_gt:   (B, 3, H, W) 真实图 [-1, 1] (梯度场 detach, 只回传预测侧)
        """
        with torch.no_grad():
            gm_gt = self._grad(x_gt).detach()
        gm_pred = self._grad(x_pred)
        return F.l1_loss(gm_pred, gm_gt)


class SkeletonLoss(nn.Module):
    """骨架损失: 只做"正向牵引" (recall), 绝不惩罚骨架以外的墨水。"""

    def __init__(self):
        super().__init__()

    def forward(self, x_pred, skel_gt):
        """
        x_pred: (B, 3, H, W) predicted image in [-1, 1]
        skel_gt: (B, 1, H, W) ground truth skeleton in [0, 1] (1 = 骨架线)
        """
        gray = 0.299 * x_pred[:, 0:1] + 0.587 * x_pred[:, 1:2] + 0.114 * x_pred[:, 2:3]
        gray = (gray + 1.0) / 2.0
        ink = 1.0 - gray  # 黑墨=1, 白底=0
        sum_skel = skel_gt.sum(dim=[1, 2, 3]).clamp(min=1.0)
        on_skel = (skel_gt * torch.abs(ink - 1.0)).sum(dim=[1, 2, 3]) / sum_skel
        return on_skel.mean()


class REPALoss(nn.Module):
    """
    Representation Alignment Loss (REPA) using a frozen DINOv2 teacher.
    Aligns DiT intermediate features with DINOv2 semantic features.
    """
    def __init__(self, student_dim, teacher_dim=384, teacher_backbone="dinov2_vits14",
                 teacher_ckpt=None):
        super().__init__()
        # Load frozen DINOv2 teacher:
        #   1) a local safetensors checkpoint (ModelScope export) if provided or found
        #   2) otherwise fall back to torch.hub from github.
        if teacher_ckpt is None:
            teacher_ckpt = _default_dino_ckpt()
        teacher = None
        if teacher_ckpt and os.path.exists(teacher_ckpt):
            try:
                teacher = _load_local_dinov2(teacher_ckpt)
                print(f"[REPALoss] loaded local DINOv2 teacher from {teacher_ckpt}")
            except Exception as e:  # noqa: BLE001 - fall back below
                print(f"[REPALoss] failed to load local teacher ({e!r}); falling back to torch.hub")
                teacher = None
        if teacher is None:
            teacher = torch.hub.load('facebookresearch/dinov2', teacher_backbone)
        self.teacher = _TeacherWrapper(teacher)
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()

        self.is_vits14 = (teacher_backbone == "dinov2_vits14")

        # Projection MLP to map student dimension to teacher dimension
        self.proj = nn.Sequential(
            nn.Linear(student_dim, teacher_dim),
            nn.SiLU(),
            nn.Linear(teacher_dim, teacher_dim)
        )

        # Normalization for input image
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, student_feats, x_0):
        """
        student_feats: (B, num_patches, student_dim) e.g. (B, 256, 384)
        x_0: (B, 3, 256, 256) original image in [-1, 1]
        """
        self.teacher.eval()
        with torch.no_grad():
            # 1. Prepare input for DINOv2: [-1, 1] -> [0, 1] -> Normalize -> Resize to 224x224
            x_0 = x_0.float()
            x_0_01 = (x_0 + 1.0) / 2.0
            x_norm = (x_0_01 - self.mean) / self.std
            x_224 = F.interpolate(x_norm, size=(224, 224), mode='bicubic', align_corners=False)

            # 2. Extract teacher features (wrapper normalizes dict / BaseModelOutput
            #    to a (B, num_patches, teacher_dim) patch-token tensor)
            teacher_feats = self.teacher.forward_features(x_224).float()  # (B, 256, teacher_dim)

        # 3. Project student features.
        # Student features may be fp16 (produced inside autocast); upcast to fp32
        # before projecting, otherwise matmul with the fp32 proj weight raises a dtype error.
        student_feats_proj = self.proj(student_feats.float())  # (B, 256, teacher_dim)

        # 4. Compute Representation Alignment Loss (Negative Cosine Similarity)
        # We maximize cosine similarity, so we minimize 1 - cosine_similarity
        cos_sim = F.cosine_similarity(student_feats_proj, teacher_feats, dim=-1) # (B, 256)
        loss = 1.0 - cos_sim.mean()

        return loss