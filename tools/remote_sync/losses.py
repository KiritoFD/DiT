import os

import torch
import torch.nn as nn
import torch.nn.functional as F

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
    here = os.path.dirname(os.path.abspath(__file__))
    cand = os.path.join(here, "pretrained_models", "dinov2_vits14_pretrain.safetensors")
    if os.path.exists(cand):
        return cand
    cand = os.path.join(here, "pretrained_models", "dinov2_vits14_pretrain.pth")
    if os.path.exists(cand):
        return cand
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


class SobelCannyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # Sobel kernels
        sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], dtype=torch.float32).view(1, 1, 3, 3)
        
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
        
    def forward(self, x_pred, canny_gt):
        """
        x_pred: (B, 3, H, W) predicted image in [-1, 1]
        canny_gt: (B, 1, H, W) ground truth canny in [0, 1]
        """
        # Convert RGB to grayscale and map from [-1, 1] to [0, 1]
        gray = 0.299 * x_pred[:, 0:1] + 0.587 * x_pred[:, 1:2] + 0.114 * x_pred[:, 2:3]
        gray = (gray + 1.0) / 2.0
        
        # Calculate spatial gradients
        grad_x = F.conv2d(gray, self.sobel_x, padding=1)
        grad_y = F.conv2d(gray, self.sobel_y, padding=1)
        
        # Gradient magnitude
        grad_mag = torch.sqrt(grad_x**2 + grad_y**2 + 1e-6)
        
        # Stable bounded normalization.
        # IMPORTANT: per-image min-max with a tiny eps blows up gradients by ~1/eps
        # on flat (pure-color) predictions, which is exactly what caused the NaN at
        # step ~350 once use_canny was enabled. We detach the denominator and clamp
        # both the scale and the output so gradients stay O(1) even on blank images.
        grad_max = grad_mag.flatten(2).max(dim=-1).values.max(dim=-1).values  # (B,)
        denom = grad_max.clamp(min=1e-3).view(-1, 1, 1, 1)
        grad_norm = (grad_mag / denom.detach()).clamp(max=1.0)
        
        loss = F.l1_loss(grad_norm, canny_gt)
        return loss

class SkeletonLoss(nn.Module):
    def __init__(self, lambda_bg=1.0):
        super().__init__()
        self.lambda_bg = lambda_bg
        
    def forward(self, x_pred, skel_gt):
        """
        x_pred: (B, 3, H, W) predicted image in [-1, 1]
        skel_gt: (B, 1, H, W) ground truth skeleton in [0, 1]
        """
        # Convert to grayscale [0, 1]
        gray = 0.299 * x_pred[:, 0:1] + 0.587 * x_pred[:, 1:2] + 0.114 * x_pred[:, 2:3]
        gray = (gray + 1.0) / 2.0
        
        # Ink intensity: black ink = 1.0, white bg = 0.0
        ink = 1.0 - gray
        
        # Expand skeleton to create background mask (dilation via max pooling)
        skel_expand = F.max_pool2d(skel_gt, kernel_size=3, stride=1, padding=1)
        bg_mask = 1.0 - skel_expand
        
        # On-skeleton penalty: ink should be 1.0 on the skeleton
        sum_skel = skel_gt.sum(dim=[1, 2, 3]).clamp(min=1.0)
        on_skel_loss = (skel_gt * torch.abs(ink - 1.0)).sum(dim=[1, 2, 3]) / sum_skel
        
        # Off-skeleton penalty: ink should be 0.0 on the background
        sum_bg = bg_mask.sum(dim=[1, 2, 3]).clamp(min=1.0)
        off_skel_loss = (bg_mask * ink).sum(dim=[1, 2, 3]) / sum_bg
        
        loss = on_skel_loss + self.lambda_bg * off_skel_loss
        return loss.mean()

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
