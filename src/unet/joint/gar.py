"""GAR objective: MSE + LPIPS/ConvNeXt perceptual + Gram texture, no edge term.

Deviation from the D-arm objective (src/unet/joint/objective.py::image_loss):
  * edge_loss is REMOVED -- it rewards flat, low-frequency structure, which is the
    mechanism behind the D1 thickening/merging drift
  * Gram matrix on VGG16 features is ADDED (weight 10) -- GAR's texture-statistics term
  * perceptual mixes LPIPS (Alex) and ConvNeXt-S features at 1:0.1
CFG is left at 1.0 because K is deployed at cfg=1.0; injecting guidance here would train
the decoder on a latent distribution it never sees at inference.
"""
import torch
import torch.nn.functional as F


class GramExtractor(torch.nn.Module):
    """VGG16 features at three depths; Gram matrices capture stroke texture statistics."""

    def __init__(self, device="cuda"):
        super().__init__()
        import torchvision
        vgg = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1)
        feats = vgg.features
        self.slices = torch.nn.ModuleList([
            feats[:4],    # relu1_2  (64ch, 256x256)
            feats[4:9],   # relu2_2  (128ch, 128x128)
            feats[9:16],  # relu3_3  (256ch, 64x64)
        ])
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        for p in self.parameters():
            p.requires_grad_(False)
        self.to(device).eval()

    def forward(self, x):
        x = (x - self.mean) / self.std
        grams = []
        h = x
        for sl in self.slices:
            h = sl(h)
            b, c, hh, ww = h.shape
            f = h.reshape(b, c, hh * ww)
            g = f @ f.transpose(1, 2) / (c * hh * ww)
            grams.append(g)
        return grams


class DinoPerceptual(torch.nn.Module):
    """DINOv2-base feature distances (GAR uses DINO-family features; ConvNeXt-S is not
    cached on this host, DINOv2-base is, and it is the stronger perceptual trunk)."""

    def __init__(self, device="cuda"):
        super().__init__()
        self.model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        for p in self.parameters():
            p.requires_grad_(False)
        self.to(device).eval()

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, size=(224, 224), mode="bilinear",
                                            align_corners=False)
        x = (x - self.mean) / self.std
        return self.model.forward_features(x)["x_norm_clstoken"]


class GARLoss(torch.nn.Module):
    def __init__(self, device="cuda", gram_weight=10.0, convnext_weight=0.1):
        super().__init__()
        import lpips
        self.lpips = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
        self.gram = GramExtractor(device)
        self.feature = DinoPerceptual(device)
        for p in self.parameters():
            p.requires_grad_(False)
        self.gram_weight = float(gram_weight)
        self.convnext_weight = float(convnext_weight)

    def forward(self, pred, target):
        # pred/target in [0,1], RGB
        mse = F.mse_loss(pred, target)
        p_lpips = self.lpips(pred * 2 - 1, target * 2 - 1).mean()
        p_feat = F.l1_loss(self.feature(pred), self.feature(target))
        perceptual = p_lpips + self.convnext_weight * p_feat
        gp, gt = self.gram(pred), self.gram(target)
        gram = sum(F.l1_loss(a, b) for a, b in zip(gp, gt)) / len(gp)
        total = mse + perceptual + self.gram_weight * gram
        return total, {"mse": float(mse.detach()), "lpips": float(p_lpips.detach()),
                       "convnext": float(p_feat.detach()), "gram": float(gram.detach())}
