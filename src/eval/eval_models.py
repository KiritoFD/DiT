import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights, resnet34, ResNet34_Weights

class MultiTaskCalligraphyEvalNet(nn.Module):
    """
    Multi-task Calligraphy Evaluation Model based on ResNet or DINOv2 backbone.
    Predicts 3 targets simultaneously:
    1. Character (OCR): 7,765 classes
    2. Calligrapher Style: 2,243 classes
    3. Script Style: 12 classes
    """
    def __init__(
        self,
        num_characters=7765,
        num_calligraphers=2243,
        num_scripts=12,
        backbone="dinov2_vits14",
        pretrained=True,
        freeze_backbone=False,
        unfreeze_blocks=0
    ):
        super().__init__()
        self.backbone_type = backbone
        self.is_dinov2 = backbone.startswith("dinov2")
        
        if self.is_dinov2:
            self.backbone = torch.hub.load('facebookresearch/dinov2', backbone)
            in_features = self.backbone.embed_dim # 384 for vits14, 768 for vitb14
            if freeze_backbone:
                for p in self.backbone.parameters():
                    p.requires_grad = False
                
                if unfreeze_blocks > 0:
                    num_blocks = len(self.backbone.blocks)
                    for i in range(num_blocks - unfreeze_blocks, num_blocks):
                        for p in self.backbone.blocks[i].parameters():
                            p.requires_grad = True
                    # Also unfreeze the final LayerNorm
                    for p in self.backbone.norm.parameters():
                        p.requires_grad = True
        elif backbone == "resnet18":
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            base = resnet18(weights=weights)
            in_features = base.fc.in_features
            self.features = nn.Sequential(*list(base.children())[:-1])
        elif backbone == "resnet34":
            weights = ResNet34_Weights.DEFAULT if pretrained else None
            base = resnet34(weights=weights)
            in_features = base.fc.in_features
            self.features = nn.Sequential(*list(base.children())[:-1])
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # 3 Multi-task Heads with independent MLP projections to prevent conflict
        self.char_head = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, num_characters)
        )
        self.callig_head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_calligraphers)
        )
        self.script_head = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Linear(256, num_scripts)
        )

    def forward(self, x):
        """
        x: (B, 3, H, W) normalized image tensor
        returns: (logits_char, logits_callig, logits_script)
        """
        if self.is_dinov2:
            # DINOv2 patch size is 14x14, so H and W must be multiples of 14 (e.g. 224x224)
            if x.shape[-2] != 224 or x.shape[-1] != 224:
                x_dino = F.interpolate(x, size=(224, 224), mode='bicubic', align_corners=False)
            else:
                x_dino = x
            feats = self.backbone(x_dino) # DINOv2 returns CLS token (B, embed_dim)
        else:
            feats = self.features(x)         # (B, in_features, 1, 1)
            feats = torch.flatten(feats, 1)  # (B, in_features)
        
        logits_char = self.char_head(feats)
        logits_callig = self.callig_head(feats)
        logits_script = self.script_head(feats)
        
        return logits_char, logits_callig, logits_script

    def extract_features(self, x):
        """
        Extracts visual features for FID or metric calculations.
        """
        if self.is_dinov2:
            if x.shape[-2] != 224 or x.shape[-1] != 224:
                x_dino = F.interpolate(x, size=(224, 224), mode='bicubic', align_corners=False)
            else:
                x_dino = x
            return self.backbone(x_dino)
        else:
            feats = self.features(x)
            return torch.flatten(feats, 1)
