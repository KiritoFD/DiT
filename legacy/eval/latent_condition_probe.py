"""Compact evaluator for recovering requested factors from 32x32 VAE latents."""

import torch.nn as nn


class _Residual(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(8, channels), nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels), nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x):
        return x + self.net(x)


class LatentConditionProbe(nn.Module):
    def __init__(self, num_characters=7026, num_calligraphers=1011,
                 num_scripts=5, width=32):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(4, width, 3, padding=1),
            _Residual(width),
            nn.Conv2d(width, width * 2, 3, stride=2, padding=1),
            _Residual(width * 2),
            nn.Conv2d(width * 2, width * 3, 3, stride=2, padding=1),
            _Residual(width * 3),
            nn.GroupNorm(8, width * 3), nn.SiLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(width * 3 * 16, 256), nn.SiLU(),
        )
        self.char_head = nn.Linear(256, num_characters)
        self.callig_head = nn.Linear(256, num_calligraphers)
        self.script_head = nn.Linear(256, num_scripts)

    def forward(self, x):
        features = self.features(x)
        return (self.char_head(features), self.callig_head(features),
                self.script_head(features))
