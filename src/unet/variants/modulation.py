"""Small initialization/gain primitives shared by optional U-Net modulation."""
import torch
from torch import nn


def zero_linear(in_features: int, out_features: int) -> nn.Linear:
    """A trainable projection with zero output, not a detached/constant path."""
    layer = nn.Linear(in_features, out_features)
    nn.init.zeros_(layer.weight)
    nn.init.zeros_(layer.bias)
    return layer


def bounded_gain(logits: torch.Tensor) -> torch.Tensor:
    """Unit initial gain and bounded positive modulation in (0.5, 1.5)."""
    return 1.0 + 0.5 * torch.tanh(logits)
