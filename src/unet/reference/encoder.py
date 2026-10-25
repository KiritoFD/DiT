"""Continuous reference style; frozen DINO features, trainable projection only."""
import torch
from torch import nn
from torch.nn import functional as F


class StyleHead(nn.Module):
    def __init__(self, input_dim=768, style_dim=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, 256), nn.SiLU(), nn.Linear(256, style_dim))

    def forward(self, features):
        return F.normalize(self.net(features.float()), dim=-1)

    def aggregate(self, features, mask=None):
        if features.ndim != 3:
            raise ValueError('Reference features must be [B,K,768]')
        embedding = self(features)
        if mask is None:
            return F.normalize(embedding.mean(1), dim=-1)
        if mask.shape != features.shape[:2] or not mask.any(1).all():
            raise ValueError('Every target needs at least one valid reference')
        weight = mask.to(embedding.dtype)[..., None]
        return F.normalize((embedding * weight).sum(1) / weight.sum(1), dim=-1)


def supcon(features, writers, chars, scripts, temperature=.1, hard_weight=2.):
    """Cross-character positives; same-script negatives avoid script shortcuts.

    Same-writer/same-character duplicates and cross-script pairs are excluded.
    Hard negatives receive a denominator weight, not a separate loss term.
    """
    if temperature <= 0 or hard_weight < 1:
        raise ValueError('Invalid contrastive temperature/hard-negative weight')
    features = F.normalize(features.float(), dim=-1)
    same_writer = writers[:, None] == writers[None, :]
    same_char = chars[:, None] == chars[None, :]
    same_script = scripts[:, None] == scripts[None, :]
    positive = same_writer & ~same_char & same_script
    negative = ~same_writer & same_script
    valid = positive.any(1) & negative.any(1)
    if not valid.any():
        raise ValueError('Batch has no cross-character positive + same-script negative; fix sampler')
    hard = negative & same_char
    logits = (features @ features.T / temperature)[valid]
    positive = positive[valid]; negative = negative[valid]; hard = hard[valid]
    weighted = logits + hard * logits.new_tensor(hard_weight).log()
    denominator = weighted.masked_fill(~(positive | negative), -torch.inf).logsumexp(1)
    rows = -(logits - denominator[:, None]).masked_fill(~positive, 0).sum(1)
    loss = (rows / positive.sum(1)).mean()
    return loss, dict(valid_anchors=valid.sum().detach(), positives=positive.sum().detach(),
                      hard_negatives=hard.sum().detach())
