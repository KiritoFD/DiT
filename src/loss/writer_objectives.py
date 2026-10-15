"""Pure writer-slot objectives; no checkpoint loading, training state or I/O."""
import torch
from torch.nn import functional as F


def real_mask(slots, device):
    return torch.tensor([[not b.startswith('std·') and a.split('·')[1] == b.split('·')[1]
                          for b in slots] for a in slots], device=device)


def writer_margin(logits, targets, mask):
    candidates = mask[targets].clone()
    if (~candidates.gather(1, targets[:, None])).any():
        raise ValueError('target must be a real writer slot')
    candidates.scatter_(1, targets[:, None], False)
    if (~candidates.any(1)).any():
        raise ValueError('writer margin needs another real writer in the same script')
    return logits.gather(1, targets[:, None]).squeeze(1) - logits.masked_fill(
        ~candidates, -torch.inf).logsumexp(1)


def calibrated_hinge(margin, target, delta=1.0):
    error = (target - margin).clamp_min(0)
    return torch.where(error <= delta, .5 * error.square(),
                       delta * (error - .5 * delta)).mean()


def cf_loss(a, b, ya, yb, mask, margin=.3):
    return (F.softplus(margin + writer_margin(b, ya, mask) - writer_margin(a, ya, mask))
            + F.softplus(margin + writer_margin(a, yb, mask) - writer_margin(b, yb, mask))).mean()
