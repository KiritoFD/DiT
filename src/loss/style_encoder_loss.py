"""Full-batch cross-character SupCon and matched-STD separation."""
import torch
from torch.nn import functional as F


def cross_character_supcon(embedding, slots, scripts, chars, temperature=.1):
    if temperature <= 0:
        raise ValueError('temperature must be positive')
    same_slot = slots[:, None] == slots[None, :]
    positives = same_slot & (chars[:, None] != chars[None, :])
    candidates = (scripts[:, None] == scripts[None, :]) & (~same_slot | positives)
    valid = positives.any(1) & ((~same_slot) & candidates).any(1)
    if not valid.any():
        raise ValueError('SupCon requires cross-character positives and same-script different-writer negatives')
    logits = embedding.float() @ embedding.float().T / temperature
    # Subset valid anchors before logsumexp: avoids all-masked NaN gradients.
    logits = logits[valid]
    log_denom = logits.masked_fill(~candidates[valid], -torch.inf).logsumexp(1)
    pos = positives[valid]
    positive_mean = (logits * pos).sum(1) / pos.sum(1)
    return (log_denom-positive_mean).mean(), valid.float().mean()


def std_separation(anchor, positive, neutral, margin=.2):
    return F.relu(margin + (anchor*neutral).sum(-1) - (anchor*positive).sum(-1)).mean()


def same_character_hard_nce(anchor,positive,negative,valid,temperature=.1):
    """Explicit cross-character positive vs SAME-character other-writer negative."""
    if not valid.any():
        raise ValueError('No same-character hard negatives in this teacher batch')
    scores=torch.stack([(anchor*positive).sum(-1),(anchor*negative).sum(-1)],1)/temperature
    return F.cross_entropy(scores[valid],torch.zeros(int(valid.sum()),dtype=torch.long,device=scores.device))

