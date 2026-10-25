"""Metric learning losses for Multi-loss Siamese CNN: Contrastive and Ranking."""
import numpy as np
import torch
from torch.nn import functional as F


def same_script_ce_loss(slot_logits, targets, script_mask=None):
    """Compute Cross-Entropy loss restricted to same-script candidate slots.
    
    Args:
        slot_logits: Tensor [B, n_slots]
        targets: Tensor [B] of slot indices
        script_mask: Optional boolean or float Tensor [B, n_slots]
    """
    if script_mask is not None:
        logits = torch.where(script_mask.bool(), slot_logits, slot_logits - 10000.0)
    else:
        logits = slot_logits
    return F.cross_entropy(logits, targets)


def euclid_contrastive_loss(embedding, chars, n_quartet_samples=None, margin=1.0):
    """Paper Contrastive Loss (CJIG 2023): Absolute Euclidean distance constraint.
    
    L_CT = 1/2 * [ E_{pos} d_pos^2 + 1/2 * (E_{hard} max(0, m - d_hard)^2 + E_{other} max(0, m - d_other)^2) ]
    
    Args:
        embedding: Tensor [B, D] L2-normalized embeddings.
        chars: List or array of character strings, length B.
        n_quartet_samples: Number of samples in embedding that belong to quartets.
        margin: Euclidean distance threshold m (default 1.0 for normalized 256D space).
    """
    if n_quartet_samples is None:
        n_quartet_samples = len(embedding)

    q_emb = embedding[:n_quartet_samples]
    q_chars = np.asarray(chars[:n_quartet_samples])

    b = len(q_emb)
    if b == 0 or b % 4 != 0:
        return torch.zeros((), device=embedding.device), {}

    base = torch.arange(b, device=embedding.device).reshape(-1, 4)
    positive = base[:, [1, 0, 3, 2]].reshape(-1)
    same = base[:, [2, 3, 0, 1]].reshape(-1)
    different = base[:, [3, 2, 1, 0]].reshape(-1)

    hard = torch.as_tensor(q_chars == q_chars[same.cpu().numpy()], device=embedding.device)
    diff = torch.as_tensor(q_chars != q_chars[different.cpu().numpy()], device=embedding.device)

    # Euclidean distance: ||e_i - e_j||_2 = sqrt(clamp(sum((e_i - e_j)^2), min=1e-12))
    d_p = torch.norm(q_emb - q_emb[positive], dim=-1)
    d_h = torch.norm(q_emb - q_emb[same], dim=-1)
    d_o = torch.norm(q_emb - q_emb[different], dim=-1)

    pos_loss = d_p.square().mean()

    # Hinge on negative distance
    neg_h = F.relu(margin - d_h).square()
    neg_o = F.relu(margin - d_o).square()

    h_loss = neg_h[hard].mean() if hard.any() else torch.zeros((), device=embedding.device)
    o_loss = neg_o[diff].mean() if diff.any() else torch.zeros((), device=embedding.device)

    total_ct = 0.5 * (pos_loss + 0.5 * (h_loss + o_loss))

    stats = {
        'mean_pos_dist': float(d_p.mean().detach()),
        'mean_hard_neg_dist': float(d_h[hard].mean().detach()) if hard.any() else 0.0,
        'mean_other_neg_dist': float(d_o[diff].mean().detach()) if diff.any() else 0.0,
        'active_hard_fraction': float((d_h[hard] < margin).float().mean().detach()) if hard.any() else 0.0,
        'contrastive_pos_loss': float(pos_loss.detach()),
        'contrastive_neg_loss': float((0.5 * (h_loss + o_loss)).detach())
    }
    return total_ct, stats


def relative_ranking_loss(embedding, chars, n_quartet_samples=None, margin=0.1, tau=0.1):
    """D1 Relative Ranking Loss: Pairwise softplus on cosine similarity margin.
    
    L_rank = 1/2 * [ softplus((s_hard - s_pos + m) / tau) + softplus((s_other - s_pos + m) / tau) ]
    """
    if n_quartet_samples is None:
        n_quartet_samples = len(embedding)

    q_emb = embedding[:n_quartet_samples]
    q_chars = np.asarray(chars[:n_quartet_samples])

    b = len(q_emb)
    if b == 0 or b % 4 != 0:
        return torch.zeros((), device=embedding.device), {}

    base = torch.arange(b, device=embedding.device).reshape(-1, 4)
    positive = base[:, [1, 0, 3, 2]].reshape(-1)
    same = base[:, [2, 3, 0, 1]].reshape(-1)
    different = base[:, [3, 2, 1, 0]].reshape(-1)

    hard = torch.as_tensor(q_chars == q_chars[same.cpu().numpy()], device=embedding.device)
    diff = torch.as_tensor(q_chars != q_chars[different.cpu().numpy()], device=embedding.device)

    s_p = (q_emb * q_emb[positive]).sum(dim=-1)
    s_h = (q_emb * q_emb[same]).sum(dim=-1)
    s_o = (q_emb * q_emb[different]).sum(dim=-1)

    hn = F.softplus((s_h - s_p + margin) / tau)
    dn = F.softplus((s_o - s_p + margin) / tau)

    h_loss = hn[hard].mean() if hard.any() else torch.zeros((), device=embedding.device)
    d_loss = dn[diff].mean() if diff.any() else torch.zeros((), device=embedding.device)

    total_rank = 0.5 * (h_loss + d_loss)

    stats = {
        'mean_pos_sim': float(s_p.mean().detach()),
        'mean_hard_neg_sim': float(s_h[hard].mean().detach()) if hard.any() else 0.0,
        'mean_other_neg_sim': float(s_o[diff].mean().detach()) if diff.any() else 0.0,
        'same_char_hard_fraction': float(hard.float().mean()),
        'cross_hard': float(h_loss.detach()),
        'cross_diff': float(d_loss.detach())
    }
    return total_rank, stats


def compute_metric_loss(embedding, chars, loss_type='ranking', n_quartet_samples=None,
                        contrastive_margin=1.0, ranking_margin=0.1, ranking_tau=0.1):
    """Unified dispatcher for metric loss ablation (Contrastive vs. Ranking)."""
    if loss_type == 'contrastive':
        return euclid_contrastive_loss(embedding, chars, n_quartet_samples=n_quartet_samples, margin=contrastive_margin)
    elif loss_type == 'ranking':
        return relative_ranking_loss(embedding, chars, n_quartet_samples=n_quartet_samples, margin=ranking_margin, tau=ranking_tau)
    else:
        raise ValueError(f"Unknown metric loss_type: {loss_type}")
