"""Train-only 2x2 writer/character blocks and mixed batch sampler.

Implements:
1. 2x2 Writer-Character Quartets (同书家异字正样本 vs 异书家同字困难负样本 vs 异书家异字普通负样本).
2. Cross-source preference for positive pairs (to avoid paper/background shortcuts).
3. 384 quartet + 128 sqrt-balanced mixed sampling per 512 microbatch.
4. Pairwise softplus contrastive loss (margin=0.1, tau=0.1).
"""
from collections import defaultdict
import numpy as np
import torch
from torch.nn import functional as F


class WriterQuartets:
    def __init__(self, train, slots, scripts, chars, sources, seed=89):
        self.rng = np.random.default_rng(seed)
        self.train = np.asarray(train, dtype=np.int64)
        self.slots = np.asarray(slots)
        self.scripts = np.asarray(scripts)
        self.chars = np.asarray(chars)
        self.sources = np.asarray(sources)

        # Build group map: slot -> char -> [train_indices]
        self.groups = defaultdict(lambda: defaultdict(list))
        self.slot_scripts = {}
        self.slot_counts = defaultdict(int)

        for i in self.train:
            slot = int(self.slots[i])
            char = str(self.chars[i])
            self.groups[slot][char].append(int(i))
            self.slot_scripts[slot] = int(self.scripts[i])
            self.slot_counts[slot] += 1

        # Precompute sqrt sampling probabilities for class-balanced sampling
        all_slots = sorted(self.slot_counts.keys())
        counts = np.array([self.slot_counts[s] for s in all_slots], dtype=np.float64)
        sqrt_counts = np.sqrt(counts)
        self.slot_p = sqrt_counts / sqrt_counts.sum()
        self.slot_list = all_slots
        self.slot_to_train_indices = {s: [int(i) for i in self.train if self.slots[i] == s] for s in all_slots}

        # Build candidate pairs of (slot_a, slot_b) sharing the same script
        self.full = []
        self.fallback = []
        keys = sorted(self.groups.keys())

        for i, a in enumerate(keys):
            for b in keys[i + 1:]:
                if self.slot_scripts[a] != self.slot_scripts[b]:
                    continue
                shared = sorted(set(self.groups[a].keys()) & set(self.groups[b].keys()))
                if len(shared) >= 2:
                    self.full.append((a, b, shared))
                elif shared and len(self.groups[a]) >= 2 and len(self.groups[b]) >= 2:
                    self.fallback.append((a, b, shared))

        if not self.full and not self.fallback:
            raise ValueError('No legal training writer-character blocks found in dataset')

        self.coverage = {
            'full_writer_pairs': len(self.full),
            'fallback_writer_pairs': len(self.fallback),
            'unique_metadata_rectangles': int(sum(len(cs) * (len(cs) - 1) // 2 for _, _, cs in self.full)),
            'slots_without_full_rectangles': sorted(set(keys) - {x for a, b, _ in self.full for x in (a, b)})
        }

    def _pick(self, slot, char, avoid_source=None):
        """Pick a sample from groups[slot][char], preferring a different source if possible."""
        pool = self.groups[slot][char]
        if avoid_source is not None:
            preferred = [i for i in pool if self.sources[i] != avoid_source]
            if preferred:
                return int(preferred[int(self.rng.integers(len(preferred)))])
        return int(pool[int(self.rng.integers(len(pool)))])

    def sample_mixed(self, total_batch=512, quartet_samples=384):
        """Sample a microbatch of total_batch containing quartet_samples and balanced random samples.
        
        Args:
            total_batch: Total microbatch size (e.g., 512).
            quartet_samples: Number of samples participating in 2x2 quartets (must be divisible by 4, e.g. 384).
            
        Returns:
            ids: 1D numpy array of sample indices, length total_batch.
                 indices [:quartet_samples] are grouped 2x2 quartets.
                 indices [quartet_samples:] are category-balanced samples for general CE.
            meta: Dictionary with quartet statistics.
        """
        if quartet_samples % 4 != 0:
            raise ValueError('quartet_samples must be divisible by 4')
        if quartet_samples > total_batch:
            raise ValueError('quartet_samples cannot exceed total_batch')

        n_quartets = quartet_samples // 4
        blocks = []
        full_cnt = 0
        cross_source_cnt = 0

        # 1. Sample 2x2 Quartets
        for _ in range(n_quartets):
            use_full = bool(self.full) and (not self.fallback or self.rng.random() < 0.90)
            pool = self.full if use_full else self.fallback
            a, b, shared = pool[int(self.rng.integers(len(pool)))]

            if use_full:
                c1, c2 = self.rng.choice(shared, 2, replace=False)
                d2 = c2
                full_cnt += 1
            else:
                c1 = shared[int(self.rng.integers(len(shared)))]
                c2 = self.rng.choice([c for c in self.groups[a] if c != c1])
                d2 = self.rng.choice([c for c in self.groups[b] if c != c1])

            # Slot A, Char 1
            ac = self._pick(a, c1)
            # Slot A, Char 2 (prefer different source from ac)
            ad = self._pick(a, c2, avoid_source=self.sources[ac])
            # Slot B, Char 1
            bc = self._pick(b, c1)
            # Slot B, Char 2 (prefer different source from bc)
            bd = self._pick(b, d2, avoid_source=self.sources[bc])

            blocks.extend([ac, ad, bc, bd])
            if self.sources[ac] != self.sources[ad]:
                cross_source_cnt += 1
            if self.sources[bc] != self.sources[bd]:
                cross_source_cnt += 1

        # 2. Sample balanced remainder (P(slot) ~ sqrt(n_slot))
        remainder = total_batch - quartet_samples
        if remainder > 0:
            chosen_slots = self.rng.choice(self.slot_list, size=remainder, p=self.slot_p)
            for s in chosen_slots:
                idx_pool = self.slot_to_train_indices[s]
                blocks.append(int(idx_pool[int(self.rng.integers(len(idx_pool)))]))

        ids = np.asarray(blocks, dtype=np.int64)
        meta = {
            'n_quartet_samples': quartet_samples,
            'full_quartet_fraction': full_cnt / max(1, n_quartets),
            'cross_source_positive_fraction': cross_source_cnt / max(1, 2 * n_quartets)
        }
        return ids, meta

    def sample(self, batch):
        """Legacy interface: all samples are quartets."""
        return self.sample_mixed(total_batch=batch, quartet_samples=batch)


def cross_loss(embedding, chars, n_quartet_samples=None, margin=0.1, tau=0.1):
    """Compute 2x2 quartet contrastive loss over quartet samples.
    
    Args:
        embedding: Tensor [B, D] L2-normalized embeddings.
        chars: List or array of character strings, length B.
        n_quartet_samples: Number of samples in embedding that belong to quartets (default: len(embedding)).
        margin: Contrastive margin m (default 0.1).
        tau: Softplus temperature tau (default 0.1).
        
    Returns:
        loss: Scalar Tensor cross-contrastive loss.
        stats: Dictionary of detailed diagnostic statistics.
    """
    if n_quartet_samples is None:
        n_quartet_samples = len(embedding)

    q_emb = embedding[:n_quartet_samples]
    q_chars = np.asarray(chars[:n_quartet_samples])

    b = len(q_emb)
    if b == 0 or b % 4 != 0:
        return torch.zeros((), device=embedding.device), {}

    base = torch.arange(b, device=embedding.device).reshape(-1, 4)
    # 4 cyclic anchor permutations:
    # 0 -> pos:1, same_char_neg:2, diff_char_neg:3
    # 1 -> pos:0, same_char_neg:3, diff_char_neg:2
    # 2 -> pos:3, same_char_neg:0, diff_char_neg:1
    # 3 -> pos:2, same_char_neg:1, diff_char_neg:0
    positive = base[:, [1, 0, 3, 2]].reshape(-1)
    same = base[:, [2, 3, 0, 1]].reshape(-1)
    different = base[:, [3, 2, 1, 0]].reshape(-1)

    hard = torch.as_tensor(q_chars == q_chars[same.cpu().numpy()], device=embedding.device)
    diff = torch.as_tensor(q_chars != q_chars[different.cpu().numpy()], device=embedding.device)

    # Cosine similarities
    s_p = (q_emb * q_emb[positive]).sum(dim=-1)
    s_h = (q_emb * q_emb[same]).sum(dim=-1)
    s_o = (q_emb * q_emb[different]).sum(dim=-1)

    # Pairwise softplus: L = softplus((s_neg - s_pos + m) / tau)
    hn = F.softplus((s_h - s_p + margin) / tau)
    dn = F.softplus((s_o - s_p + margin) / tau)

    h_loss = hn[hard].mean() if hard.any() else torch.zeros((), device=embedding.device)
    d_loss = dn[diff].mean() if diff.any() else torch.zeros((), device=embedding.device)

    total_cross = 0.5 * (h_loss + d_loss)

    stats = {
        'same_char_hard_fraction': float(hard.float().mean()),
        'cross_hard': float(h_loss.detach()),
        'cross_diff': float(d_loss.detach()),
        'mean_pos_sim': float(s_p.mean().detach()),
        'mean_hard_neg_sim': float(s_h[hard].mean().detach()) if hard.any() else 0.0
    }
    return total_cross, stats


def build_same_script_mask(targets, slot_scripts, n_slots):
    """Construct boolean mask [B, n_slots] where True indicates candidates sharing target's script.
    
    Args:
        targets: Tensor [B] of slot indices.
        slot_scripts: List or 1D array of length n_slots containing integer script IDs.
        n_slots: Total number of slots.
        
    Returns:
        mask: Boolean Tensor [B, n_slots].
    """
    dev = targets.device
    slot_s = torch.as_tensor(slot_scripts, dtype=torch.long, device=dev)
    target_s = slot_s[targets]
    mask = (target_s.unsqueeze(1) == slot_s.unsqueeze(0))
    return mask

