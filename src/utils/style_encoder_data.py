"""Deterministic disjoint splits and explicit triplet sampling; no implicit fallback pairs."""
from collections import defaultdict
import numpy as np


def make_splits(chars, slots, sources, reserved_chars=(), seed=89):
    rng = np.random.default_rng(seed)
    chars, slots, sources = map(np.asarray, (chars, slots, sources))
    vocabulary = np.array(sorted(set(chars)-set(reserved_chars)))
    rng.shuffle(vocabulary)
    nv, nt = max(1, len(vocabulary)//10), max(1, len(vocabulary)//10)
    validation_chars = set(vocabulary[:nv])
    test_chars = set(vocabulary[nv:nv+nt]) | set(reserved_chars)
    groups = defaultdict(set)
    for slot, source in zip(slots, sources):
        groups[int(slot)].add(str(source))
    held_sources = set()
    for slot in sorted(groups):
        values = sorted(groups[slot])
        if len(values) >= 3:
            held_sources.add((slot, values[int(rng.integers(len(values)))]))
    is_source = np.array([(int(s), str(w)) in held_sources for s,w in zip(slots,sources)])
    is_val = np.isin(chars, list(validation_chars))
    is_test = np.isin(chars, list(test_chars))
    masks = {'train': ~(is_source | is_val | is_test),
             'val_char': is_val & ~is_source, 'test_char': is_test & ~is_source,
             'test_source': is_source & ~is_val & ~is_test}
    result = {k: np.flatnonzero(v) for k,v in masks.items()}
    if set(slots[result['train']]) != set(slots):
        raise ValueError('Split removed an entire writer/script slot from training')
    return result


class CrossCharacterSampler:
    """Each quartet: real anchor, different-char positive, hard negative, matched STD.

    All real samples contribute to CE and SupCon. Missing hard negatives are
    explicit same-script negatives, counted separately rather than self-paired.
    Source preference is a proxy, not a claim of verified work identity.
    """
    def __init__(self, indices, slots, scripts, chars, sources, seed=89, balance_sources=False,statistics=None):
        self.balance_sources=balance_sources
        self.rng = np.random.default_rng(seed)
        self.slots, self.scripts, self.chars, self.sources = map(np.asarray, (slots,scripts,chars,sources))
        self.by_slot = defaultdict(list)
        self.by_char = defaultdict(list)
        self.by_script = defaultdict(list)
        self.positive_groups = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        self.negative_groups = defaultdict(lambda: defaultdict(list))
        self.script_groups = defaultdict(lambda: defaultdict(list))
        for i in indices:
            self.by_slot[int(slots[i])].append(int(i))
            self.by_char[(int(scripts[i]), str(chars[i]))].append(int(i))
            self.by_script[int(scripts[i])].append(int(i))
            self.positive_groups[int(slots[i])][str(sources[i])][str(chars[i])].append(int(i))
            self.negative_groups[(int(scripts[i]),str(chars[i]))][int(slots[i])].append(int(i))
            self.script_groups[int(scripts[i])][int(slots[i])].append(int(i))
        self.keys = sorted(self.by_slot)
        for s, values in self.by_slot.items():
            if len(set(self.chars[values])) < 2:
                raise ValueError(f'Slot {s} has no cross-character positive')
            if len(set(self.slots[self.by_script[int(self.scripts[values[0]])]])) < 2:
                raise ValueError(f'Slot {s} has no same-script different-writer negative')
        self.stat_positive={};self.stat_negative={}
        if statistics is not None:
            statistics=np.asarray(statistics)
            for a in indices:
                slot=int(slots[a]);sid=int(scripts[a])
                candidates=np.asarray(self.by_slot[slot])
                candidates=candidates[self.chars[candidates]!=self.chars[a]]
                preferred=candidates[self.sources[candidates]!=self.sources[a]]
                candidates=preferred if len(preferred) else candidates
                def closest(values):
                    distance=((statistics[values,0]-statistics[a,0])/.02)**2+((statistics[values,1]-statistics[a,1])/max(.15*statistics[a,1],1e-6))**2
                    distance=np.nan_to_num(distance,nan=np.inf)
                    return values[np.argsort(distance,kind='stable')[:16]]
                self.stat_positive[int(a)]=closest(candidates)
                candidates=np.asarray([i for i in self.by_char[(sid,str(chars[a]))] if slots[i]!=slot],dtype=int)
                if not len(candidates):
                    candidates=np.asarray([i for i in self.by_script[sid] if slots[i]!=slot],dtype=int)
                self.stat_negative[int(a)]=closest(candidates)

    def sample(self, n):
        def pick(values):
            return values[int(self.rng.integers(len(values)))]
        aa, pp, nn, hard, cross_source = [], [], [], 0, 0
        for slot in self.rng.choice(self.keys, n):
            if self.balance_sources:
                anchor_group=self.positive_groups[int(slot)]
                anchor_source=pick(list(anchor_group))
                anchor_char=pick(list(anchor_group[anchor_source]))
                a=int(pick(anchor_group[anchor_source][anchor_char]))
            else:
                a = int(pick(self.by_slot[int(slot)]))
            groups = self.positive_groups[int(slot)]
            available = [s for s,v in groups.items() if len(v)>1 or self.chars[a] not in v]
            other_source = [s for s in available if s != self.sources[a]]
            source = pick(other_source or available)
            character = pick([c for c in groups[source] if c != self.chars[a]])
            p = int(pick(groups[source][character]))
            groups = self.negative_groups[(int(self.scripts[a]),str(self.chars[a]))]
            available = [s for s in groups if s != slot]
            hard += bool(available)
            if not available:
                groups = self.script_groups[int(self.scripts[a])]
                available = [s for s in groups if s != slot]
            b = int(pick(groups[pick(available)]))
            if self.stat_positive:
                p=int(pick(self.stat_positive[a]))
                b=int(pick(self.stat_negative[a]))
            aa.append(a); pp.append(p); nn.append(b)
            cross_source += self.sources[a] != self.sources[p]
        return np.array(aa+pp+nn), {'hard_negative_fraction': hard/n,
                                     'positive_cross_source_fraction': cross_source/n}
