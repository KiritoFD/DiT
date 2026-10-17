"""GT feature bank is supervision only; never a generator condition."""
from collections import defaultdict
import numpy as np
import torch
from torch.nn import functional as F


class CrossCharacterFeatureSampler:
    def __init__(self,writers,scripts,chars,sources,seed=89):
        self.writers,self.scripts,self.chars,self.sources=map(np.asarray,(writers,scripts,chars,sources))
        self.rng=np.random.default_rng(seed)
        self.slots=defaultdict(lambda:defaultdict(lambda:defaultdict(list)))
        self.hard=defaultdict(lambda:defaultdict(list))
        self.by_script=defaultdict(lambda:defaultdict(list))
        for i,(w,s,c,source) in enumerate(zip(writers,scripts,chars,sources)):
            self.slots[(int(w),int(s))][str(source)][str(c)].append(i)
            self.hard[(int(s),str(c))][int(w)].append(i)
            self.by_script[int(s)][int(w)].append(i)

    def sample(self,writers,scripts,chars,sources,n_neg=8):
        def pick(values):
            return values[int(self.rng.integers(len(values)))]
        positives,negatives=[],[]
        hard_count=cross_source=0
        for w,s,c,source in zip(writers,scripts,chars,sources):
            groups=self.slots.get((int(w),int(s)),{})
            available=[v for v,by_char in groups.items() if len(by_char)>1 or str(c) not in by_char]
            if not available:
                raise ValueError(f'No different-character training positive for {(w,s,c)}')
            preferred=[v for v in available if v!=str(source)]
            chosen_source=pick(preferred or available)
            chosen_char=pick([ch for ch in groups[chosen_source] if ch!=str(c)])
            positives.append(pick(groups[chosen_source][chosen_char]))
            cross_source+=chosen_source!=str(source)
            neg_groups=self.hard.get((int(s),str(c)),{})
            rivals=[v for v in neg_groups if v!=int(w)]
            hard_count+=bool(rivals)
            if not rivals:
                neg_groups=self.by_script.get(int(s),{})
                rivals=[v for v in neg_groups if v!=int(w)]
            if not rivals:
                raise ValueError(f'No same-script other-writer training negative for {(w,s,c)}')
            negatives.append([pick(neg_groups[pick(rivals)]) for _ in range(n_neg)])
        n=len(positives)
        return np.asarray(positives),np.asarray(negatives),{
            'hard_negative_fraction':hard_count/max(n,1),'positive_cross_source_fraction':cross_source/max(n,1)}


def cross_character_nce(query,reference_features,positive_indices,negative_indices,temperature=.1):
    if temperature<=0:
        raise ValueError('temperature must be positive')
    q=F.normalize(query.float(),dim=-1)
    bank=reference_features.detach()
    pos=F.normalize(bank[positive_indices].float(),dim=-1)
    neg=F.normalize(bank[negative_indices].float(),dim=-1)
    scores=torch.cat([(q*pos).sum(-1,keepdim=True),torch.einsum('bd,bkd->bk',q,neg)],1)/temperature
    return F.cross_entropy(scores,torch.zeros(len(q),dtype=torch.long,device=q.device))
