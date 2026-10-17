"""Train-only 2x2 writer/character blocks; fallback coverage is explicit."""
from collections import defaultdict
import numpy as np


class WriterQuartets:
    def __init__(self,train,slots,scripts,chars,sources,seed=89):
        self.rng=np.random.default_rng(seed);self.chars=np.asarray(chars);self.sources=np.asarray(sources)
        self.slots=np.asarray(slots);self.scripts=np.asarray(scripts)
        self.groups=defaultdict(lambda:defaultdict(list))
        self.slot_scripts={}
        for i in train:
            slot=int(slots[i]);self.groups[slot][str(chars[i])].append(int(i));self.slot_scripts[slot]=int(scripts[i])
        self.full=[];self.fallback=[]
        keys=sorted(self.groups)
        for i,a in enumerate(keys):
            for b in keys[i+1:]:
                if self.slot_scripts[a]!=self.slot_scripts[b]:continue
                shared=sorted(set(self.groups[a])&set(self.groups[b]))
                if len(shared)>=2:self.full.append((a,b,shared))
                elif shared and len(self.groups[a])>=2 and len(self.groups[b])>=2:self.fallback.append((a,b,shared))
        if not self.full and not self.fallback:raise ValueError('No legal training writer-character blocks')
        self.coverage={'full_writer_pairs':len(self.full),'fallback_writer_pairs':len(self.fallback),
            'unique_metadata_rectangles':int(sum(len(cs)*(len(cs)-1)//2 for _,_,cs in self.full)),
            'full_pairs':[{'a':a,'b':b,'shared_characters':len(cs)} for a,b,cs in self.full],
            'slots_without_full_rectangles':sorted(set(keys)-{x for a,b,_ in self.full for x in (a,b)})}

    def sample(self,batch):
        if batch%4:raise ValueError('Batch must be divisible by four')
        blocks=[];full=0;cross=0
        def pick(values,avoid_source=None):
            preferred=[i for i in values if self.sources[i]!=avoid_source] if avoid_source is not None else []
            pool=preferred or values
            return int(pool[int(self.rng.integers(len(pool)))])
        for _ in range(batch//4):
            # 90% full blocks when fallback writer pairs exist, otherwise 100%.
            use_full=bool(self.full) and (not self.fallback or self.rng.random()<.9)
            pool=self.full if use_full else self.fallback
            a,b,shared=pool[int(self.rng.integers(len(pool)))]
            if use_full:
                c1,c2=self.rng.choice(shared,2,replace=False);d2=c2;full+=1
            else:
                c1=shared[int(self.rng.integers(len(shared)))];c2=self.rng.choice([c for c in self.groups[a] if c!=c1]);d2=self.rng.choice([c for c in self.groups[b] if c!=c1])
            ac=pick(self.groups[a][c1]);ad=pick(self.groups[a][c2],self.sources[ac])
            bc=pick(self.groups[b][c1]);bd=pick(self.groups[b][d2],self.sources[bc])
            blocks.extend([ac,ad,bc,bd]);cross+=2*int(self.sources[ac]!=self.sources[ad])+2*int(self.sources[bc]!=self.sources[bd])
        ids=np.array(blocks);return ids,{'full_quartet_fraction':full/(batch//4),'cross_source_positive_fraction':cross/batch}


def cross_loss(embedding,chars,margin=.1,tau=.1):
    import torch
    from torch.nn import functional as F
    b=len(embedding);base=torch.arange(b,device=embedding.device).reshape(-1,4)
    positive=base[:,[1,0,3,2]].reshape(-1);same=base[:,[2,3,0,1]].reshape(-1);different=base[:,[3,2,1,0]].reshape(-1)
    c=np.asarray(chars);hard=torch.as_tensor(c==c[same.cpu().numpy()],device=embedding.device)
    diff=torch.as_tensor(c!=c[different.cpu().numpy()],device=embedding.device)
    ep=(embedding*embedding[positive]).sum(-1)
    hn=F.softplus(((embedding*embedding[same]).sum(-1)-ep+margin)/tau)
    dn=F.softplus(((embedding*embedding[different]).sum(-1)-ep+margin)/tau)
    # Missing hard pairs are not silently relabeled as same-character negatives.
    h=hn[hard].mean() if hard.any() else embedding.sum()*0
    dd=dn[diff].mean() if diff.any() else embedding.sum()*0
    return .5*(h+dd),{'same_char_hard_fraction':float(hard.float().mean()),'cross_hard':float(h.detach()),'cross_diff':float(dd.detach())}
