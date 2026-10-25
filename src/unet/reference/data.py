"""Training-only reference bank and same-script writer/character quartet sampler."""
from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import Sampler, DataLoader


class ReferenceBank:
    def __init__(self, path):
        data = np.load(path, allow_pickle=False)
        self.ids = data['ids']; self.features = data['features']
        self.writers = data['writers']; self.chars = data['chars']; self.scripts = data['scripts']
        if self.features.shape != (len(self.ids), 768) or not np.isfinite(self.features).all():
            raise ValueError('Expected finite DINOv2-B global features [N,768]')
        if len(np.unique(self.ids)) != len(self.ids):
            raise ValueError('Reference bank image IDs must be unique')
        self.lookup = {int(value): i for i, value in enumerate(self.ids)}
        self.pools = defaultdict(list)
        for i, key in enumerate(zip(self.writers, self.scripts)):
            self.pools[key].append(i)
        self.pools = {key: np.asarray(value) for key, value in self.pools.items()}

    def positions(self, ids):
        return np.asarray([self.lookup[int(value)] for value in ids])

    def sample(self, positions, rng, max_refs=2):
        result = np.empty((len(positions), max_refs), dtype=np.int64)
        mask = np.zeros_like(result, dtype=bool)
        for i, index in enumerate(positions):
            pool = self.pools[self.writers[index], self.scripts[index]]
            pool = pool[self.chars[pool] != self.chars[index]]
            # Strict exclusion of ALL images of the target character, not just target ID.
            if not len(pool):
                raise ValueError(f'No cross-character reference for image {self.ids[index]}')
            chosen = rng.permutation(pool)
            _, first = np.unique(self.chars[chosen], return_index=True)
            chosen = chosen[np.sort(first)]
            k = int(rng.integers(1, min(max_refs, len(chosen)) + 1))
            result[i] = chosen[0]; result[i, :k] = chosen[:k]; mask[i, :k] = True
        return result, mask


class QuartetSampler(Sampler):
    """Each quartet = Ax, Ay, Bx, By. Strict different writers/chars, same script."""
    def __init__(self, samples, bank, batch, steps, seed, start=0):
        if batch % 4:
            raise ValueError('Reference training batch must be a multiple of four')
        groups = defaultdict(lambda: defaultdict(list))
        for index, row in enumerate(samples):
            pos = bank.lookup[int(row['img_id'])]
            groups[bank.scripts[pos]][bank.writers[pos]].append((bank.chars[pos], index))
        self.tables = {}
        self.pairs = []
        self.by_script={}
        for script, writers in groups.items():
            names = sorted(writers)
            for writer, records in writers.items():
                table = defaultdict(list)
                for char, index in records: table[char].append(index)
                self.tables[script, writer] = table
            self.by_script[script]=[w for w in names if len(self.tables[script,w])>=2]
            for i, w in enumerate(names):
                for v in names[i+1:]:
                    common = sorted(self.tables[script, w].keys() & self.tables[script, v].keys())
                    if len(common) >= 2: self.pairs.append((script, w, v, common))
        if not self.pairs: raise ValueError('No valid 2x2 writer-character quartets')
        self.by_script={s:w for s,w in self.by_script.items() if len(w)>=2}
        self.batch=batch; self.steps=steps; self.seed=seed; self.start=start; self.epoch=0

    def set_epoch(self, epoch): self.epoch=epoch
    def __len__(self): return self.batch*self.steps

    def __iter__(self):
        rng=np.random.default_rng(self.seed+self.start+self.epoch)
        for q in range(self.steps*self.batch//4):
            # Half hard quartets, half writer-balanced cross-character pairs.
            # This exposes rare/non-shared characters rather than training only the intersection.
            if self.batch>=8 and q%(self.batch//4)>=(self.batch//4)//2:
                script=rng.choice(list(self.by_script))
                for writer in rng.choice(self.by_script[script],2,replace=False):
                    table=self.tables[script,writer]
                    for char in rng.choice(list(table),2,replace=False):
                        yield int(rng.choice(table[char]))
                continue
            script,w,v,common=self.pairs[int(rng.integers(len(self.pairs)))]
            x,y=rng.choice(common,2,replace=False)
            for writer,char in ((w,x),(w,y),(v,x),(v,y)):
                yield int(rng.choice(self.tables[script,writer][char]))


def reference_loader(result, config, start, bank=None):
    dataset,_,_,tick,inline=result
    bank=bank or ReferenceBank(config.reference_cache)
    sampler=QuartetSampler(dataset.samples,bank,config.batch,tick,config.seed,start)
    batches=DataLoader(dataset,batch_size=config.batch,sampler=sampler,num_workers=config.workers,
                       pin_memory=True,drop_last=True,persistent_workers=config.workers>0,
                       prefetch_factor=4 if config.workers else None)
    return dataset,sampler,batches,tick,inline
