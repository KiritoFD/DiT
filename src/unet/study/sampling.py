"""Training-only mild reweighting and legal Writer x Script strata, with pair coverage recorded."""
from collections import Counter,defaultdict
import numpy as np
import torch
from torch.utils.data import Sampler,DataLoader


class StudySampler(Sampler):
    def __init__(self,rows,config,steps,spec,start=0):
        self.rows=rows;self.batch=config.batch;self.steps=steps;self.seed=config.seed;self.epoch=0
        if start%steps:raise ValueError('Study sampler full-resume requires a checkpoint boundary')
        self.offset=start//steps;self.n_pair=2*int(self.batch*config.pair_fraction/2)
        self.mode=spec['sampling'];self.group=defaultdict(list);cells=defaultdict(lambda:defaultdict(list))
        for i,r in enumerate(rows):
            key=(r['calligrapher'],r['script']);self.group[key].append(i)
            cells[r['script'],r['character']][r['calligrapher']].append(i)
        self.keys=sorted(self.group);self.cells=defaultdict(list)
        for (script,char),writers in cells.items():
            if len(writers)>1:
                for writer,ids in writers.items():
                    other=[i for w,jj in writers.items() if w!=writer for i in jj]
                    self.cells[writer,script].append((ids,other))
        self.pair_keys=sorted(self.cells)
        if self.n_pair and not self.pair_keys:raise ValueError('No legal paired strata')
        self.weights=None
        if self.mode=='complexity':
            archive=np.load(spec['weights']);ids=archive['img_ids'];current=np.array([int(r['img_id']) for r in rows])
            if not np.array_equal(ids,current):raise ValueError('Complexity weights do not match training CSV order')
            self.weights=archive['weights'].astype(float)
            if self.weights.min()<1 or self.weights.max()>1.500001:raise ValueError('Difficulty weight cap violated')
            self.weights/=self.weights.sum()

    def __len__(self):return self.steps*self.batch
    def set_epoch(self,epoch):self.epoch=epoch
    def __iter__(self):
        rng=np.random.default_rng(self.seed+(self.epoch+self.offset)*1000003)
        for _ in range(self.steps):
            ids=[]
            for _ in range(self.n_pair//2):
                key=self.pair_keys[int(rng.integers(len(self.pair_keys)))]
                aa,bb=self.cells[key][int(rng.integers(len(self.cells[key])))]
                ids.extend([int(rng.choice(aa)),int(rng.choice(bb))])
            if self.mode=='writer_script':
                for _ in range(self.batch-len(ids)):
                    key=self.keys[int(rng.integers(len(self.keys)))];ids.append(int(rng.choice(self.group[key])))
            else:ids.extend(rng.choice(len(self.rows),self.batch-len(ids),p=self.weights).tolist())
            yield from ids


def study_loader(result,config,spec,start):
    dataset,_,_,tick,inline=result
    if spec['sampling']=='complexity':
        # Keep the original paired endpoint distribution; reweight only the ordinary share.
        from src.train.pair_rank import PairMixtureSampler
        sampler=StudySampler(dataset.samples,config,tick,spec,start)
        natural=PairMixtureSampler(dataset,config.batch,tick,config.seed,config.pair_fraction)
        class DifficultySampler(Sampler):
            def set_epoch(self,epoch):sampler.set_epoch(epoch);natural.set_epoch(epoch+sampler.offset)
            def __len__(self):return len(sampler)
            def __iter__(self):
                a=iter(sampler);b=iter(natural)
                for _ in range(tick):
                    left=[next(a) for _ in range(config.batch)];right=[next(b) for _ in range(config.batch)]
                    yield from right[:sampler.n_pair]+left[sampler.n_pair:]
        final=DifficultySampler()
    else:final=StudySampler(dataset.samples,config,tick,spec,start)
    batches=DataLoader(dataset,batch_size=config.batch,sampler=final,num_workers=config.workers,
        pin_memory=True,drop_last=True,persistent_workers=config.workers>0,prefetch_factor=4 if config.workers else None)
    return dataset,final,batches,tick,inline
