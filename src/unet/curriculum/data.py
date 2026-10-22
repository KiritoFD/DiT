"""Replay is an explicit probability, not the raw ratio of two CSV sizes."""
import torch
from torch.utils.data import DataLoader,Sampler


class ReplaySampler(Sampler):
    def __init__(self,samples,batch,steps,fraction,seed,start=0):
        generic=torch.tensor([str(r.get('generic','0'))=='1' for r in samples])
        n=int(generic.sum());m=len(samples)-n
        if not n or not m or not 0<fraction<1:raise ValueError('Replay requires both generic and labelled pools')
        self.weights=torch.where(generic,fraction/n,(1-fraction)/m).double()
        self.length=batch*steps;self.seed=seed;self.epoch=0;self.offset=start//steps

    def set_epoch(self,epoch):self.epoch=epoch
    def __len__(self):return self.length
    def __iter__(self):
        gen=torch.Generator().manual_seed(self.seed+self.offset+self.epoch)
        return iter(torch.multinomial(self.weights,self.length,replacement=True,generator=gen).tolist())


def replay_loader(result,config,start):
    dataset,_,_,tick,inline=result
    sampler=ReplaySampler(dataset.samples,config.batch,tick,config.generic_replay_fraction,config.seed,start)
    loader=DataLoader(dataset,batch_size=config.batch,sampler=sampler,num_workers=config.workers,
        pin_memory=True,drop_last=True,persistent_workers=config.workers>0,
        prefetch_factor=4 if config.workers else None)
    return dataset,sampler,loader,tick,inline
