"""Training-only GT attributes; cross-character Writer distributions at adaptation."""
import numpy as np
import torch


def sample_writer_attributes(cache,writer,script,seed):
    """Inference helper: sample only the frozen training bank, never a target image."""
    data=np.load(cache,allow_pickle=False)
    pool=np.flatnonzero(data['train']&data['valid']&(data['writers']==writer)&(data['scripts']==script))
    if not len(pool):return np.zeros(5,np.float32),False
    return data['normalized'][np.random.default_rng(seed).choice(pool)].copy(),True


class BrushTraining:
    def __init__(self,raw,ema,config,base):
        data=np.load(config.brush_cache,allow_pickle=False)
        self.ids=data['ids'];self.values=data['normalized'];self.valid=data['valid']
        self.chars=data['chars'];self.writers=data['writers'];self.scripts=data['scripts'];self.train=data['train']
        self.base=base;self.mode=config.writer_mode;self.pools={}
        defaults=np.zeros((config.num_writers+1,config.num_scripts+1,5),np.float32)
        for writer in range(config.num_writers):
            for script in range(config.num_scripts):
                indices=np.flatnonzero(self.train&self.valid&(self.writers==writer)&(self.scripts==script))
                if len(indices):
                    self.pools[writer,script]=indices
                    defaults[writer,script]=self.values[indices].mean(0)*len(indices)/(len(indices)+20)
        for model in (raw,ema):
            with torch.no_grad():model.brush_defaults.copy_(torch.from_numpy(defaults).to(model.brush_defaults))

    def __call__(self,model,batch,kwargs,x0,step):
        self.base(model,batch,kwargs,x0,step)
        ids=batch['img_id'].cpu().numpy();positions=np.searchsorted(self.ids,ids)
        if np.any(positions>=len(self.ids)) or not np.array_equal(self.ids[positions],ids):
            raise ValueError('Missing Brush attributes for batch')
        positions=positions.copy();valid=self.valid[positions].copy()
        if self.mode=='film':
            for i,index in enumerate(positions):
                writer=int(self.writers[index]);script=int(self.scripts[index])
                if writer<0:continue  # Generic training replay uses its own training-only attribute.
                pool=self.pools.get((writer,script),np.empty(0,dtype=int))
                pool=pool[self.chars[pool]!=self.chars[index]]
                if not len(pool):valid[i]=False;continue
                positions[i]=np.random.choice(pool);valid[i]=True
        kwargs['brush_attrs']=torch.from_numpy(self.values[positions]).to(x0.device,non_blocking=True)
        kwargs['brush_valid']=torch.from_numpy(valid).to(x0.device,non_blocking=True)
