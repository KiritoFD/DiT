"""Cached final-image latents and the existing offline STD bank."""
import json
from pathlib import Path
import torch
from src.utils.std_shape_aug import mixture


def remap(path):
    return {int(k):int(v) for k,v in json.loads(Path(path).read_text(encoding='utf-8')).items()}


def loader(config,args,logger=None,resume_start_step=0):
    maps={name:remap(getattr(config,f'{name}_remap_json')) for name in ('char','callig','font')}
    if set(maps['callig'].values())!=set(range(config.num_writers)):raise ValueError('Writer mapping mismatch')
    if set(maps['font'].values())!=set(range(config.num_scripts)):raise ValueError('Script mapping mismatch')
    from src.train.data_setup import build_data
    return build_data(args,rank=0,world_size=1,resume_start_step=resume_start_step,
                      std_soft_cond=True,char_remaps=maps,aux_dirs=[],logger=logger)


class StdBank:
    def __init__(self,config):
        self.mode=config.std_shape_aug
        self.bank=None
        if self.mode!='off':
            data=torch.load(config.std_shape_cache,map_location='cpu',weights_only=True,mmap=True)
            if data['version']!=2:raise ValueError('Offline bank version must be 2')
            self.bank=data['variants'].pin_memory()

    def __call__(self,batch,device,step,clean=None):
        if clean is None:clean=batch['std_soft'].to(device,non_blocking=True).float()
        if clean.ndim==3:clean=clean[:,None]
        if self.bank is None:return clean
        ids=batch['std_shape_index'].cpu()
        tiers=torch.multinomial(torch.tensor(mixture(step)),len(ids),replacement=True)
        if self.mode=='mild':tiers=tiers.clamp_max(1)
        variants=torch.randint(self.bank.shape[2],(len(ids),))
        selected=self.bank[ids,(tiers-1).clamp_min(0),variants].pin_memory()
        sampled=selected.to(device,non_blocking=True).float()
        return torch.where((tiers==0).to(device)[:,None,None,None],clean,sampled)

    def transform(self,model,batch,kwargs,x0,step):
        """Same condition-transform signature as StdShapeTraining, no DiT health probes."""
        kwargs['g']=self(batch,x0.device,step,clean=kwargs['g'])
