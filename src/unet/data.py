"""Cached final-image latents and the existing offline STD bank."""
import json
from pathlib import Path
import torch
from src.utils.std_shape_aug import mixture


def remap(path):
    return {int(k):int(v) for k,v in json.loads(Path(path).read_text(encoding='utf-8')).items()}


def loader(config,args,logger=None,resume_start_step=0):
    maps={name:remap(getattr(config,f'{name}_remap_json')) for name in ('char','callig','font')}
    real={k:v for k,v in maps['callig'].items() if v!=config.num_writers}
    if set(real.values())!=set(range(config.num_writers)):raise ValueError('Writer mapping mismatch')
    if config.writer_mode=='shared' and len(real)!=len(maps['callig']) and not getattr(config,'null_writer_rows',False):
        raise ValueError('Legacy route cannot map generic replay to null')
    if any(not 0<=v<=config.num_writers for v in maps['callig'].values()):raise ValueError('Invalid writer row')
    if set(maps['font'].values())!=set(range(config.num_scripts)):raise ValueError('Script mapping mismatch')
    if config.writer_mode=='film':
        import csv
        with open(config.data_csv,encoding='utf-8-sig') as f:
            for row in csv.DictReader(f):
                if (row.get('generic','0')=='1')!=(maps['callig'][int(row['calligrapher_id'])]==config.num_writers):
                    raise ValueError('Generic replay flag and hard-null writer mapping disagree')
    from src.train.data_setup import build_data
    result=build_data(args,rank=0,world_size=1,resume_start_step=resume_start_step,
                      std_soft_cond=True,char_remaps=maps,aux_dirs=[],logger=logger)
    if config.reference_dim:
        from .reference.data import reference_loader
        if getattr(config,'reference_extra_cache',''):
            from .reference.group_runtime import MergedBank
            bank=MergedBank(config.reference_cache,config.reference_extra_cache)
            return reference_loader(result,config,resume_start_step,bank=bank)
        return reference_loader(result,config,resume_start_step)
    if config.generic_replay_fraction:
        from .curriculum.data import replay_loader
        return replay_loader(result,config,resume_start_step)
    if config.brush_dim and config.writer_mode=='none':
        import copy
        from .curriculum.data import replay_loader
        mixture=copy.copy(config);mixture.generic_replay_fraction=config.brush_generic_fraction
        return replay_loader(result,mixture,resume_start_step)
    return result


class StdBank:
    def __init__(self,config):
        self.mode=config.std_shape_aug
        self.drop_path_max=config.drop_path_max
        self.drop_path_start=config.drop_path_start;self.drop_path_end=config.drop_path_end
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
        if self.drop_path_max:
            from .variants.regularization import schedule_scale
            kwargs['drop_path_scale']=x0.new_tensor(schedule_scale(step,self.drop_path_start,self.drop_path_end))
