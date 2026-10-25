"""Small adapters for the existing condition-transform and joint-loss lifecycle."""
import numpy as np
import torch
import logging
import hashlib
from pathlib import Path
import json
from .data import ReferenceBank
from .encoder import supcon


class ReferenceTraining:
    def __init__(self, config, base):
        self.bank=ReferenceBank(config.reference_cache)
        manifest=json.loads(Path(config.reference_cache).with_suffix('.json').read_text())
        if hashlib.sha256(Path(config.data_csv).read_bytes()).hexdigest()!=manifest['csv_sha256']:
            raise ValueError('Reference feature cache belongs to a different training CSV')
        self.base=base; self.config=config
        self.stats={}

    def __call__(self, model, batch, kwargs, x0, step):
        self.step=step
        self.base(model,batch,kwargs,x0,step)
        positions=self.bank.positions(batch['img_id'].cpu().numpy())
        rng=np.random.default_rng(self.config.seed+step)
        refs,mask=self.bank.sample(positions,rng,self.config.reference_max_refs)
        kwargs['reference_features']=torch.as_tensor(self.bank.features[refs],device=x0.device)
        kwargs['reference_mask']=torch.as_tensor(mask,device=x0.device)

    def loss(self, model, batch):
        raw=getattr(model,'_orig_mod',model)
        p=self.bank.positions(batch['img_id'].cpu().numpy())
        device=next(raw.parameters()).device
        feature=raw.reference_head(torch.as_tensor(self.bank.features[p],device=device))
        values=[torch.as_tensor(field[p],device=device) for field in
                (self.bank.writers,self.bank.chars,self.bank.scripts)]
        loss,stats=supcon(feature,*values,temperature=self.config.reference_temperature,
                          hard_weight=self.config.reference_hard_weight)
        self.stats=dict(loss=loss.detach(),**stats)
        if self.step % self.config.log_every == 0:
            logging.getLogger('unet').info('REFERENCE step=%d supcon=%.6f weight=%g valid=%d hard=%d',
                self.step,float(loss.detach()),self.config.reference_con_weight,
                int(stats['valid_anchors']),int(stats['hard_negatives']))
        return self.config.reference_con_weight*loss


def bind_evaluation(model,config,bank=None):
    """Fixed training-only references from row metadata, never GT pixels or target features."""
    bank=bank or ReferenceBank(config.reference_cache)
    stored={}
    def provide(rows):
        key=tuple(int(row['img_id']) for row in rows)
        if key not in stored:
            ids=[];masks=[]
            for row in rows:
                writer=int(row['style_writer_id']);script={'楷':0,'行':1,'隶':2}[row['script']]
                pool=bank.pools[writer,script]
                pool=pool[bank.chars[pool]!=ord(row['character'])]
                if not len(pool):raise ValueError('No strictly cross-character training reference for eval target')
                rng=np.random.default_rng(config.seed+int(row['img_id']))
                chosen=rng.permutation(pool)
                _,first=np.unique(bank.chars[chosen],return_index=True)
                chosen=chosen[np.sort(first)][:config.reference_max_refs]
                ids.append(np.pad(chosen,(0,config.reference_max_refs-len(chosen)),mode='edge'))
                masks.append(np.arange(config.reference_max_refs)<len(chosen))
            stored[key]=dict(reference_features=torch.from_numpy(bank.features[np.asarray(ids)]),
                             reference_mask=torch.tensor(np.asarray(masks)))
        return stored[key]
    model.reference_eval_provider=provide
