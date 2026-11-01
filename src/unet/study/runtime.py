"""Optional experiment adapter. Ordinary FM adds only CPU exposure accounting."""
from collections import Counter
import json
from pathlib import Path
import torch


class StudyRuntime:
    def __init__(self,config,run,transform,raw,logger):
        self.spec=json.loads(Path(config.study_spec).read_text())
        self.run=Path(run);self.previous=transform;self.config=config;self.logger=logger
        self.counts=Counter();self.groups=Counter();self.total=0
        self.needs_prediction=self.spec.get('objective','fm') in ('content','global_style','patch_style')
        self.objective_kind=self.spec.get('objective','fm')
        if self.objective_kind not in ('fm','writer_time') and not self.needs_prediction:
            raise ValueError('Unknown study objective')
        if self.needs_prediction:
            from .objectives import Auxiliary
            self.aux=Auxiliary(config,self.spec,logger)
        # Time-weighted arm must know the actual dropout result, not its probability.
        self.external_dropout=self.spec.get('external_dropout',False)
        if self.external_dropout:
            self.drop=(raw.drop_all,raw.drop_writer,raw.drop_script)
        logger.info('STUDY_SPEC %s',self.spec)

    def transform(self,model,batch,kwargs,z,step):
        self.previous(model,batch,kwargs,z,step)
        writers=batch['y_callig_raw'].tolist();scripts=batch['y_script'].tolist()
        self.counts.update(map(str,writers));self.groups.update(f'{w}:{s}' for w,s in zip(writers,scripts))
        self.total+=len(writers)
        if self.external_dropout:
            # Exactly the same three draws as native dropout; native masks are disabled.
            all_drop=torch.rand(len(z),device=z.device)<self.drop[0]
            wd=all_drop|(torch.rand(len(z),device=z.device)<self.drop[1])
            sd=all_drop|(torch.rand(len(z),device=z.device)<self.drop[2])
            kwargs['y_callig']=torch.where(wd,self.config.num_writers,kwargs['y_callig'])
            kwargs['y_script']=torch.where(sd,self.config.num_scripts,kwargs['y_script'])
            kwargs['condition_dropped']=True
        if step%100==0 or step+1==(self.config.stop_at_step or self.config.max_steps):
            from tools.three_day.contracts import save
            save(self.run/'exposure.json',dict(steps=step+1,examples=self.total,
                writer_counts=dict(self.counts),writer_script_counts=dict(self.groups),
                top10_examples=sum(v for k,v in self.counts.items() if int(k)<10),
                generic_null_examples=self.counts[str(self.config.num_writers)]))

    def objective(self,result,z,t,kwargs,batch,step):
        loss=result['loss']
        if self.objective_kind=='writer_time':
            weights=torch.where((t>.7)&(kwargs['y_callig']!=self.config.num_writers),1.5,1.)
            return (loss*weights).sum()/weights.sum()
        if self.needs_prediction:
            return loss.mean()+self.aux(result['pred_xstart'],z,t,kwargs,batch,step)
        return loss.mean()
