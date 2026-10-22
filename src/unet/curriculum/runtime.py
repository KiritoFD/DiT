"""Adapter-only warmup -> low-LR joint tuning, reusing the common training loop."""
from ..data import StdBank


def is_adapter(name):return '.writer_film.' in name


def is_backbone(name,config):
    return not name.startswith(('writer.','writer_proj.')) and not is_adapter(name) and (
        '*' in config.finetune_backbone_patterns or name.startswith(tuple(config.finetune_backbone_patterns)))


def groups(model,config):
    adapter=[p for n,p in model.named_parameters() if is_adapter(n)]
    backbone=[p for n,p in model.named_parameters() if is_backbone(n,config)]
    if not adapter or not backbone:raise ValueError('Adaptation needs Writer branches AND a nonempty joint backbone subset')
    return [dict(params=adapter,lr=config.lr),dict(params=backbone,lr=config.lr*config.backbone_lr_scale)]


class Adaptation:
    def __init__(self,raw,ema,config,logger,start=0):
        self.raw=raw;self.ema=ema;self.config=config;self.logger=logger;self.joint=None
        self.bank=StdBank(config);self.set_phase(start)

    def set_phase(self,step):
        joint=step>=self.config.adapter_only_steps
        if joint==self.joint:return
        for name,p in self.raw.named_parameters():
            p.requires_grad_(is_adapter(name) or (joint and is_backbone(name,self.config)))
        # EMA caches frozen identities: refresh once at unfreeze or backbone EMA would copy raw forever.
        for name in ('_ema_pairs','_ema_fixed_pairs'):
            if hasattr(self.ema,name):delattr(self.ema,name)
        self.joint=joint
        if self.logger:self.logger.info('ADAPTATION phase=%s step=%d trainable=%d',
            'joint' if joint else 'writer_only',step,sum(p.numel() for p in self.raw.parameters() if p.requires_grad))

    def __call__(self,model,batch,kwargs,x0,step):
        self.set_phase(step);self.bank.transform(model,batch,kwargs,x0,step)
