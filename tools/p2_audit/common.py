"""Shared native evaluation contracts; no independent ODE or label heuristics."""
import csv
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn


def read(path):
    with Path(path).open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def setup(checkpoint,csv_path,out,sigma=0.):
    from src.unet.io import load,eval_args,diffusion
    from src.eval.in_mem_eval import _get_cache
    from tools.u100_pretrain.eval_offline import rewrite
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    mapped=rewrite(csv_path,out/'eval.csv')
    model,config,ck=load(checkpoint,'cuda',use_ema=True)
    config=replace(config,eval_csv=str(mapped),std_soft_eval_sigma=sigma)
    args=eval_args(config);cache=_get_cache(str(mapped),len(read(mapped)),None,'',args)
    rows=read(mapped)
    train=read(config.data_csv)
    legal={}
    for row in train:legal.setdefault(row['script'],set()).add(int(row['calligrapher_id']))
    legal={s:sorted(w for w in ws if w<10) for s,ws in legal.items()}
    save(out/'identity.json',dict(checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),state='ema',
        eval_csv_sha256=sha(mapped),n=len(rows),sigma=sigma,seed=0,steps=config.eval_steps,
        sampler='native FlowMatching/Heun',time='t:1->0; model t*1000',legal_writer_rows=legal))
    return model,config,ck,cache,rows,legal,diffusion(config,evaluation=True)


class GuidedVelocity(nn.Module):
    std_soft_cond=True;use_glyph_cond=True;in_channels=4;out_channels=4
    def __init__(self,base,scale=1.,window=None):
        super().__init__();self.base=base;self.scale=float(scale);self.window=window
        self.branch_sample_evals=0;self.network_calls=0

    def forward(self,x,t,y_callig,y_char=None,g=None,y_script=None,**unused):
        conditional=self.base(x,t,y_callig,g=g,y_script=y_script)
        self.branch_sample_evals+=len(x);self.network_calls+=1
        active=self.scale!=1
        if active and self.window is not None:
            u=float(t[0])/1000.
            active=self.window[0]<=u<=self.window[1]
        if not active:return conditional
        null=self.base(x,t,torch.full_like(y_callig,self.base.num_writers),g=g,y_script=y_script)
        self.branch_sample_evals+=len(x);self.network_calls+=1
        return null+self.scale*(conditional-null)

    def forward_with_cfg(self,x,t,y_callig,y_char=None,cfg_scale=1.,g=None,y_script=None,**kwargs):
        return self(x,t,y_callig,y_char,g=g,y_script=y_script,**kwargs)


def paired_ci(values,chars,seed=90):
    values=np.asarray(values);chars=np.asarray(chars);groups=np.unique(chars)
    chunks=[values[chars==c] for c in groups];rng=np.random.default_rng(seed)
    boot=[np.concatenate([chunks[i] for i in rng.integers(len(chunks),size=len(chunks))]).mean() for _ in range(1000)]
    return dict(mean=float(values.mean()),ci95=np.quantile(boot,[.025,.975]).tolist(),n=len(values),n_chars=len(groups))
