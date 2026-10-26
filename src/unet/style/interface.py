"""Writer-isolated matrix U-Net. Legacy content weights stay a strict subtree."""
import torch
from torch import nn
from torch.nn import functional as F
from ..blocks import timestep_embedding
from .representations import Representation
from .injections import Injection


class MatrixUNet(nn.Module):
    std_soft_cond=True;use_glyph_cond=True;in_channels=4;out_channels=4

    def __init__(self,backbone,representation,injection,writers=10,vector_dim=32):
        super().__init__();self.backbone=backbone
        self.style_representation=Representation(representation,writers,vector_dim)
        channels=[stage.blocks[-1].conv2.out_channels for stage in backbone.decoder]
        self.style_injections=nn.ModuleList([Injection(injection,c,d,representation)
                                           for c,d in zip(channels,(64,128,192))])
        self.num_writers=backbone.num_writers;self.num_scripts=backbone.num_scripts
        self.drop_writer=backbone.drop_writer;self.style_enabled=True
        backbone.writer.requires_grad_(False);backbone.writer_proj.requires_grad_(False)

    @property
    def writer(self):return self.backbone.writer
    @property
    def script(self):return self.backbone.script

    def forward(self,x,t,y_callig,y_char=None,g=None,y_script=None,
                condition_dropped=False,drop_path_scale=None,**unused):
        if y_script is None or g is None:raise ValueError('STD and Script required')
        base=self.backbone;w=y_callig
        if self.training and not condition_dropped:
            all_drop=torch.rand(len(x),device=x.device)<base.drop_all
            wd=all_drop|(torch.rand(len(x),device=x.device)<self.drop_writer)
            sd=all_drop|(torch.rand(len(x),device=x.device)<base.drop_script)
            w=torch.where(wd,self.num_writers,w)
            y_script=torch.where(sd,self.num_scripts,y_script)
        # The OLD route always reads the SAME null row. It never sees w.
        old_writer=base.writer(torch.full_like(w,self.num_writers))
        conds=base._compute_stage_conditions(timestep_embedding(t),old_writer,base.script(y_script))
        glyph=base.std_stem(g);pyramid=[glyph]
        for down in base.std_down:pyramid.append(base.act(down(pyramid[-1])))
        state=self.style_representation(w,pyramid,g) if self.style_enabled else None
        h=base.input(x);skips=[]
        for i,stage in enumerate(base.encoder):
            h=stage(h+base.std_proj[i](pyramid[i]),conds[i],old_writer,drop_path_scale)
            skips.append(h)
            if i<2:h=base.down[i](h)
        h=base.middle_attn(base.middle1(h,conds[2],old_writer,drop_path_scale))
        ctx=(self.style_injections[2],state,pyramid[2],2) if state is not None else None
        h=base.middle2(h,conds[2],old_writer,drop_path_scale,style_context=ctx)
        for i in (2,1,0):
            ctx=(self.style_injections[i],state,pyramid[i],i) if state is not None and i<2 else None
            h=base.decoder[i](torch.cat((h,skips[i]),1),conds[i],old_writer,
                              drop_path_scale,style_context=ctx)
            if i:h=base.up[i-1](F.interpolate(h,scale_factor=2,mode='nearest'))
        return base.output(h)

    def forward_with_cfg(self,x,t,y_callig,y_char=None,cfg_scale=1.,**kwargs):
        if self.training:raise RuntimeError('Sampling requires eval')
        cond=self(x,t,y_callig,y_char,**kwargs)
        if cfg_scale==1:return cond
        null=self(x,t,torch.full_like(y_callig,self.num_writers),y_char,**kwargs)
        return null+float(cfg_scale)*(cond-null)
