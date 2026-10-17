"""Direct 4x32x32 latent CNN U-Net, not a skeleton predictor or DiT wrapper."""
import torch
from torch import nn
from torch.nn import functional as F
from .blocks import Stage, ResBlock, Attention, norm, timestep_embedding


class LatentUNet(nn.Module):
    std_soft_cond=True
    use_glyph_cond=True
    in_channels=4
    out_channels=4

    def __init__(self,channels=(128,256,384),num_writers=10,num_scripts=3,
                 drop_all=.05,drop_writer=.08,drop_script=.08):
        super().__init__()
        if len(channels)!=3:raise ValueError('Expected three scales')
        self.num_writers=num_writers;self.num_scripts=num_scripts
        self.drop_all=drop_all;self.drop_writer=drop_writer;self.drop_script=drop_script
        self.writer=nn.Embedding(num_writers+1,32)
        self.script=nn.Embedding(num_scripts+1,16)
        self.time_proj=nn.Linear(128,512)
        self.writer_proj=nn.Linear(32,512)
        self.script_proj=nn.Linear(16,512)
        self.condition=nn.Sequential(nn.SiLU(),nn.Linear(512,512))
        self.std_stem=nn.Sequential(nn.Conv2d(1,64,3,stride=2,padding=1),nn.SiLU(),nn.Conv2d(64,64,3,padding=1))
        self.std_down=nn.ModuleList([nn.Conv2d(64,128,3,stride=2,padding=1),
                                     nn.Conv2d(128,192,3,stride=2,padding=1)])
        self.std_proj=nn.ModuleList([nn.Conv2d(s,c,1) for s,c in zip((64,128,192),channels)])
        self.input=nn.Conv2d(4,channels[0],3,padding=1)
        self.encoder=nn.ModuleList([Stage(c,c,i>0) for i,c in enumerate(channels)])
        self.down=nn.ModuleList([nn.Conv2d(channels[i],channels[i+1],3,stride=2,padding=1) for i in range(2)])
        self.middle1=ResBlock(channels[-1],channels[-1]);self.middle_attn=Attention(channels[-1])
        self.middle2=ResBlock(channels[-1],channels[-1])
        self.decoder=nn.ModuleList([Stage(c*2,c,i>0) for i,c in enumerate(channels)])
        self.up=nn.ModuleList([nn.Conv2d(channels[i+1],channels[i],3,padding=1) for i in range(2)])
        self.output=nn.Sequential(norm(channels[0]),nn.SiLU(),nn.Conv2d(channels[0],4,3,padding=1))
        nn.init.zeros_(self.output[-1].weight);nn.init.zeros_(self.output[-1].bias)

    def forward(self,x,t,y_callig,y_char=None,g=None,y_script=None,y_callig_raw=None,y_pair=None):
        # y_char/raw/pair are accepted only for the shared sampler interface.
        # Character IDs are never embedded or used as an inference condition.
        if x.ndim!=4 or x.shape[1:]!=(4,32,32):raise ValueError('Expected latent [B,4,32,32]')
        if y_script is None:raise ValueError('Script condition required')
        if g is None:g=x.new_zeros((len(x),1,64,64))
        if g.shape!=(len(x),1,64,64):raise ValueError('Expected STD [B,1,64,64]')
        if self.training:
            all_drop=torch.rand(len(x),device=x.device)<self.drop_all
            wd=all_drop|(torch.rand(len(x),device=x.device)<self.drop_writer)
            sd=all_drop|(torch.rand(len(x),device=x.device)<self.drop_script)
            y_callig=torch.where(wd,self.num_writers,y_callig)
            y_script=torch.where(sd,self.num_scripts,y_script)
        # Match the DiT route: all/style dropout never blanks the STD spatial input.
        c=self.condition(self.time_proj(timestep_embedding(t))+
                         self.writer_proj(self.writer(y_callig))+
                         self.script_proj(self.script(y_script)))
        glyph=self.std_stem(g);h=self.input(x);skips=[]
        for i,stage in enumerate(self.encoder):
            if i:glyph=F.silu(self.std_down[i-1](glyph))
            h=stage(h+self.std_proj[i](glyph),c);skips.append(h)
            if i<2:h=self.down[i](h)
        h=self.middle2(self.middle_attn(self.middle1(h,c)),c)
        for i in (2,1,0):
            h=self.decoder[i](torch.cat([h,skips[i]],dim=1),c)
            if i:h=self.up[i-1](F.interpolate(h,scale_factor=2,mode='nearest'))
        return self.output(h)

    def forward_with_cfg(self,x,t,y_callig,y_char=None,cfg_scale=1.,g=None,y_script=None,**unused):
        if self.training:raise RuntimeError('CFG sampling requires eval mode')
        conditional=self(x,t,y_callig,y_char,g=g,y_script=y_script)
        if cfg_scale==1:return conditional
        unconditional=self(x,t,torch.full_like(y_callig,self.num_writers),
                           g=g,y_script=y_script)
        return unconditional+float(cfg_scale)*(conditional-unconditional)
