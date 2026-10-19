"""Direct 4x32x32 latent CNN U-Net, not a skeleton predictor or DiT wrapper."""
import torch
from torch import nn
from torch.nn import functional as F
from .blocks import Stage, ResBlock, Attention, norm, timestep_embedding, get_activation
from .variants.modulation import bounded_gain, zero_linear


class LatentUNet(nn.Module):
    std_soft_cond=True
    use_glyph_cond=True
    in_channels=4
    out_channels=4

    def __init__(self,channels=(128,256,384),num_writers=10,num_scripts=3,
                 num_res_blocks=2,drop_all=.05,drop_writer=.08,drop_script=.08,
                 decoder_res_blocks=None,skip_gate_scales=(),std_gate_scales=(),writer_film_rank=0,
                 writer_film_layers=(),residual_cond_gate=False,stage_specific_cond=False,
                 stage_cond_mode='adapter',act_fn='silu'):
        super().__init__()
        if len(channels)!=3:raise ValueError('Expected three scales')
        if not isinstance(writer_film_rank,int) or writer_film_rank<0:
            raise ValueError('writer_film_rank must be a nonnegative integer')
        if not isinstance(residual_cond_gate,bool):raise ValueError('residual_cond_gate must be boolean')
        if not isinstance(stage_specific_cond,bool):raise ValueError('stage_specific_cond must be boolean')
        if stage_cond_mode not in ('adapter','independent'):raise ValueError(f'Invalid stage_cond_mode: {stage_cond_mode}')
        if act_fn not in ('silu','gelu'):raise ValueError(f'Invalid act_fn: {act_fn}')
        self.num_writers=num_writers;self.num_scripts=num_scripts
        self.num_res_blocks=num_res_blocks
        self.stage_specific_cond=stage_specific_cond
        self.stage_cond_mode=stage_cond_mode
        self.act_fn=act_fn
        self.act=get_activation(act_fn)
        act_layer=nn.SiLU if act_fn=='silu' else nn.GELU

        decoder_depth=tuple(decoder_res_blocks) if decoder_res_blocks is not None else (num_res_blocks,)*3
        if len(decoder_depth)!=3 or min(decoder_depth)<1:raise ValueError('Expected three positive decoder depths')
        self.skip_gates=self._scale_parameters(skip_gate_scales)
        self.std_gates=self._scale_parameters(std_gate_scales)
        # Explicit, nonpersistent OOD interventions; no effect in normal training.
        self.inference_skip_scales=(1.,1.,1.)
        self.inference_std_scales=(1.,1.,1.)
        self.drop_all=drop_all;self.drop_writer=drop_writer;self.drop_script=drop_script

        self.writer=nn.Embedding(num_writers+1,32)
        self.script=nn.Embedding(num_scripts+1,16)

        if not stage_specific_cond or stage_cond_mode=='adapter':
            self.time_proj=nn.Linear(128,512)
            self.writer_proj=nn.Linear(32,512)
            self.script_proj=nn.Linear(16,512)
            self.condition=nn.Sequential(act_layer(),nn.Linear(512,512))
            if stage_specific_cond:
                self.stage_adapters=nn.ModuleList([zero_linear(512,512) for _ in range(3)])
        else:
            self.stage_time_proj=nn.ModuleList([nn.Linear(128,512) for _ in range(3)])
            self.stage_writer_proj=nn.ModuleList([nn.Linear(32,512) for _ in range(3)])
            self.stage_script_proj=nn.ModuleList([nn.Linear(16,512) for _ in range(3)])
            self.stage_condition=nn.ModuleList([nn.Sequential(act_layer(),nn.Linear(512,512)) for _ in range(3)])

        self.std_stem=nn.Sequential(nn.Conv2d(1,64,3,stride=2,padding=1),act_layer(),nn.Conv2d(64,64,3,padding=1))
        self.std_down=nn.ModuleList([nn.Conv2d(64,128,3,stride=2,padding=1),
                                     nn.Conv2d(128,192,3,stride=2,padding=1)])
        self.std_proj=nn.ModuleList([nn.Conv2d(s,c,1) for s,c in zip((64,128,192),channels)])
        self.input=nn.Conv2d(4,channels[0],3,padding=1)
        self.encoder=nn.ModuleList([Stage(c,c,i>0,num_res_blocks=num_res_blocks,act_fn=act_fn) for i,c in enumerate(channels)])
        self.down=nn.ModuleList([nn.Conv2d(channels[i],channels[i+1],3,stride=2,padding=1) for i in range(2)])
        self.middle1=ResBlock(channels[-1],channels[-1],act_fn=act_fn);self.middle_attn=Attention(channels[-1])
        self.middle2=ResBlock(channels[-1],channels[-1],act_fn=act_fn)
        self.decoder=nn.ModuleList([Stage(c*2,c,i>0,num_res_blocks=decoder_depth[i],act_fn=act_fn) for i,c in enumerate(channels)])
        self.up=nn.ModuleList([nn.Conv2d(channels[i+1],channels[i],3,padding=1) for i in range(2)])
        self.output=nn.Sequential(norm(channels[0]),act_layer(),nn.Conv2d(channels[0],4,3,padding=1))
        nn.init.zeros_(self.output[-1].weight);nn.init.zeros_(self.output[-1].bias)
        self._configure_modulation(writer_film_rank,residual_cond_gate,writer_film_layers)

    def _configure_modulation(self,writer_rank,residual_gate,writer_film_layers=()):
        # Build optional modules last, preserving historical keys and initialization.
        if writer_rank:
            patterns=tuple(writer_film_layers) if writer_film_layers else ()
            for name,block in self.named_modules():
                if isinstance(block,ResBlock):
                    if not patterns or any(pat in name for pat in patterns):
                        block.enable_writer_film(writer_rank)
        if residual_gate:
            blocks=[block for block in self.modules() if isinstance(block,ResBlock)]
            for block in blocks:block.enable_residual_gate()
    @staticmethod
    def _scale_parameters(active):
        active=tuple(active)
        if any(i not in (0,1,2) for i in active):raise ValueError('Scale indices are 0:32,1:16,2:8')
        # No parameters added for a baseline: its old state_dict remains strict-loadable.
        return nn.ParameterList([nn.Parameter(torch.zeros(()),requires_grad=i in active) for i in range(3)]) if active else nn.ParameterList()

    def set_inference_scales(self,*,skip=(1.,1.,1.),std=(1.,1.,1.)):
        if self.training:raise RuntimeError('Inference interventions require eval mode')
        if len(skip)!=3 or len(std)!=3 or any(not 0<float(v)<float('inf') for v in (*skip,*std)):
            raise ValueError('Expected three finite positive scales')
        self.inference_skip_scales=tuple(map(float,skip))
        self.inference_std_scales=tuple(map(float,std))

    @staticmethod
    def _scaled(value,gates,i,override):
        if gates:value=value*bounded_gain(gates[i])
        return value*override[i] if override[i]!=1. else value

    def _compute_stage_conditions(self,t_emb,writer,script):
        if not self.stage_specific_cond:
            c=self.condition(self.time_proj(t_emb)+self.writer_proj(writer)+self.script_proj(script))
            return (c,c,c)
        if self.stage_cond_mode=='adapter':
            c=self.condition(self.time_proj(t_emb)+self.writer_proj(writer)+self.script_proj(script))
            return tuple(c+adapter(c) for adapter in self.stage_adapters)
        return tuple(cond(tp(t_emb)+wp(writer)+sp(script)) for tp,wp,sp,cond in
                     zip(self.stage_time_proj,self.stage_writer_proj,self.stage_script_proj,self.stage_condition))

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
        writer=self.writer(y_callig)
        script=self.script(y_script)
        t_emb=timestep_embedding(t)
        conds=self._compute_stage_conditions(t_emb,writer,script)

        glyph=self.std_stem(g);h=self.input(x);skips=[]
        for i,stage in enumerate(self.encoder):
            if i:glyph=self.act(self.std_down[i-1](glyph))
            projected=self._scaled(self.std_proj[i](glyph),self.std_gates,i,self.inference_std_scales)
            h=stage(h+projected,conds[i],writer);skips.append(h)
            if i<2:h=self.down[i](h)
        h=self.middle2(self.middle_attn(self.middle1(h,conds[2],writer)),conds[2],writer)
        for i in (2,1,0):
            skip=self._scaled(skips[i],self.skip_gates,i,self.inference_skip_scales)
            h=self.decoder[i](torch.cat([h,skip],dim=1),conds[i],writer)
            if i:h=self.up[i-1](F.interpolate(h,scale_factor=2,mode='nearest'))
        return self.output(h)

    def forward_with_cfg(self,x,t,y_callig,y_char=None,cfg_scale=1.,g=None,y_script=None,**unused):
        if self.training:raise RuntimeError('CFG sampling requires eval mode')
        conditional=self(x,t,y_callig,y_char,g=g,y_script=y_script)
        if cfg_scale==1:return conditional
        unconditional=self(x,t,torch.full_like(y_callig,self.num_writers),
                           g=g,y_script=y_script)
        return unconditional+float(cfg_scale)*(conditional-unconditional)
