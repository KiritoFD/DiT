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
                 writer_film_layers=(),writer_film_spec=None,decouple_writer_condition=False,
                 decoder_std_residual=False,decoder_std_scale=0.2,residual_cond_gate=False,
                 stage_specific_cond=False,stage_cond_mode='adapter',act_fn='silu',drop_path_max=0.,writer_mode='shared',
                 brush_dim=0,brush_dropout=.1,brush_noise=.075,reference_dim=0,
                 reference_injection='add',reference_dropout=0.):
        super().__init__()
        if len(channels)!=3:raise ValueError('Expected three scales')
        if not isinstance(writer_film_rank,int) or writer_film_rank<0:
            raise ValueError('writer_film_rank must be a nonnegative integer')
        if not isinstance(residual_cond_gate,bool):raise ValueError('residual_cond_gate must be boolean')
        if not isinstance(stage_specific_cond,bool):raise ValueError('stage_specific_cond must be boolean')
        if stage_cond_mode not in ('adapter','independent'):raise ValueError(f'Invalid stage_cond_mode: {stage_cond_mode}')
        if act_fn not in ('silu','gelu'):raise ValueError(f'Invalid act_fn: {act_fn}')
        self.num_writers=num_writers;self.num_scripts=num_scripts
        self.reference_dim=reference_dim
        self.reference_injection=reference_injection
        self.reference_dropout=reference_dropout
        if reference_dim:
            if reference_injection not in ('add','writer_slot'):
                raise ValueError('reference_injection must be add or writer_slot')
            if writer_mode not in ('none','shared') or brush_dim:
                raise ValueError('Reference route requires writer_mode none/shared (writer_slot) or none, and no Brush condition')
            from .reference.encoder import StyleHead
            self.reference_head=StyleHead(style_dim=reference_dim)
            if reference_injection=='writer_slot':
                # Continuous style rides the same FiLM pathway the discrete-ID line validated.
                self.reference_bridge=nn.Linear(reference_dim,32)
            else:
                self.reference_proj=nn.Linear(reference_dim,512)
        self.writer_mode=writer_mode
        self.brush_dim=brush_dim;self.brush_dropout=brush_dropout;self.brush_noise=brush_noise
        self.num_res_blocks=num_res_blocks
        self.stage_specific_cond=stage_specific_cond
        self.stage_cond_mode=stage_cond_mode
        self.decouple_writer_condition=decouple_writer_condition
        self.decoder_std_residual=decoder_std_residual
        self.decoder_std_scale=float(decoder_std_scale)
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
        if decoder_std_residual:
            self.dec_std_proj=nn.ModuleList([nn.Conv2d(64,channels[0],1),nn.Conv2d(128,channels[1],1)])
            for p in self.dec_std_proj:
                nn.init.zeros_(p.weight);nn.init.zeros_(p.bias)
        self.input=nn.Conv2d(4,channels[0],3,padding=1)
        self.encoder=nn.ModuleList([Stage(c,c,i>0,num_res_blocks=num_res_blocks,act_fn=act_fn) for i,c in enumerate(channels)])
        self.down=nn.ModuleList([nn.Conv2d(channels[i],channels[i+1],3,stride=2,padding=1) for i in range(2)])
        self.middle1=ResBlock(channels[-1],channels[-1],act_fn=act_fn);self.middle_attn=Attention(channels[-1])
        self.middle2=ResBlock(channels[-1],channels[-1],act_fn=act_fn)
        self.decoder=nn.ModuleList([Stage(c*2,c,i>0,num_res_blocks=decoder_depth[i],act_fn=act_fn) for i,c in enumerate(channels)])
        self.up=nn.ModuleList([nn.Conv2d(channels[i+1],channels[i],3,padding=1) for i in range(2)])
        self.output=nn.Sequential(norm(channels[0]),act_layer(),nn.Conv2d(channels[0],4,3,padding=1))
        nn.init.zeros_(self.output[-1].weight);nn.init.zeros_(self.output[-1].bias)
        self._configure_modulation(writer_film_rank,residual_cond_gate,writer_film_layers,writer_film_spec)
        if brush_dim:
            self.brush_mlp=nn.Sequential(nn.Linear(brush_dim,128),nn.SiLU(),nn.Linear(128,256))
            self.register_buffer('brush_defaults',torch.zeros(num_writers+1,num_scripts+1,brush_dim))
            for name,block in self.named_modules():
                if isinstance(block,ResBlock) and name.startswith(('middle','decoder.')):
                    block.brush_film=zero_linear(256,2*block.conv2.in_channels)
        blocks=[b for stage in self.encoder for b in stage.blocks]
        blocks += [self.middle1,self.middle2]
        blocks += [b for i in (2,1,0) for b in self.decoder[i].blocks]
        for i,block in enumerate(blocks):block.drop_path_rate=drop_path_max*i/max(1,len(blocks)-1)

    def _configure_modulation(self,writer_rank,residual_gate,writer_film_layers=(),writer_film_spec=None):
        # Build optional modules last, preserving historical keys and initialization.
        if writer_rank:
            if writer_film_spec:
                for name,block in self.named_modules():
                    if isinstance(block,ResBlock):
                        matched_amp=None
                        for pat,amp in sorted(writer_film_spec.items(),key=lambda kv:len(kv[0]),reverse=True):
                            if pat in name:
                                matched_amp=amp;break
                        if matched_amp is not None:
                            block.enable_writer_film(writer_rank,max_amp=matched_amp)
            else:
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
        wp = 0 if self.decouple_writer_condition else self.writer_proj(writer)
        if not self.stage_specific_cond:
            c=self.condition(self.time_proj(t_emb)+wp+self.script_proj(script))
            return (c,c,c)
        if self.stage_cond_mode=='adapter':
            c=self.condition(self.time_proj(t_emb)+wp+self.script_proj(script))
            return tuple(c+adapter(c) for adapter in self.stage_adapters)
        return tuple(cond(tp(t_emb)+(0 if self.decouple_writer_condition else swp(writer))+sp(script)) for tp,swp,sp,cond in
                     zip(self.stage_time_proj,self.stage_writer_proj,self.stage_script_proj,self.stage_condition))

    def forward(self,x,t,y_callig,y_char=None,g=None,y_script=None,y_callig_raw=None,y_pair=None,drop_path_scale=None,
                brush_attrs=None,brush_valid=None,reference_features=None,reference_mask=None):
        # y_char/raw/pair are accepted only for the shared sampler interface.
        # Character IDs are never embedded or used as an inference condition.
        if x.ndim!=4 or x.shape[1:]!=(4,32,32):raise ValueError('Expected latent [B,4,32,32]')
        if y_script is None:raise ValueError('Script condition required')
        if g is None:g=x.new_zeros((len(x),1,64,64))
        if g.shape!=(len(x),1,64,64):raise ValueError('Expected STD [B,1,64,64]')
        brush=None
        if self.brush_dim:
            if brush_attrs is None:
                ids=torch.full_like(y_script,self.num_writers) if self.writer_mode=='none' else y_callig
                brush_attrs=self.brush_defaults[ids,y_script]
            if brush_valid is None:brush_valid=torch.ones(len(x),device=x.device,dtype=torch.bool)
            if self.training:
                brush_attrs=brush_attrs+torch.randn_like(brush_attrs)*self.brush_noise
                brush_valid=brush_valid&(torch.rand(len(x),device=x.device)>=self.brush_dropout)
            brush=(self.brush_mlp(brush_attrs),brush_valid)
        if self.training:
            all_drop=torch.rand(len(x),device=x.device)<self.drop_all
            wd=all_drop|(torch.rand(len(x),device=x.device)<self.drop_writer)
            sd=all_drop|(torch.rand(len(x),device=x.device)<self.drop_script)
            if self.writer_mode!='none':y_callig=torch.where(wd,self.num_writers,y_callig)
            y_script=torch.where(sd,self.num_scripts,y_script)
        # Match the DiT route: all/style dropout never blanks the STD spatial input.
        writer=(x.new_zeros((len(x),32)) if self.writer_mode=='none' else self.writer(y_callig))
        writer_mask=(y_callig!=self.num_writers) if self.writer_mode=='film' else None
        script=self.script(y_script)
        t_emb=timestep_embedding(t)
        style=None
        if self.reference_dim:
            if reference_features is None:raise ValueError('Reference model requires cross-character reference features')
            style=self.reference_head.aggregate(reference_features,reference_mask)
            if self.training and self.reference_dropout>0:
                # Zeroed aggregate is the learned null style (bridge bias absorbs the null slot).
                keep=torch.rand(len(x),device=x.device)>=self.reference_dropout
                style=style*keep[:,None]
            if self.reference_injection=='writer_slot':
                writer=self.reference_bridge(style)
        conds=self._compute_stage_conditions(t_emb,writer,script)
        if style is not None and self.reference_injection!='writer_slot':
            conds=tuple(c+self.reference_proj(style) for c in conds)

        glyph=self.std_stem(g);h=self.input(x);skips=[]
        std_glyphs=[glyph]
        for i,stage in enumerate(self.encoder):
            if i:
                glyph=self.act(self.std_down[i-1](glyph))
                std_glyphs.append(glyph)
            projected=self._scaled(self.std_proj[i](glyph),self.std_gates,i,self.inference_std_scales)
            enc_writer=writer if self.writer_mode=='film' else (None if self.decouple_writer_condition else writer)
            h=stage(h+projected,conds[i],enc_writer,drop_path_scale,writer_mask);skips.append(h)
            if i<2:h=self.down[i](h)
        h=self.middle2(self.middle_attn(self.middle1(h,conds[2],writer,drop_path_scale,writer_mask,brush)),conds[2],writer,drop_path_scale,writer_mask,brush)
        for i in (2,1,0):
            skip=self._scaled(skips[i],self.skip_gates,i,self.inference_skip_scales)
            h=self.decoder[i](torch.cat([h,skip],dim=1),conds[i],writer,drop_path_scale,writer_mask,brush)
            if getattr(self,'decoder_std_residual',False) and i in (1,0):
                h=h+self.decoder_std_scale*self.dec_std_proj[i](std_glyphs[i])
            if i:h=self.up[i-1](F.interpolate(h,scale_factor=2,mode='nearest'))
        return self.output(h)

    def forward_with_cfg(self,x,t,y_callig,y_char=None,cfg_scale=1.,g=None,y_script=None,brush_attrs=None,brush_valid=None,
                         reference_features=None,reference_mask=None,**unused):
        if self.training:raise RuntimeError('CFG sampling requires eval mode')
        if self.brush_dim and brush_attrs is None:
            ids=torch.full_like(y_script,self.num_writers) if self.writer_mode=='none' else y_callig
            brush_attrs=self.brush_defaults[ids,y_script]
        if self.reference_dim and cfg_scale!=1:raise ValueError('Reference CFG is not trained; use cfg_scale=1')
        conditional=self(x,t,y_callig,y_char,g=g,y_script=y_script,brush_attrs=brush_attrs,brush_valid=brush_valid,
                         reference_features=reference_features,reference_mask=reference_mask)
        if cfg_scale==1 or self.writer_mode=='none':return conditional
        unconditional=self(x,t,torch.full_like(y_callig,self.num_writers),
                           g=g,y_script=y_script,brush_attrs=brush_attrs,brush_valid=brush_valid)
        return unconditional+float(cfg_scale)*(conditional-unconditional)
