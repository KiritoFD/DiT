"""Writer-isolated multi-scale velocity residual (S9/S10/S11)."""
import torch
from torch import nn
from torch.nn import functional as F
from .blocks import ResBlock, timestep_embedding, norm


class WriterVelocityUNet(nn.Module):
    std_soft_cond=True;use_glyph_cond=True;in_channels=4;out_channels=4

    def __init__(self,backbone,mode='cnn',width=192):
        super().__init__()
        if mode not in ('cnn','geometry','attention'):raise ValueError(mode)
        self.backbone=backbone;self.mode=mode
        self.num_writers=backbone.num_writers;self.num_scripts=backbone.num_scripts
        self.drop_writer=backbone.drop_writer
        backbone.writer_mode='none'
        backbone.writer.requires_grad_(False);backbone.writer_proj.requires_grad_(False)
        self.writer=nn.Embedding(self.num_writers+1,128)
        self.condition=nn.Sequential(nn.Linear(128+128+16,512),nn.SiLU(),nn.Linear(512,512))
        channels=[s.blocks[-1].conv2.out_channels for s in backbone.decoder]
        self.inputs=nn.ModuleList([nn.Conv2d(c+(6 if mode=='geometry' else 1),width,1) for c in channels])
        self.blocks=nn.ModuleList([nn.ModuleList([ResBlock(width,width) for _ in range(5)]) for _ in channels])
        self.attention=nn.MultiheadAttention(width,4,batch_first=True) if mode=='attention' else None
        self.query=nn.Linear(128,width) if mode=='attention' else None
        self.output=nn.Sequential(norm(width),nn.SiLU(),nn.Conv2d(width,4,3,padding=1))
        nn.init.zeros_(self.output[-1].weight);nn.init.zeros_(self.output[-1].bias)

    @staticmethod
    def geometry(g):
        # Exact separable Euclidean distance on the 32-square condition grid.
        # Binarization only defines ink support; all output fields are continuous.
        g=F.interpolate(g,size=(32,32),mode='bilinear',align_corners=False).float()
        def distance(mask):
            n=g.shape[-1];axis=torch.arange(n,device=g.device,dtype=g.dtype)
            cost=(axis[:,None]-axis[None,:]).square()
            f=torch.where(mask,0.,float(2*n*n))
            f=(f.unsqueeze(-2)+cost).amin(-1)
            f=f.transpose(-1,-2)
            f=(f.unsqueeze(-2)+cost).amin(-1).transpose(-1,-2)
            return f.sqrt().clamp_max(16.)
        ink=g<.5;inside=distance(~ink);outside=distance(ink)
        dx=F.pad(g[:,:,:,2:]-g[:,:,:,:-2],(1,1,0,0))*.5
        dy=F.pad(g[:,:,2:,:]-g[:,:,:-2,:],(0,0,1,1))*.5
        mag=(dx.square()+dy.square()+1e-6).sqrt()
        nx=dx/mag;ny=dy/mag
        curvature=(F.pad(nx[:,:,:,2:]-nx[:,:,:,:-2],(1,1,0,0))+
                   F.pad(ny[:,:,2:,:]-ny[:,:,:-2,:],(0,0,1,1)))*.5
        return torch.cat((g,nx,ny,inside/8.,(inside-outside)/16.,curvature),1)

    def components(self,x,t,y_callig,g,y_script,condition_dropped=False):
        base=self.backbone;w=y_callig;s=y_script
        if self.training and not condition_dropped:
            all_drop=torch.rand(len(x),device=x.device)<base.drop_all
            w=torch.where(all_drop|(torch.rand(len(x),device=x.device)<self.drop_writer),self.num_writers,w)
            s=torch.where(all_drop|(torch.rand(len(x),device=x.device)<base.drop_script),self.num_scripts,s)
        zero=x.new_zeros((len(x),32));te=timestep_embedding(t)
        conds=base._compute_stage_conditions(te,zero,base.script(s))
        glyph=base.std_stem(g);h=base.input(x);skips=[]
        for i,stage in enumerate(base.encoder):
            if i:glyph=base.act(base.std_down[i-1](glyph))
            h=stage(h+base.std_proj[i](glyph),conds[i],zero);skips.append(h)
            if i<2:h=base.down[i](h)
        h=base.middle2(base.middle_attn(base.middle1(h,conds[2],zero)),conds[2],zero)
        features={}
        for i in (2,1,0):
            h=base.decoder[i](torch.cat((h,skips[i]),1),conds[i],zero);features[i]=h
            if i:h=base.up[i-1](F.interpolate(h,scale_factor=2,mode='nearest'))
        content=base.output(h)
        e=self.writer(w);c=self.condition(torch.cat((te,e,base.script(s)),1))
        spatial=self.geometry(g) if self.mode=='geometry' else g
        residual=None
        for i in (2,1,0):
            f=features[i];local=F.interpolate(spatial,size=f.shape[-2:],mode='bilinear',align_corners=False)
            h=self.inputs[i](torch.cat((f,local),1))
            if residual is not None:h=h+F.interpolate(residual,size=h.shape[-2:],mode='nearest')
            for block in self.blocks[i]:h=block(h,c)
            if self.attention is not None:
                tokens=h.flatten(2).transpose(1,2);q=self.query(e)[:,None]
                context=self.attention(q,tokens,tokens,need_weights=False)[0]
                h=h+context.transpose(1,2)[:,:,:,None]
            residual=h
        style=self.output(residual)*(w!=self.num_writers)[:,None,None,None]
        return content,style

    def forward(self,x,t,y_callig,y_char=None,g=None,y_script=None,condition_dropped=False,**unused):
        content,style=self.components(x,t,y_callig,g,y_script,condition_dropped)
        return content+style

    def forward_with_cfg(self,x,t,y_callig,y_char=None,cfg_scale=1.,g=None,y_script=None,**unused):
        if self.training:raise RuntimeError('Sampling requires eval')
        content,style=self.components(x,t,y_callig,g,y_script,True)
        return content+float(cfg_scale)*style
