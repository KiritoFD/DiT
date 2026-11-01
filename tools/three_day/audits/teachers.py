"""Strict frozen teacher loading; preserve legacy Pixel D1 implementation."""
import importlib.util
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F


class Teacher(nn.Module):
    def __init__(self,name,root,nslots):
        super().__init__();self.name=name;self.domain='latent' if name in ('D1','D2','D3') else 'pixel'
        if name=='DINO':
            from transformers import Dinov2Config,Dinov2Model
            from safetensors.torch import load_file
            self.model=Dinov2Model(Dinov2Config(hidden_size=384,num_hidden_layers=12,num_attention_heads=6,
                intermediate_size=1536,image_size=518,patch_size=14,layerscale_value=1.,layer_norm_eps=1e-6))
            weights=load_file('pretrained_models/dinov2_vits14_pretrain.safetensors')
            if any('.attention.q_proj.' in k for k in self.model.state_dict()):
                renames={'.attention.attention.query.':'.attention.q_proj.',
                         '.attention.attention.key.':'.attention.k_proj.',
                         '.attention.attention.value.':'.attention.v_proj.',
                         '.attention.output.dense.':'.attention.o_proj.'}
                for old,new in renames.items():weights={k.replace(old,new):v for k,v in weights.items()}
            self.model.load_state_dict(weights,strict=True)
        else:
            ck=torch.load(Path(root)/'teachers'/(('pixel' if name=='pixel_patch' else name)+'.pt'),map_location='cpu',weights_only=False)
            if name in ('pixel','pixel_patch'):
                spec=importlib.util.spec_from_file_location('legacy_pixel',Path(root)/'teachers/legacy_pixel.py')
                module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
                self.model=module.WriterStylePixel(nslots)
            else:
                from src.model.writer_style_latent import build_writer_style_latent
                self.model=build_writer_style_latent(nslots,ck['config'].get('architecture','d1'))
            self.model.load_state_dict(ck['model'],strict=True)
        self.eval().requires_grad_(False)

    def forward(self,x,patch=False):
        if self.domain=='latent':return self.model(x)['embedding']
        if x.shape[1]==1:gray=x;rgb=x.expand(-1,3,-1,-1)
        else:rgb=x;gray=(x*x.new_tensor([.299,.587,.114])[None,:,None,None]).sum(1,keepdim=True)
        if self.name in ('pixel','pixel_patch'):
            gray=F.interpolate(gray,(128,128),mode='bicubic',align_corners=False,antialias=True).clamp(0,1)
            if self.name=='pixel_patch':
                h=self.model.stages[1](self.model.stages[0](self.model.stem(2*gray-1)))
                fg=(F.adaptive_avg_pool2d(1-gray,h.shape[-2:])>.03).float()
                mean=(h.float()*fg).sum((2,3))/fg.sum((2,3)).clamp_min(1)
                var=((h.float()-mean[:,:,None,None]).square()*fg).sum((2,3))/fg.sum((2,3)).clamp_min(1)
                return F.normalize(torch.cat([mean,(var+1e-6).sqrt()],1),dim=-1)
            return self.model(gray)['embedding']
        rgb=F.interpolate(rgb,(224,224),mode='bicubic',align_corners=False,antialias=True).clamp(0,1)
        rgb=(rgb-rgb.new_tensor([.485,.456,.406])[None,:,None,None])/rgb.new_tensor([.229,.224,.225])[None,:,None,None]
        tokens=self.model(pixel_values=rgb).last_hidden_state
        return F.normalize(tokens[:,1:] if patch else tokens[:,0],dim=-1)


def decode(vae,z,scale=.18215,shift=0.):
    return (vae.decode(z/scale+shift).sample.float().clamp(-1,1)+1)/2


def payloads():
    pixel=torch.load('scratch/writer_pixel_d0d1_20261016_01/assets/data/data.pt',map_location='cpu',weights_only=False,mmap=True)
    latent=torch.load('assets/writer_latent_data_20261016/data.pt',map_location='cpu',weights_only=False,mmap=True)
    for k in ('writer','script','char','labels','source'):
        import numpy as np
        if not np.array_equal(pixel[k],latent[k]):raise ValueError('Teacher pixel/latent row alignment mismatch: '+k)
    return pixel,latent
