"""Verify conditional decoder equivalence and image-gradient routing."""
import argparse
import json
import torch
from src.unet.io import Config,build
from src.unet.joint.objective import JointDecoderObjective


def main():
    p=argparse.ArgumentParser();p.add_argument('--spec',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    torch.set_num_threads(8);spec=json.load(open(a.spec));cfg=Config(**spec['config'])
    model=build(cfg).cuda();result={}
    z=torch.randn(2,4,32,32,device='cuda')*.15
    g=torch.rand(2,1,64,64,device='cuda');w=torch.tensor([0,1],device='cuda');s=torch.tensor([0,1],device='cuda')
    for arm in ('G2J4','G2J5','G2J6','G2J7'):
        model.zero_grad(set_to_none=True)
        obj=JointDecoderObjective(arm,cfg,image_micro=1,image_batch=2,img_weight=.02)
        with torch.no_grad():
            base=obj.vae.decoder.base if hasattr(obj.vae.decoder,'base') else obj.vae.decoder
            original=base(obj.vae.post_quant_conv(z/cfg.vae_scaling_factor+cfg.vae_shift_factor))
            same=obj.decode(z,w,s,g);error=float((original-same).abs().max())
            assert error<1e-6,error
        stats=obj.backward(model,z,g,(w,s),0,targets=original.detach())
        unet=sum(float(p.grad.float().square().sum()) for p in model.parameters() if p.grad is not None)**.5
        decoder=sum(float(p.grad.float().square().sum()) for p in obj.vae.decoder.parameters() if p.grad is not None)**.5
        assert decoder>0 and torch.isfinite(torch.tensor(decoder))
        assert (unet>0)==(arm=='G2J7'),(arm,unet)
        result[arm]=dict(initial_max_error=error,unet_image_grad=unet,decoder_grad=decoder,stats=stats)
        del obj;torch.cuda.empty_cache()
    from pathlib import Path
    Path(a.out).parent.mkdir(parents=True,exist_ok=True);Path(a.out).write_text(json.dumps(result,indent=2));print(json.dumps(result))


if __name__=='__main__':main()
