"""CPU-capable architecture/augmentation/backprop/strict-reload check, not a quality experiment."""
import argparse
import json
from pathlib import Path
import sys
import time
import torch
from torch.nn import functional as F
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from src.train.cli import parse_args
from src.model.build import build_dit_from_args
from src.eval.model_io import load_model_from_ckpt
from src.utils.std_soft_glyph import StdSoftGlyphStore
from src.train.std_shape import encoder_health
from tools.analysis.probe_v90_std import layer_probe


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--out',required=True)
    p.add_argument('--device',default='cpu');p.add_argument('--batch',type=int,default=2)
    a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.manual_seed(90)
    args=parse_args(['--config',a.config])
    assert args.model=='DiT-2Cond-S384-20/2' and not args.pretrained and not args.warm_start_from and not args.resume_full
    assert args.std_shape_aug in ('strong','mild')
    model=build_dit_from_args(args,device=a.device)
    data=torch.load(args.std_shape_cache,map_location='cpu',weights_only=True,mmap=True)
    store=StdSoftGlyphStore(args.std_soft_root,eval_sigma=.65,preload=False,strict=True)
    ids=list(range(a.batch))
    clean=torch.stack([store.get(*data['keys'][i]) for i in ids]).to(a.device)
    z=torch.randn(a.batch,4,32,32,device=a.device);saved=z.clone()
    kwargs=dict(y_callig=torch.arange(a.batch,device=a.device)%10,
        y_script=torch.zeros(a.batch,device=a.device,dtype=torch.long),
        y_char=torch.zeros(a.batch,device=a.device,dtype=torch.long))
    opt=torch.optim.AdamW(model.parameters(),lr=1e-4)
    history=[];started=time.monotonic()
    for step in range(3):
        opt.zero_grad(set_to_none=True)
        g=data['variants'][ids,1,step%data['variants'].shape[2]].to(a.device).float()
        eps=torch.randn_like(z);t=torch.full((a.batch,),.5,device=a.device);xt=.5*z+.5*eps
        output=model(xt,t*1000,g=g,return_intermediate_layer=8,**kwargs)
        velocity,features=output
        assert features.shape[:2]==(a.batch,256) and features.shape[-1]==384
        loss=F.mse_loss(velocity[:,:4],eps-z);loss.backward()
        grad=model.glyph_embedder[-1].weight.grad
        history.append({'step':step,'fm':float(loss.detach()),'glyph_output_grad':float(grad.norm()),
                        'augmentation_source':'offline_bank'})
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite preflight FM')
        opt.step()
    assert torch.equal(z,saved) and history[-1]['glyph_output_grad']>0
    model.eval();g=clean
    health=encoder_health(model,g,z)
    layer=layer_probe(model,z,g,dict(kwargs,g=g))
    # Keep the full checkpoint: proves the production eval loader reconstructs20L and bounded gates.
    checkpoint=out/'checkpoint.pt';torch.save({'model':model.state_dict(),'args':args},checkpoint)
    restored,_=load_model_from_ckpt(str(checkpoint),device=a.device,use_ema=False)
    assert len(restored.blocks)==20 and restored.glyph_vec_cond and not restored.std_soft_bias
    assert set(model.state_dict())==set(restored.state_dict())
    report={'status':'passed','device':a.device,'test_batch':a.batch,'configured_training_batch':args.global_batch_size,
        'parameters':sum(p.numel() for p in model.parameters()),'depth':20,'hidden_size':384,'heads':8,
        'strict_reload':True,'history':history,'health':health,'layers':layer,
        'elapsed_seconds':time.monotonic()-started,
        'limitations':'Synthetic FM and real-font STD CPU check only. No real REPA loss, style quality, GPU peak or throughput conclusion.'}
    (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
