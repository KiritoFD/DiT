"""Fixed-noise independent attribute interventions; never reads evaluation GT attributes."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from diffusers import AutoencoderKL
from src.unet.io import load
from src.unet.brush.attributes import measure,NAMES
from src.utils.std_soft_glyph import StdSoftGlyphStore
from src.loss.flow_matching import FlowMatching
from tools.generic_pretrain.contracts import read_rows,write_rows


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    model,c,_=load(a.checkpoint,'cuda');model.eval()
    cache=np.load(c.brush_cache);center=cache['center'];scale=cache['scale']
    vae=AutoencoderKL.from_pretrained(c.vae_path).cuda().eval()
    store=StdSoftGlyphStore(c.std_soft_root,is_train=False,eval_sigma=0)
    root=Path(c.brush_cache).parent
    rows=read_rows(root/'control.csv')[:16]
    if len(rows)<8:raise ValueError('Insufficient strict character-heldout control glyphs')
    mapping=json.loads(Path(c.font_remap_json).read_text());records=[];panels=[]
    flow=FlowMatching(num_steps=50,sampler='heun')
    for i,row in enumerate(rows):
        rawsid=int(row['script_id']);sid=mapping[str(rawsid)]
        g=store.get(rawsid,row['character']).unsqueeze(0).repeat(3,1,1,1).cuda()
        for seed in range(3):
            noise=torch.randn(1,4,32,32,generator=torch.Generator().manual_seed(9000+seed)).repeat(3,1,1,1).cuda()
            for feature in (1,3,4):
                attrs=np.zeros((3,5),np.float32);attrs[:,feature]=[-1,0,1]
                physical=center[sid]+attrs*scale[sid]
                if feature==1:physical[:,1]=np.clip(physical[:,1],physical[:,0],physical[:,2])
                if feature==3:physical[:,3]=np.clip(physical[:,3],.5,1)
                if feature==4:physical[:,4]=np.maximum(physical[:,4],0)
                attrs=(physical-center[sid])/scale[sid]
                kw=dict(g=g,y_callig=torch.zeros(3,dtype=torch.long,device='cuda'),
                    y_script=torch.full((3,),sid,dtype=torch.long,device='cuda'),
                    brush_attrs=torch.from_numpy(attrs).float().cuda())
                with torch.autocast('cuda',dtype=torch.bfloat16):z=flow.ddim_sample_loop(model,noise.shape,x_T=noise,model_kwargs=kw,device='cuda')
                with torch.autocast('cuda',dtype=torch.float16):images=vae.decode(z.float()/c.vae_scaling_factor).sample
                images=((images.float().clamp(-1,1)+1)/2).mean(1).cpu().numpy()
                measured=[];validity=[]
                for j,image in enumerate(images):
                    value,valid=measure(1-image);measured.append(float(value[feature]));validity.append(valid)
                    name=f'{i:02d}_{seed}_{feature}_{j}.png';Image.fromarray((image*255).astype(np.uint8)).save(out/name)
                records.append(dict(character=row['character'],script=sid,seed=seed,feature=NAMES[feature],
                    target_low=float(physical[0,feature]),target_mid=float(physical[1,feature]),target_high=float(physical[2,feature]),
                    low=measured[0],mid=measured[1],high=measured[2],valid=all(validity),
                    monotonic=bool(all(validity) and measured[0]<=measured[1]<=measured[2]),
                    positive=bool(all(validity) and measured[2]>measured[0])))
                if seed==0:
                    panel=Image.new('L',(768,256),255)
                    for j,image in enumerate(images):panel.paste(Image.fromarray((image*255).astype(np.uint8)),(j*256,0))
                    panels.append(panel)
        print('CONTROL',i+1,'/',len(rows),flush=True)
    write_rows(out/'results.csv',records)
    summary={name:dict(monotonic=float(np.mean([r['monotonic'] for r in records if r['feature']==name])),
        positive=float(np.mean([r['positive'] for r in records if r['feature']==name]))) for name in (NAMES[1],NAMES[3],NAMES[4])}
    passed=all(v['monotonic']>=.6 and v['positive']>=.75 for v in summary.values())
    (out/'summary.json').write_text(json.dumps(dict(features=summary,p1_pass=passed,
        limitations='engineering gate, correlated seeds; manual content/topology review still required; no GT attribute used'),indent=2))
    poster=Image.new('L',(384,len(panels)*128),255)
    for i,panel in enumerate(panels):poster.paste(panel.resize((384,128)),(0,i*128))
    poster.save(out/'control_poster.png')


if __name__=='__main__':main()
