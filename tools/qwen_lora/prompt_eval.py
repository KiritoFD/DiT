"""Explicit writer/script prompts: pure non-Turbo base and trained Turbo adapter."""
import argparse
import json
from pathlib import Path
import sys
import shutil

from cache import setup,text_prompt


def main(source,out):
    setup(source)
    import numpy as np
    import torch
    from PIL import Image,ImageDraw,ImageFont
    from diffusers import QwenImage21Pipeline
    from condition import Conditions
    from src.eval.metrics import ssim,get_lpips
    from src.eval.metrics_ink import ink_iou
    torch.set_num_threads(8)
    out.mkdir(parents=True,exist_ok=False)
    data=json.loads((source/'data.json').read_text(encoding='utf-8'))
    rows=[]
    for wi in range(3):rows.extend([r for r in data['datasets']['validation'] if r['writer_index']==wi][:2])
    state=dict(phase='encoding',completed=0,total=18)
    def save(): (out/'status.json').write_text(json.dumps(state,indent=2))
    save()
    groups={}
    for label,model,steps in [('base',Path('/home/ds/Workspace/xy/qwen-lora-20261023/base'),40),
                              ('turbo6000',source/'base',8)]:
        encoder=QwenImage21Pipeline.from_pretrained(model,transformer=None,vae=None,scheduler=None,
                                                   dtype=torch.bfloat16,local_files_only=True).to('cuda')
        encoder.text_encoder.eval().requires_grad_(False)
        prompts={variant:[(f'书法家：{r["calligrapher"]}。书体：{r["script"]}。请用该书法家的{r["script"]}风格书写。'
                           if variant=='explicit' else '')+text_prompt(r['character']) for r in rows]
                 for variant in ('original','explicit')}
        features={}
        with torch.inference_mode():
            for variant in prompts:
                emb,mask,pad=encoder.encode_prompt(prompts[variant],device='cuda')
                if pad.any():raise ValueError('Unexpected image tokens')
                features[variant]=[(emb[i,mask[i].bool()] if mask is not None else emb[i]).cpu() for i in range(len(rows))]
        encoder.text_encoder.to('cpu');del encoder
        pipe=QwenImage21Pipeline.from_pretrained(model,text_encoder=None,dtype=torch.bfloat16,local_files_only=True).to('cuda')
        pipe.set_progress_bar_config(disable=True)
        cond=None
        if label=='turbo6000':
            checkpoint=source/'checkpoints/0006000'
            pipe.load_lora_weights(checkpoint)
            saved=torch.load(checkpoint/'state.pt',map_location='cpu',weights_only=False)
            cond=Conditions(source/'writer_table.npy').cuda().eval()
            cond.load_state_dict(saved['condition'],strict=True)
        for variant in ('original','explicit'):
            name=f'{label}_{variant}';dest=out/name;dest.mkdir()
            if label=='turbo6000' and variant=='original':
                previous=source/'evaluations/0006000'
                metrics=json.loads((previous/'metrics.json').read_text())
                for p in previous.glob('*.png'):shutil.copy2(p,dest/p.name)
                groups[name]=metrics
                continue
            state.update(phase='generating',group=name);save()
            metrics=[]
            for i,row in enumerate(rows):
                emb=features[variant][i][None].cuda();mask=torch.ones(emb.shape[:2],device='cuda',dtype=torch.bool)
                with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                    if cond is not None:
                        emb,mask=cond(emb,mask,torch.tensor([row['writer_index']],device='cuda'),torch.tensor([0],device='cuda'))
                    image=pipe(prompt_embeds=emb,prompt_embeds_mask=mask,height=512,width=512,output_resolution=512,
                               num_inference_steps=steps,true_cfg_scale=1.,use_kv_cache=True,
                               generator=torch.Generator('cpu').manual_seed(90)).images[0]
                actual=len(pipe.scheduler.timesteps)
                if actual!=steps:raise ValueError(f'Steps mismatch: {actual}')
                white=Image.new('RGBA',image.size,'white');white.alpha_composite(image.convert('RGBA'))
                image=white.convert('RGB').resize((256,256),Image.Resampling.LANCZOS)
                file=f'U{ord(row["character"]):04X}_w{row["writer_index"]}.png';image.save(dest/file)
                pred=np.asarray(image,dtype=np.float32)/255
                gt=np.asarray(Image.open(source/row['file']).convert('RGB'),dtype=np.float32)/255
                lp,error=get_lpips('cuda')
                if lp is None:raise RuntimeError(error)
                ten=lambda a:torch.from_numpy(a.copy()).permute(2,0,1)[None].cuda()*2-1
                with torch.inference_mode():distance=float(lp(ten(pred),ten(gt)).item())
                metrics.append(dict(character=row['character'],writer=row['calligrapher'],prompt=prompts[variant][i],
                                    gt=row['file'],steps=steps,ssim=ssim(pred,gt),lpips=distance,ink_iou=ink_iou(pred,gt)))
                state['completed']+=1;save()
            groups[name]=metrics
            (dest/'metrics.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2),encoding='utf-8')
        pipe.to('cpu');del pipe,cond
    summary={g:{k:float(np.mean([r[k] for r in values])) for k in ('ssim','lpips','ink_iou')} for g,values in groups.items()}
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    canvas=Image.new('RGB',(1300,1790),'#eef2f6');draw=ImageDraw.Draw(canvas)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
    names=list(groups)
    for j,header in enumerate(['GT']+names):draw.text((j*260+5,10),header,font=font,fill='black')
    for i,row in enumerate(rows):
        name=f'U{ord(row["character"]):04X}_w{row["writer_index"]}.png';y=44+i*288
        paths=[source/row['file']]+[out/g/name for g in names]
        for j,path in enumerate(paths):canvas.paste(Image.open(path).convert('RGB'),(j*260,y))
        for j,g in enumerate(names,1):
            m=groups[g][i];draw.text((j*260+3,y+258),f'S {m["ssim"]:.3f} L {m["lpips"]:.3f} I {m["ink_iou"]:.3f}',font=font,fill='black')
    canvas.save(out/'poster.jpg',quality=65,optimize=True)
    state.update(phase='completed',summary=summary);save()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();main(a.source.resolve(),a.out.resolve())
