"""Reuse saved native-evaluation PNGs; no resampling or evaluation classifier training."""
import json
from pathlib import Path
import subprocess
import sys
from PIL import Image,ImageDraw
import numpy as np
from tools.p2_audit.common import read,save,paired_ci


def evaluate(root,configs):
    out=root/'eval';out.mkdir()
    sources={'P2':Path('assets/p2_audit_20261025/baseline/val100/eval_samples_ctrl/step0020000/eval200fix')}
    steps=configs['A_strong']['max_steps'];csv=configs['A_strong']['eval_csv']
    for arm in configs:
        run=next((root/arm/'runs').glob('*-Unet-flow'))
        for step in (steps//2,steps):
            sources[f'{arm}_{step}']=run/'eval_samples_ctrl'/f'step{step:07d}'/'eval200fix'
    for name,path in sources.items():
        dest=out/name;dest.mkdir();(dest/'samples').symlink_to(path.resolve(),target_is_directory=True)
        save(dest/'samples.json',read(csv));save(dest/'generation.json',dict(swap=False,source=str(path)))
        cmd=[sys.executable,'-u','-m','tools.p2_audit.score','--run',str(dest)]
        if name!='P2':cmd+=['--baseline',str(out/'P2')]
        subprocess.run(cmd,check=True)
    data={name:json.loads((out/name/'scores.json').read_text()) for name in sources}
    # Define proxy groups from the frozen P2 alone, never from the winning arm.
    base=data['P2']['per_image'];problem=[i for i,r in enumerate(base) if r['frag']>=3 or r['soft_ink']<.6 or r['soft_ink']>1.5 or r['ink_iou']<.15]
    groups={'all':list(range(len(base))),'problem_proxy':problem,'other':sorted(set(range(len(base)))-set(problem))}
    # Locked from the P2 poster before either trained arm has been evaluated.
    # Deliberately small visual case lists, not another statistical test set.
    groups['issue_visual']=[4,15,52,56,82]
    groups['normal_visual']=[23,45,87,92]
    if len(base)!=100:raise ValueError('Visual cases belong to the pinned val100 protocol')
    result={}
    for name,d in data.items():
        result[name]={}
        for group,ids in groups.items():
            if not ids:continue
            rr=[d['per_image'][i] for i in ids]
            result[name][group]={key:float(np.mean([r[key] for r in rr])) for key in ('ssim','lpips','ink_iou','frag','width_w1','soft_ink')}
            result[name][group]['n']=len(ids)
        if name!='P2':
            result[name]['paired_vs_P2']={key:paired_ci([r[key]-b[key] for r,b in zip(d['per_image'],base)],[r['char'] for r in base]) for key in ('lpips','ink_iou','frag','width_w1')}
    save(out/'comparison.json',dict(results=result,groups=groups,
        visual_cases={g:[dict(idx=i,char=base[i]['char'],writer=base[i]['writer']) for i in ids] for g,ids in groups.items() if g.endswith('_visual')},
        caveat='Proxy issue bins are not OCR or human content correctness; fixed visual lists are qualitative and tiny; Writer probe is auxiliary only'))
    names=['P2',f'A_strong_{steps}',f'B_mild_{steps}']
    for page,start in enumerate(range(0,len(base),20)):
        grid=Image.new('RGB',(800,32+180*min(20,len(base)-start)),'#eeeeee');draw=ImageDraw.Draw(grid)
        for col,name in enumerate(['GT']+names):draw.text((col*200+5,8),name,fill='black')
        for row,i in enumerate(range(start,min(start+20,len(base)))):
            files=[out/'P2'/'samples'/f'gt{i}.png']+[out/name/'samples'/f'g{i}.png' for name in names]
            for col,file in enumerate(files):
                grid.paste(Image.open(file).convert('RGB').resize((174,174)),(200*col,32+row*180))
        grid.save(out/f'comparison_page{page}.png')
