"""Auditable real-font augmentation sheet and fallback/mass statistics, CPU only."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
from PIL import Image,ImageDraw
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from src.utils.std_soft_glyph import StdSoftGlyphStore
from src.utils.std_shape_aug import StdShapeAugment


def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',required=True);p.add_argument('--store',required=True)
    p.add_argument('--out',required=True);a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    data=torch.load(a.cache,map_location='cpu',weights_only=True,mmap=True)
    index={tuple(k):i for i,k in enumerate(data['keys'])}
    chosen=[]
    for script in (0,3,4):
        keys=[(script,c) for c in '永家端書風之字水龍器人無' if (script,c) in index][:4]
        chosen.extend(keys)
    store=StdSoftGlyphStore(a.store,eval_sigma=.65,preload=False,strict=True)
    raw=data['masks'][[index[k] for k in chosen]]
    clean=torch.stack([store.get(*k) for k in chosen]);aug=StdShapeAugment()
    cell=88;left=100;header=30
    canvas=Image.new('RGB',(left+cell*4,header+cell*len(chosen)),'white');draw=ImageDraw.Draw(canvas)
    metrics={}
    for tier,name in enumerate(('clean','mild','strong','very_strong')):
        draw.text((left+tier*cell+4,8),name,fill='black')
        result,stats=aug(raw,clean,10000,force_tier=tier,generator=torch.Generator().manual_seed(90+tier))
        metrics[name]={key:[float(v) for v in stats[key]] for key in ('mass_ratio','soft_overlap','clipped_fraction','fallback')}
        for row,key in enumerate(chosen):
            y=header+row*cell
            if tier==0:draw.text((4,y+5),f'script{key[0]} U+{ord(key[1]):04X}',fill='black')
            im=Image.fromarray(((1-result[row,0].numpy())*255).astype(np.uint8))
            canvas.paste(im.resize((cell-4,cell-4),Image.Resampling.NEAREST),(left+tier*cell,y))
            if stats['fallback'][row]:draw.text((left+tier*cell,y+cell-15),'FALLBACK',fill='red')
    canvas.save(out/'preview.png')
    (out/'audit.json').write_text(json.dumps({'rows':chosen,'metrics':metrics},ensure_ascii=False,indent=2),encoding='utf-8')
    print('STD_SHAPE_PREVIEW_READY',out,flush=True)


if __name__=='__main__':main()
