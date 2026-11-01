"""D1 data: render TTF glyphs for the OCR char vocabulary.

Renders every (char,font) pair supported by the installed fonts at 192px,
using fontTools cmaps for coverage. Output: assets/p125_20261030/ocr/fonts/
  {font_tag}/{char_codepoint}.png  + render_manifest.json
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont

FONTS=[("kai","tools/fonts/simkai.ttf"),("song","tools/fonts/simsunb.ttf"),
       ("hei","tools/fonts/SimHei.ttf"),("li","tools/fonts/SIMLI.TTF"),
       ("xing","tools/fonts/STXINGKA.TTF"),("noto","tools/fonts/NotoSerifSC-VF.ttf"),
       ("fz","tools/fonts/candidate_fonts/FZSTK.TTF"),
       ("zhi","tools/fonts/candidate_fonts/ZhiMangXing-Regular.ttf")]
SIZE=192


def render_char(font,char,size=SIZE):
    img=Image.new('L',(size,size),255)
    d=ImageDraw.Draw(img)
    bbox=d.textbbox((0,0),char,font=font)
    w=bbox[2]-bbox[0];h=bbox[3]-bbox[1]
    if w<=0 or h<=0:return None
    scale=min((size*0.86)/w,(size*0.86)/h)
    f2=ImageFont.truetype(font.path,int(max(10,font.size*scale)))
    bbox=d.textbbox((0,0),char,font=f2)
    d.text(((size-(bbox[2]-bbox[0]))/2-bbox[0],(size-(bbox[3]-bbox[1]))/2-bbox[1]),char,font=f2,fill=0)
    return img


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--csv',default='assets/p125_20261030/ocr/ocr_train.csv')
    p.add_argument('--out',default='assets/p125_20261030/ocr/fonts')
    p.add_argument('--max-per-font',type=int,default=0,help='0=all')
    a=p.parse_args()
    from fontTools.ttLib import TTFont as FTFont
    rows=list(csv.DictReader(open(a.csv,encoding='utf-8-sig')))
    chars=sorted({r['character'] for r in rows})
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    manifest=[]
    for tag,path in FONTS:
        if not Path(path).is_file():print('skip missing font',path);continue
        ft=FTFont(path,fontNumber=0)
        cmap=ft.getBestCmap()
        fdir=out/tag;fdir.mkdir(parents=True,exist_ok=True)
        fsize=max(20,int(SIZE*0.7))
        font=ImageFont.truetype(path,fsize)
        n=0
        for ch in chars:
            if ord(ch) not in cmap:continue
            dst=fdir/f'{ord(ch)}.png'
            img=render_char(font,ch)
            if img is not None:img.save(dst);n+=1
            if a.max_per_font and n>=a.max_per_font:break
        manifest.append(dict(font=tag,path=path,rendered=n))
        print(tag,n,flush=True)
    json.dump(manifest,open(out/'render_manifest.json','w'),ensure_ascii=False,indent=1)
    print('total renders:',sum(m['rendered'] for m in manifest))


if __name__=='__main__':main()
