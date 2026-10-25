"""Re-render frozen font assignments: bbox-fit256 display + exact BOX64 model cache."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image,ImageDraw,ImageFont,ImageOps

from src.utils.std_soft_glyph import FontFace,render_base_mask,sha256_file


def write_csv(path,rows):
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--out',required=True)
    args=p.parse_args();source=Path(args.source);out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    with (source/'index.csv').open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
    for name in ('display256','condition64'):(out/name).mkdir()
    audit=[];fonts={};preview=[]
    for i,row in enumerate(rows):
        face=FontFace(row['font_path'],int(row['font_face']))
        if face.path not in fonts:fonts[face.path]=sha256_file(face.path)
        old=Image.open(source/row['path']).convert('L')
        full=render_base_mask(row['character'],face,output_size=256)
        small=full.resize((64,64),Image.Resampling.BOX)
        bounds=full.getbbox();box64=small.getbbox()
        if max(bounds[2]-bounds[0],bounds[3]-bounds[1])!=256:raise ValueError('Display bbox not filling longest side')
        if max(box64[2]-box64[0],box64[3]-box64[1])!=64:raise ValueError('Condition bbox not filling longest side')
        for name,image in (('display256',full),('condition64',small)):
            path=out/name/row['path'];path.parent.mkdir(parents=True,exist_ok=True);image.save(path)
        oldbox=old.getbbox();oldspan=max(oldbox[2]-oldbox[0],oldbox[3]-oldbox[1])/old.width
        audit.append(dict(script_id=row['script_id'],character=row['character'],font_file=row['font_file'],
                          old_longest_fraction=oldspan,bbox256=str(bounds),bbox64=str(box64)))
        if row['character'] in '一丶永得尚靑青巫皇入體':preview.append((row,old,full,small))
        if (i+1)%1000==0:print('BUILT',i+1,'/',len(rows),flush=True)
    provenance=dict(source=str(source),source_index_sha256=sha256_file(source/'index.csv'),n=len(rows),
        geometry='nonzero ink bbox, longest side fit256, centered, preserve aspect; no margin',
        polarity='ink255/background0',condition='canonical256 BOX-downsample64 before optional blur',
        font_assignment='exact source index, no font reselection',font_sha256=fonts,
        old_longest_fraction_quantiles=np.quantile([r['old_longest_fraction'] for r in audit],[0,.5,1]).tolist(),
        verified_display_longest=256,verified_condition_longest=64)
    for name in ('display256','condition64'):
        write_csv(out/name/'index.csv',rows)
        (out/name/'provenance.json').write_text(json.dumps(provenance,indent=2,ensure_ascii=False),encoding='utf-8')
    write_csv(out/'glyph_audit.csv',audit)
    (out/'audit.json').write_text(json.dumps(provenance,indent=2,ensure_ascii=False),encoding='utf-8')
    # Representative shape extremes, all three scripts; no inference performed.
    examples=[]
    for sid in ('0','3','4'):
        bychar={r[0]['character']:r for r in preview if r[0]['script_id']==sid}
        for char in ('一','永','得','入'):
            if char in bychar:examples.append(bychar[char])
    cell=256;left=180;header=64
    panel=Image.new('RGB',(left+3*cell,header+cell*len(examples)),'#eef2f6');d=ImageDraw.Draw(panel)
    font=ImageFont.truetype('tools/fonts/simkai.ttf',20)
    for j,title in enumerate(('旧固定em','新bbox / 256','模型64上采样')):d.text((left+j*cell+12,18),title,font=font,fill='#223348')
    for i,(row,*images) in enumerate(examples):
        y=header+i*cell;d.text((12,y+30),row['character']+' / '+row['script_key'],font=font,fill='#223348')
        for j,image in enumerate(images):
            image=image.resize((cell,cell),Image.Resampling.NEAREST)
            panel.paste(ImageOps.invert(image).convert('RGB'),(left+j*cell,y))
    panel.save(out/'preview.png')
    print('BBOX_CACHE_DONE',len(rows),json.dumps(provenance['old_longest_fraction_quantiles']),flush=True)


if __name__=='__main__':main()
