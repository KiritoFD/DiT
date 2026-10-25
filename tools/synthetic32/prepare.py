"""Prepare training-only cross-character references and a frozen 32-image pilot."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import tarfile

import numpy as np
from scipy import ndimage
from PIL import Image,ImageDraw,ImageFont,ImageOps


WRITERS=('颜真卿','智永','赵佶')
TARGETS='泉梅岳霞'
SEEDS=(20261022,20261023)
DATA=Path('assets/data_cohorts/top20_20261021_v1')


def read(p):
    with Path(p).open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',default='assets/synthetic/qwen32_20261022')
    p.add_argument('--refs',type=int,default=3);p.add_argument('--style-only',action='store_true');args=p.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False);(out/'inputs').mkdir()
    base=DATA/'datasets_final/top20';train=read(base/'train.csv')
    heldout=read(base/'validation.csv')+read(base/'test.csv')
    legacy=Path('exp-std/csv/eval200_fixed.csv')
    if legacy.exists():heldout+=read(legacy)
    heldout_chars={r['character'] for r in heldout}
    assert not set(TARGETS)&heldout_chars,'Target character occurs in a frozen holdout'
    for writer in WRITERS:
        assert not set(TARGETS)&{r['character'] for r in train if r['calligrapher']==writer}
    refs={};reference_records={}
    font=ImageFont.truetype('tools/fonts/simkai.ttf',22)
    sheet=Image.new('RGB',((args.refs+1)*256,3*290),'#eef2f6');draw=ImageDraw.Draw(sheet)
    for wi,writer in enumerate(WRITERS):
        pool=[r for r in train if r['calligrapher']==writer and r['script']=='楷' and r['character'] not in TARGETS]
        preferred='永之書道家心國龍德天人山水中大上長風'
        ranks={ch:i for i,ch in enumerate(preferred)}
        # Fixed source category is not claimed to be a verified artwork ID.
        source_counts=Counter(r.get('source','') for r in pool)
        pool.sort(key=lambda r:(-source_counts[r.get('source','')],ranks.get(r['character'],999),sha_text(r['image_path'])))
        chosen=[];seen=set()
        for row in pool:
            if row['character'] in seen:continue
            image=Image.open(row['image_path']).convert('RGB').resize((256,256),Image.Resampling.BICUBIC)
            gray=np.asarray(image.convert('L'));ink=gray<128
            edge=float(np.concatenate((ink[:2].ravel(),ink[-2:].ravel(),ink[:,:2].ravel(),ink[:,-2:].ravel())).mean())
            frame=max(float(ink[:48].mean(1).max()),float(ink[-48:].mean(1).max()),
                      float(ink[:,:48].mean(0).max()),float(ink[:,-48:].mean(0).max()))
            labels,_=ndimage.label(ink)
            isolated_frame=False
            for sy,sx in ndimage.find_objects(labels):
                h=sy.stop-sy.start;w=sx.stop-sx.start
                isolated_frame |= (w>40 and h<=3 and (sy.start<48 or sy.stop>208))
                isolated_frame |= (h>102 and w<=3 and (sx.start<48 or sx.stop>208))
            if not .025<float(ink.mean())<.55 or edge>.05 or frame>.65 or isolated_frame:continue
            rel=f'inputs/w{wi}_r{len(chosen)}.png';image.save(out/rel)
            record=dict(row,file=rel,source_image_sha256=sha(row['image_path']),input_sha256=sha(out/rel),ink_fraction=float(ink.mean()))
            chosen.append(record);seen.add(row['character'])
            if len(chosen)==args.refs:break
        if len(chosen)!=args.refs:raise ValueError('Insufficient clean distinct references for '+writer)
        refs[writer]=[r['file'] for r in chosen];reference_records[writer]=chosen
        draw.text((10,wi*290+5),writer+' · 楷',font=font,fill='#203248')
        for j,r in enumerate(chosen):
            sheet.paste(Image.open(out/r['file']),((j+1)*256,wi*290+32))
            draw.text(((j+1)*256+4,wi*290+5),r['character'],font=font,fill='#203248')
    std={}
    for char in TARGETS:
        source=DATA/'std_bbox_v1/display256/kai'/f'U+{ord(char):05X}.png'
        image=ImageOps.invert(Image.open(source).convert('L')).convert('RGB')
        rel=f'inputs/std_U{ord(char):04X}.png';image.save(out/rel);std[char]=rel
    jobs=[]
    targets=TARGETS[:2] if args.style_only else TARGETS
    seeds=SEEDS[:1] if args.style_only else SEEDS
    for char in targets:
        for seed in seeds:
            for wi,writer in enumerate(WRITERS):
                jobs.append(dict(id=f'U{ord(char):04X}_w{wi}_s{seed}',character=char,writer=writer,
                    seed=seed,images=refs[writer] if args.style_only else [std[char],*refs[writer]],
                    mode='style_only' if args.style_only else 'cross_character_references'))
            if not args.style_only:
                jobs.append(dict(id=f'U{ord(char):04X}_stdonly_s{seed}',character=char,writer=None,
                    seed=seed,images=[std[char]],mode='std_only'))
    manifest=dict(model='Qwen/Qwen-Image-2.1-Turbo',n=len(jobs),writers=WRITERS,script='楷',targets=list(targets),
        seeds=seeds,reference_records=reference_records,jobs=jobs,
        source_csv_sha256={str(base/f'{s}.csv'):sha(base/f'{s}.csv') for s in ('train','validation','test')},
        input_sha256={str(p.relative_to(out)):sha(p) for p in (out/'inputs').glob('*.png')},
        exclusions='Targets absent from all scripts of the three writers in frozen train and all frozen holdout characters; refs training only and other characters',
        caveat='Source category is not an artwork ID; teacher pretraining contamination cannot be excluded; outputs are unverified synthetic candidates, not GT',
        license='Research/evaluation pilot; official Qwen Research License retained; no commercial approval implied')
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    sheet.save(out/'references.png')
    with tarfile.open(out/'inputs.tar.gz','w:gz') as tar:
        tar.add(out/'manifest.json',arcname='manifest.json');tar.add(out/'inputs',arcname='inputs');tar.add(out/'references.png',arcname='references.png')
    print('PREPARED',len(jobs),json.dumps({w:[r['character'] for r in rr] for w,rr in reference_records.items()},ensure_ascii=True),flush=True)


def sha_text(value):return hashlib.sha256(value.encode()).hexdigest()


if __name__=='__main__':main()
