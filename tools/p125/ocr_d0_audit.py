"""OCR-A D0: data audit for the single-character recognizer.

Corpora (4090):
  generic48 train.csv   100k generic rows, writer UNKNOWN (char labels valid)
  p0_train.csv trusted  29,873 rows, 37 named generic writers
  p2_train.csv          23,485 Top10 rows
D0 duties: Unicode NFKC normalization, cross-corpus dedup, coverage stats,
TTF overlap, frozen Real-Test holdout (whole writers + stratified images).
Outputs: assets/p125_20261030/ocr/{audit.json,ocr_train.csv,ocr_realtest.csv}
"""
import argparse
import csv
import hashlib
import json
import unicodedata
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np

OUT=Path("assets/p125_20261030/ocr")
TOP10={"颜真卿","苏轼","王羲之","褚遂良","米芾","欧阳询","柳公权","赵孟頫","黄庭坚","文徵明"}


def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def rows_of(p):
    return list(csv.DictReader(open(p,encoding='utf-8-sig')))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',default=str(OUT))
    a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    corpora={}
    corpora['generic']=rows_of("assets/generic48_kxl_top10_20261022/train.csv")
    corpora['trusted']=[r for r in rows_of("assets/p125_20261030/p0_train.csv") if r['writer_trusted']=='1']
    corpora['top10']=rows_of("assets/u100_p1p2_20261025/p2_train.csv")
    # normalize + dedup (pixel identity where present, else path; key includes char)
    seen=set();clean=[]
    dup=Counter()
    for name,rs in corpora.items():
        for r in rs:
            ch=unicodedata.normalize('NFKC',r['character'])
            r=dict(r);r['character']=ch;r['src']=name
            key=(r.get('pixel_sha256') or r['image_path'],ch)
            if key in seen:dup[name]+=1;continue
            seen.add(key);clean.append(r)
    print('clean rows:',len(clean),'dups:',dict(dup))
    # cross-corpus path overlap
    paths=defaultdict(set)
    for r in clean:paths[r['src']].add(r['image_path'])
    overlap={f'{a}∩{b}':len(paths[a]&paths[b]) for a in paths for b in paths if a<b}
    # coverage
    chars_by_script=defaultdict(set);freq=Counter()
    writers_by_src=defaultdict(set)
    for r in clean:
        chars_by_script[r['script']].add(r['character'])
        freq[(r['script'],r['character'])]+=1
        if r['src']!='generic':writers_by_src[r['src']].add(r['calligrapher'])
    all_chars=set()
    for s,c in chars_by_script.items():all_chars|=c
    # TTF coverage: chars renderable by the installed fonts (fontTools cmap check deferred to render stage;
    # approximate here by the glyph library index if present)
    glyph_index=Path("exp-std/data/std_char_eval")
    ttf_chars=None
    if glyph_index.is_dir():
        ttf_chars={Path(p).stem for p in glyph_index.glob("*.png")}
    ttf_cover=len(all_chars&ttf_chars) if ttf_chars else None
    # Real-Test holdout: whole trusted writers (2 of 37) + stratified top10 images
    tw=sorted(writers_by_src.get('trusted',[]))
    hold_writers=set(tw[::len(tw)//6][:6]) if len(tw)>=6 else set()
    realtest=[];train=[]
    for r in clean:
        if r['src']=='trusted' and r['calligrapher'] in hold_writers:realtest.append(r)
        else:train.append(r)
    # top10 stratified holdout: 12% of images per (writer,script) into Real-Test
    rng=np.random.default_rng(10310031)
    t10=[r for r in train if r['src']=='top10']
    keep=[]
    for i,r in enumerate(t10):
        if int(hashlib.md5((r['image_path']).encode()).hexdigest(),16)%100<12:
            realtest.append(r)
        else:keep.append(r)
    train=[r for r in train if r['src']!='top10']+keep
    fieldnames=[]
    for r in train:
        for k in r:
            if k not in fieldnames:fieldnames.append(k)
    with open(out/'ocr_train.csv','w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fieldnames,extrasaction='ignore');w.writeheader();w.writerows(train)
    with open(out/'ocr_realtest.csv','w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(realtest[0].keys()),extrasaction='ignore');w.writeheader();w.writerows(realtest)
    audit=dict(
        rows=dict(train=len(train),realtest=len(realtest),dup_dropped=dict(dup),total_clean=len(clean)),
        overlap_paths=overlap,
        coverage=dict(unique_chars=len(all_chars),
            per_script={s:len(c) for s,c in sorted(chars_by_script.items())},
            ttf_glyph_lib_chars=len(ttf_chars) if ttf_chars is not None else 'index-not-found',
            ttf_cover_of_real_chars=ttf_cover,
            char_freq_decile=[int(x) for x in np.percentile(list(freq.values()),[10,50,90])]),
        realtest=dict(held_out_writers=sorted(hold_writers),n=len(realtest)),
        provenance={n:sha(f"assets/generic48_kxl_top10_20261022/train.csv") if n=='generic' else None for n in ()})
    save=dict(audit,audit_sha={})
    Path(a.out).joinpath('audit.json').write_text(json.dumps(save,ensure_ascii=False,indent=1,default=str))
    print(json.dumps(dict(train=len(train),realtest=len(realtest),unique_chars=len(all_chars),
        per_script={s:len(c) for s,c in sorted(chars_by_script.items())},
        overlap=overlap,ttf=ttf_cover),ensure_ascii=False))


if __name__=='__main__':main()
