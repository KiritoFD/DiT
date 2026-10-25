"""Recover supervision labels on the existing protected K/X/L pool; no new images."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path


def read(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def write(path, rows):
    with Path(path).open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def prepare(source, raw, out):
    source=Path(source);out=Path(out);out.mkdir(parents=True,exist_ok=False)
    records=read(raw)
    # Original source names are restored by exact image path, not by incompatible ID spaces.
    by_file={Path(r['image_path']).name:r for r in records}
    splits={name:read(source/file) for name,file in [('train','train.csv'),('validation','validation.csv')]}
    keys=set()
    for rows in splits.values():
        for row in rows:
            original=by_file[Path(row['image_path']).name]
            if original['script']!=row['script'] or original['character']!=row['character']:
                raise ValueError('Original label mismatch')
            if row['script'] not in ('楷','行','隶'):raise ValueError('Only K/X/L is allowed')
            row['style_writer_key']=original['source']+'|'+original['calligrapher']
            row['style_writer_name']=original['calligrapher']
            keys.add(row['style_writer_key'])
    mapping={key:i for i,key in enumerate(sorted(keys))}
    for rows in splits.values():
        for row in rows:row['style_writer_id']=str(mapping[row['style_writer_key']])
    groups=defaultdict(set)
    for row in splits['train']:
        groups[row['style_writer_id'],row['script_id']].add(row['character'])
    eligible=[];excluded=[]
    for row in splits['train']:
        (eligible if len(groups[row['style_writer_id'],row['script_id']])>=2 else excluded).append(row)
    train_pixels={r['pixel_sha256'] for r in eligible}
    if train_pixels & {r['pixel_sha256'] for r in splits['validation']}:
        raise ValueError('Train/validation exact pixel overlap')
    write(out/'train.csv',eligible);write(out/'validation.csv',splits['validation'])
    if excluded:write(out/'excluded_no_crosschar.csv',excluded)
    report=dict(original_train=len(splits['train']),train=len(eligible),validation=len(splits['validation']),
        excluded_no_crosschar=len(excluded),scripts=dict(Counter(r['script'] for r in eligible)),
        writers=len({r['style_writer_id'] for r in eligible}),source_scoped_writer=True,
        policy='Only existing protected K/X/L pool; no Top10 training mix-in; no test image as reference',
        source_proxy_caveat='source|writer avoids cross-dataset ID collisions; source is not verified artwork',
        input_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                      [Path(raw),source/'train.csv',source/'validation.csv']})
    (out/'audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--raw',required=True)
    p.add_argument('--out',required=True);a=p.parse_args();prepare(a.source,a.raw,a.out)
