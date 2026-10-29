"""S5-P125 Stage-D: build the P0 manifest, writer slots and frozen char sets.

Inputs (4090):
  assets/generic48_kxl_top10_20261022/train.csv   100k generic rows, labels UNKNOWN -> null
  assets/train_fame-kxl-tj-px60.csv               curated trusted writers
  assets/train_hcsu_kxl.csv                       curated trusted writers
  assets/u100_p1p2_20261025/p2_train.csv          Top10 training rows (23,485)
  assets/u100_pretrain_20261024/std_shape_bank.pt STD bank keys (coverage filter)

Outputs (assets/p125_20261030/):
  p0_train.csv          writer_slot column: Top10 0-9 reserved, generic 10.., null last
  p125_manifest.json    char sets, slot map, provenance, sha256
Not trained on: Top10 rows (P0); All-Stages unseen chars excluded from P0/P1/P2.
"""
import argparse
import csv
import hashlib
import json
from collections import Counter,defaultdict
from pathlib import Path
import sys

TOP10=["颜真卿","苏轼","王羲之","褚遂良","米芾","欧阳询","柳公权","赵孟頫","黄庭坚","文徵明"]
GENERIC_ROOT="assets/generic48_kxl_top10_20261022"
BANK="assets/u100_pretrain_20261024/std_shape_bank.pt"
TOP10_TRAIN="assets/u100_p1p2_20261025/p2_train.csv"
OUT=Path("assets/p125_20261030")


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def bank_keys(path=BANK):
    import torch
    data=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
    return {tuple(k) for k in data['keys']}


def rows_of(path):
    return list(csv.DictReader(open(path,encoding='utf-8-sig')))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',default=str(OUT))
    a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    keys=bank_keys()
    print('STD bank keys:',len(keys))
    top10_train=rows_of(TOP10_TRAIN)
    top10_chars=defaultdict(set)
    for r in top10_train:top10_chars[(int(r['script_id']),r['character'])].add(r['character'])
    top10_train_char_set={r['character'] for r in top10_train}
    # 1) generic null pool
    null_rows=[]
    for r in rows_of(f"{GENERIC_ROOT}/train.csv"):
        key=(int(r['script_id']),r['character'])
        if key not in keys:continue
        null_rows.append(r)
    # 2) curated trusted generic writers
    trusted=[];writer_counts=Counter();dropped_std=0;dropped_top10=0;dropped_file=0
    seen_paths=set()
    for path in ("assets/train_fame-kxl-tj-px60.csv","assets/train_hcsu_kxl.csv"):
        for r in rows_of(path):
            if r['calligrapher'] in TOP10:dropped_top10+=1;continue
            if (r.get('aug') or '').strip():continue
            if r['image_path'] in seen_paths:continue
            if not Path(r['image_path']).is_file():dropped_file+=1;continue
            key=(int(r['script_id']),r['character'])
            if key not in keys:dropped_std+=1;continue
            seen_paths.add(r['image_path'])
            r['character_id']=str(ord(r['character']))  # u100 char remap keys are codepoints
            trusted.append(r);writer_counts[r['calligrapher']]+=1
    # slot allocation: keep writers with >=200 images, others -> null (untrusted-small)
    SLOT_MIN=200
    keep_writers=sorted(w for w,c in writer_counts.items() if c>=SLOT_MIN)
    demoted=sum(c for w,c in writer_counts.items() if c<SLOT_MIN)
    slot_map={name:i for i,name in enumerate(TOP10)}
    generic_slots={name:10+i for i,name in enumerate(keep_writers)}
    null_slot=10+len(keep_writers)
    for r in trusted:
        w=r['calligrapher']
        r['writer_slot']=generic_slots.get(w,null_slot)
        r['writer_trusted']='1' if w in generic_slots else '0'
    for r in null_rows:
        r['writer_slot']=null_slot;r['writer_trusted']='0'
    p0=null_rows+trusted
    for i,r in enumerate(p0):
        if not (r.get('img_id') or '').strip():r['img_id']=str(31_000_000+i)
    # 3) frozen char sets
    p0_chars=defaultdict(set)
    for r in p0:p0_chars[(int(r['script_id']),r['character'])].add(r['character'])
    all_stage_unseen={}
    rng=hashlib.sha256(b'p125-unseen').hexdigest()
    per_key=list(sorted(p0_chars.keys()|top10_chars.keys()))
    held=set()
    for key in per_key:
        if int(hashlib.sha256((rng+str(key)).encode()).hexdigest(),16)%100<8:held.add(key)
    held_list=[list(k) for k in sorted(held)]
    provenance={pth:sha(pth) for pth in (f"{GENERIC_ROOT}/train.csv",TOP10_TRAIN,
        "assets/train_fame-kxl-tj-px60.csv","assets/train_hcsu_kxl.csv",BANK)}
    manifest=dict(
        top10=TOP10,
        slot_map=dict(top10={n:i for n,i in slot_map.items()},
                      generic_trusted={w:int(i) for w,i in generic_slots.items()},
                      null_slot=null_slot,n_generic_writers=len(keep_writers),
                      demoted_small_writers=demoted),
        counts=dict(p0_rows=len(p0),null_rows=len(null_rows),trusted_rows=len(trusted),
            dropped_top10=dropped_top10,dropped_std_uncovered=dropped_std,dropped_missing_file=dropped_file),
        char_sets=dict(
            top10_train_chars=len(top10_train_char_set),
            p0_train_chars=sum(len(v) for v in p0_chars.values()),
            all_stage_unseen_keys=held_list,
            all_stage_unseen_note='8% deterministic hash holdout of (script,char) keys, excluded from P0/P1/P2 targets'),
        provenance=provenance)
    fieldnames=[]
    for r in p0:
        for k in r:
            if k not in fieldnames:fieldnames.append(k)
    with open(out/'p0_train.csv','w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fieldnames,extrasaction='ignore')
        w.writeheader();w.writerows(p0)
    save=lambda pth,v:Path(pth).write_text(json.dumps(v,ensure_ascii=False,indent=2))
    save(out/'p125_manifest.json',manifest)
    print(json.dumps(dict(p0_rows=len(p0),null=len(null_rows),trusted=len(trusted),
        generic_writers=len(keep_writers),demoted=demoted,null_slot=null_slot,
        unseen_keys=len(held)),ensure_ascii=False))


if __name__=='__main__':main()
