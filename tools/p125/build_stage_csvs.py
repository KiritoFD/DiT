"""S5-P125: build per-stage training CSVs from the frozen P0 manifest.

Design: mixtures are realized by CSV composition (deterministic row subsets and
capped replication), NOT by a new sampler — the existing OffsetSampler resume
determinism is preserved exactly. All-Stages unseen (script,char) keys are
excluded from every stage. calligrapher_id is rewritten to the writer_slot so a
single identity remap covers all stages (num_writers=48).
"""
import argparse
import csv
import hashlib
import json
import random
from collections import Counter,defaultdict
from pathlib import Path

OUT=Path("assets/p125_20261030")
GENERIC_SHARDS="assets/generic48_kxl_top10_20261022/latents"
TRUSTED_SHARDS="assets/p125_20261030/latents_trusted"


def rows_of(path):
    return list(csv.DictReader(open(path,encoding='utf-8-sig')))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',default=str(OUT))
    a=p.parse_args()
    out=Path(a.out)
    manifest=json.loads((out/'p125_manifest.json').read_text())
    null_slot=manifest['slot_map']['null_slot'];n_writers=null_slot+1
    held={tuple(k) for k in manifest['char_sets']['all_stage_unseen_keys']}
    def not_held(r):return (int(r['script_id']),r['character']) not in held
    p0=rows_of(out/'p0_train.csv')
    # Backfill empty img_ids from the null-missing encode CSV (same derivation order:
    # null rows, p0_train.csv order, not covered by the generic48 shards at build time).
    enc_missing=rows_of(out/'p125_null_missing_encode.csv')
    path_to_id={r['image_path']:r['img_id'] for r in enc_missing}
    for r in p0:
        if r['image_path'] in path_to_id:
            r['img_id']=path_to_id[r['image_path']]
    top10=[r for r in rows_of("assets/u100_p1p2_20261025/p2_train.csv") if not (r.get('aug') or '').strip()]
    # coverage check: Top10 img_ids must be in the encoded corpus
    enc_ids={r['img_id'] for r in rows_of("assets/generic48_kxl_top10_20261022/encode.csv")}
    missing=sum(1 for r in top10 if r['img_id'] not in enc_ids)
    print('top10 rows',len(top10),'img_ids missing from encode.csv:',missing)
    def with_slot(r,slot):
        r=dict(r);r['calligrapher_id']=str(slot);return r
    # P0
    stage_p0=[with_slot(r,r['writer_slot']) for r in p0 if not_held(r)]
    # P1: 70% generic null + 30% top10, by row counts
    null_pool=[r for r in stage_p0 if int(r['calligrapher_id'])==null_slot]
    top10_pool=[with_slot(r,int(r['calligrapher_id'])) for r in top10 if not_held(r)]
    n_top=len(top10_pool);n_null=int(round(n_top*7/3))
    rng=random.Random(125001)
    null_pick=sorted(rng.sample(range(len(null_pool)),min(n_null,len(null_pool))))
    p1_null=[null_pool[i] for i in null_pick]
    stage_p1=p1_null+top10_pool
    # P2: 50% frequency + 50% slot-balanced (capped replication)
    slots=defaultdict(list)
    for i,r in enumerate(top10_pool):slots[(r['calligrapher'],r['script'])].append(i)
    max_slot=max(len(v) for v in slots.values())
    rng2=random.Random(125002)
    balanced=[]
    for slot,idxs in sorted(slots.items()):
        cap=min(max_slot,len(idxs)*4)  # replication cap 4x per row
        pick=[idxs[i%len(idxs)] for i in range(cap)]
        rng2.shuffle(pick);balanced+=pick
    target_total=len(top10_pool)+len(balanced)
    freq_n=target_total//2;bal_n=target_total-freq_n
    freq_pick=sorted(rng2.sample(range(len(top10_pool)),min(freq_n,len(top10_pool))))
    scale=bal_n/max(len(balanced),1)
    bal_take=[balanced[int(i)] for i in np_linspace(scale,len(balanced))]
    stage_p2=[top10_pool[i] for i in freq_pick]+[top10_pool[i] for i in bal_take]
    rng3=random.Random(125003);rng3.shuffle(stage_p2)
    # remap + write
    remap={str(i):i for i in range(n_writers)}  # identity 0..47; 47->47 is the architecture null row
    (out/'callig_remap_p125.json').write_text(json.dumps(remap,indent=1))
    # char remap: union of the u100 table plus ord(c) for every character in the
    # stage CSVs (the matrix route ignores y_char, but build_data requires coverage).
    u100=json.load(open('assets/u100_pretrain_20261024/tables/char_remap.json'))
    all_cids=set()
    for rs in (stage_p0,stage_p1,stage_p2):
        for r in rs:all_cids.add(str(r['character_id']))
    missing_cids=sorted(c for c in all_cids if c not in u100)
    char_remap=dict(u100)
    for c in missing_cids:char_remap[c]=len(char_remap)
    (out/'char_remap_p125.json').write_text(json.dumps(char_remap,indent=1))
    print('char remap: u100',len(u100),'added',len(missing_chars),'total',len(char_remap))
    report=dict(n_writers=n_writers,null_slot=null_slot,held_keys=len(held),
        top10_missing_encode_ids=missing,
        p0=len(stage_p0),
        p1=dict(total=len(stage_p1),null=len(p1_null),top10=len(top10_pool),
            null_share=round(len(p1_null)/len(stage_p1),3)),
        p2=dict(total=len(stage_p2),freq=freq_n,balanced=bal_n,
            slots={f'{w}_{s}':len(idxs) for (w,s),idxs in sorted(slots.items())}),
        seeds=dict(p1=125001,p2_balance=125002,p2_shuffle=125003,
            p0_shuffle=125004,p1_shuffle=125005))
    # shard coverage check: every stage row's img_id must exist in some shard
    shard_ids=set()
    for f in Path('assets/p125_20261030/shards_all').glob('*.npz'):
        import numpy as np
        shard_ids|={str(x) for x in np.load(f)['img_ids'].tolist()}
    for name,rs in (('p0',stage_p0),('p1',stage_p1),('p2',stage_p2)):
        missing=sum(1 for r in rs if str(r['img_id']) not in shard_ids)
        if missing:raise ValueError(f'{name}: {missing} rows without latents')
    print('shard coverage OK; shards:',len(shard_ids),'distinct img_ids')
    def write(name,rs):
        with open(out/name,'w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(rs[0].keys()),extrasaction='ignore')
            w.writeheader();w.writerows(rs)
    # Uniform per-batch 75/25 (P0) and 70/30 (P1) mixtures require shuffling:
    # sequential blocks would give the style branch zero gradient for hours.
    rng4=random.Random(125004);rng4.shuffle(stage_p0)
    rng5=random.Random(125005);rng5.shuffle(stage_p1)
    write('stage_p0.csv',stage_p0);write('stage_p1.csv',stage_p1);write('stage_p2.csv',stage_p2)
    (out/'p125_stages.json').write_text(json.dumps(report,ensure_ascii=False,indent=1))
    print(json.dumps({k:v for k,v in report.items() if k!='p2'},ensure_ascii=False))
    print('p2 slots:',{k:v for k,v in list(report['p2']['slots'].items())[:5]},'...')


def np_linspace(scale,n):
    import numpy as np
    idx=np.linspace(0,n-1,int(round(n*scale)))
    return [int(round(i)) for i in idx]


if __name__=='__main__':main()
