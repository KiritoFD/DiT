"""Exp A/B/C on 48: pretraining causal ablation, all local (no 4090 weights).

  P0s: 30k generic pretrain (mccd corpus 12,945 rows, null writer, 75% drop)
  B:   FT 15k Top10 directly from P0s@30k      (cum stops 35k/40k/45k)
  C1:  5k mixed (70% generic-null/30% Top10)   (cum stop 35k)
  C2:  10k Top10 from C1@35k                   (cum stops 40k/45k)
  A1:  45k Top10 scratch, phase1 0-30k         (cum stops 5k/10k/15k/25k/30k)
  A2:  15k continuation with optimizer reset   (cum stops 35k/45k)

All-Stages unseen chars (8% deterministic hash holdout) are excluded from every
training CSV. LR restarts at every phase boundary are explicit
(schedule_restart with origin). Evaluations: val100 at each listed stop.
"""
import argparse
import copy
import csv
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
import sys
sys.path.insert(0,'.')
from tools.style_matrix.runtime.common import save

OUT=Path("assets/p125_20261030/abc48")
GEN="assets/p125_20261030/train100k_min.csv"
TOP="assets/u100_p1p2_20261025/p2_train.csv"
BANK="assets/u100_pretrain_20261024/std_shape_bank.pt"
CHAR_REMAP_SRC="assets/u100_pretrain_20261024/tables/char_remap.json"
VAL100="assets/u100_pretrain_20261024/validation_val100.csv"
HELD_SHARE=8


def bank_keys():
    import torch
    d=torch.load(BANK,map_location='cpu',weights_only=True,mmap=True)
    return {tuple(k) for k in d['keys']}


def rows_of(path):
    return list(csv.DictReader(open(path,encoding='utf-8-sig')))


def held_keys(all_keys):
    rng=hashlib.sha256(b'abc48-unseen').hexdigest()
    return {k for k in all_keys
            if int(hashlib.sha256((rng+str(k)).encode()).hexdigest(),16)%100<HELD_SHARE}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',default=str(OUT))
    a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    keys=bank_keys();held=held_keys(keys)
    # corpora
    def cov(r):
        try:return (int(r['script_id']),r['character']) in keys
        except Exception:return False
    gen=[r for r in rows_of(GEN) if cov(r)]  # latent-only training: no image_path needed
    top=[r for r in rows_of(TOP) if not (r.get('aug') or '').strip() and cov(r)]
    gen=[dict(r,calligrapher_id='47',generic='1') for r in gen]  # 47 -> null row (num_writers=47... see below)
    # writer slots: Top10 0-9 + null=10 -> num_writers=10, table rows 11
    for r in top:r['calligrapher_id']=r['calligrapher_id']  # canonical 0-9 already
    remap_null=dict({str(i):i for i in range(10)},**{'47':10})  # generic rows carry 47? no:
    # generic rows keep calligrapher_id=-1 -> remap '-1':10
    remap_null=dict({str(i):i for i in range(10)},**{'10':10})
    remap_id={str(i):i for i in range(10)}
    # char remap union (u100 + any missing column values)
    u100=json.loads(Path(CHAR_REMAP_SRC).read_text())
    all_cids=set()
    for rs in (gen,top):
        for r in rs:all_cids.add(str(r['character_id']))
    char_remap=dict(u100)
    for c in sorted(all_cids):
        if c not in char_remap:char_remap[c]=len(char_remap)
    Path(out/'char_remap_abc48.json').write_text(json.dumps(char_remap,indent=1))
    # stage CSVs (unseen excluded)
    def wcsv(name,rs):
        with open(out/name,'w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(rs[0].keys()),extrasaction='ignore')
            w.writeheader();w.writerows(rs)
    p0s_rows=[r for r in gen if (int(r['script_id']),r['character']) not in held]
    ft_rows=[r for r in top if (int(r['script_id']),r['character']) not in held]
    # C1 mixed: generic rows x2 + top10 subsample to 70/30
    rng=random.Random(48125)
    g_n=len(p0s_rows);t_n=int(round(g_n*3/7))
    t_pick=sorted(rng.sample(range(len(ft_rows)),min(t_n,len(ft_rows))))
    c1_rows=[dict(r,calligrapher_id='10') for r in p0s_rows]+ \
            [dict(ft_rows[i],calligrapher_id=ft_rows[i]['calligrapher_id']) for i in t_pick]
    rng.shuffle(c1_rows)
    wcsv('stage_p0s.csv',p0s_rows);wcsv('stage_ft.csv',ft_rows);wcsv('stage_c1.csv',c1_rows)
    Path(out/'remap_null.json').write_text(json.dumps(remap_null,indent=1))
    Path(out/'remap_id.json').write_text(json.dumps(remap_id,indent=1))
    # base config from the 48_v3 S5 spec (architecture/data conventions)
    base=json.loads(Path('assets/style_focus_85k_20261029/48_v3/specs/S5.json').read_text())['config']
    def base_cfg(stage):
        c=copy.deepcopy(base)
        c.update(num_writers=10,batch=512,compile=True,null_writer_rows=False,
            char_remap_json=str(out/'char_remap_abc48.json'),
            latent_shards_dir='assets/generic48_kxl_top10_20261022/latents',
            data_csv=str(out/f'stage_{stage}.csv'),
            ckpt_every=5000,eval_every=5000,workers=8)
        return c
    contract=dict(abc48='v1',global_batch=512,seed=90,compile=True,
        gradient_checkpointing=False,num_writers=10,
        latent_root='assets/generic48_kxl_top10_20261022/latents')
    specs={}
    def spec(name,stage,cfg,source=None,continues=None,joint=None,restart=None):
        s=dict(arm=name,machine='48',
            kind='joint' if joint else 'style',
            initialization='scratch' if stage in ('p0s','a1') else 'stage_transfer',
            source=source,config=cfg,
            milestones=sorted(set([1000,5000]+[cfg['max_steps']])),
            identity=dict(arm=name,contract=contract,config=cfg))
        if restart:s['schedule_restart']=restart
        if joint:s['joint']=joint
        if continues:s['continues_identity']=continues
        path=out/'specs'/f'{name}.json';save(path,s)
        return s
    # P0s
    c=base_cfg('p0s');c.update(lr=7e-5,warmup=1500,lr_decay_steps=30000,max_steps=30000,
        drop_writer=0.75,pair_fraction=0.3,callig_remap_json=str(out/'remap_null.json'))
    specs['P0s']=spec('P0s','p0s',c)
    # A1 phase1: scratch Top10 0-30k
    c=base_cfg('a1');c.update(lr=7e-5,warmup=1500,lr_decay_steps=30000,max_steps=30000,
        drop_writer=0.1,pair_fraction=0.3,callig_remap_json=str(out/'remap_id.json'))
    specs['A1']=spec('A1','a1',c)
    # A2: restart at 30000, 15k cosine at 1e-5
    c=base_cfg('a2');c.update(lr=1e-5,lr_backbone=1e-5,lr_style_repr=5e-5,lr_style_inj=5e-5,
        lr_writer_table=2.5e-5,warmup=200,lr_decay_steps=15000,max_steps=15000,
        drop_writer=0.1,pair_fraction=0.3,callig_remap_json=str(out/'remap_id.json'))
    specs['A2']=spec('A2','a2',c,restart=dict(origin=30000,horizon=15000,warmup=200,peak_scale=1.0),
        continues=specs['A1']['identity'])
    # B: FT 15k from P0s@30000
    c=base_cfg('b');c.update(lr=1e-5,lr_backbone=1e-5,lr_style_repr=5e-5,lr_style_inj=5e-5,
        lr_writer_table=2.5e-5,warmup=200,lr_decay_steps=15000,max_steps=15000,
        drop_writer=0.1,pair_fraction=0.3,callig_remap_json=str(out/'remap_id.json'))
    specs['B']=spec('B','b',c,restart=dict(origin=30000,horizon=15000,warmup=200,peak_scale=1.0),
        continues=specs['P0s']['identity'])
    # C1: mixed 5k from P0s@30000
    c=base_cfg('c1');c.update(lr=1e-5,lr_backbone=1e-5,lr_style_repr=5e-5,lr_style_inj=5e-5,
        lr_writer_table=2.5e-5,warmup=200,lr_decay_steps=5000,max_steps=5000,
        drop_writer=0.25,callig_remap_json=str(out/'remap_null.json'),null_writer_rows=True)
    specs['C1']=spec('C1','c1',c,restart=dict(origin=30000,horizon=5000,warmup=200,peak_scale=1.0),
        continues=specs['P0s']['identity'])
    # C2: Top10 10k from C1@5000
    c=base_cfg('c2');c.update(lr=1e-5,lr_backbone=1e-5,lr_style_repr=5e-5,lr_style_inj=5e-5,
        lr_writer_table=2.5e-5,warmup=200,lr_decay_steps=10000,max_steps=10000,
        drop_writer=0.1,pair_fraction=0.3,callig_remap_json=str(out/'remap_id.json'))
    specs['C2']=spec('C2','c2',c,restart=dict(origin=35000,horizon=10000,warmup=200,peak_scale=1.0),
        continues=specs['C1']['identity'])
    # jobs: cumulative stops on the P0s->B/C clock; A1/A2 separate arms
    def job(arm,stop):
        path=out/'specs'/f'{arm}.json'
        import hashlib
        return dict(arm=arm,spec=str(path),spec_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            stop=stop,previous_stop=0)
    plans={}
    def plan(name,stages_jobs,wall):
        st={str(i):v for i,v in enumerate(stages_jobs,1)}
        save(out/f'plan_{name}.json',dict(machine='48',root=str(out/'execution'),
            ready=str(out/'ready.json'),contract=contract,wall_hours=wall,
            estimated_upper_hours=0,promotion_limits={},stages=st,
            evaluations=[dict(name='heldout',csv=VAL100,writer_space='top10-row')]))
    plan('p0s',[[job('P0s',30000)]],6.0)
    plan('b',[[job('B',35000),job('B',40000),job('B',45000)]],3.2)
    plan('c',[[job('C1',35000),job('C2',40000),job('C2',45000)]],3.2)
    plan('a',[[job('A1',5000),job('A1',10000),job('A1',15000),job('A1',25000),job('A1',30000)],
              [job('A2',35000),job('A2',45000)]],8.5)
    # seal ready.json: hash small files; large assets recorded by path+size only
    files=[]
    seeds=set()
    for st in plans.values():
        pass
    def add_file(f):
        f=Path(f)
        if not f.is_file():raise FileNotFoundError(f)
        if f.stat().st_size>1<<20:
            files.append(dict(path=str(f),sha256=None,size=f.stat().st_size))
        else:
            files.append(dict(path=str(f),sha256=sha_of(str(f))))
    def sha_of(p):
        import hashlib
        return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    seen_p=set()
    def add_tree(root,pattern):
        for f in Path(root).glob(pattern):
            if f.is_file() and str(f) not in seen_p:
                seen_p.add(str(f));add_file(f)
    for st in plans.values():
        for job in st:
            add_file(job['spec'])
    for name in ('stage_p0s.csv','stage_ft.csv','stage_c1.csv','remap_null.json',
                 'remap_id.json','char_remap_abc48.json'):
        add_file(out/name)
    add_file(VAL100)
    add_file(BANK)
    add_tree('assets/generic48_kxl_top10_20261022/latents','*.npz')
    add_tree('assets/generic48_kxl_top10_20261022/std_bbox64','index.csv')
    for f in Path('pretrained_models/sd-vae-ft-ema').rglob('*'):
        if f.is_file():add_file(f)
    for root in ('src','tools'):
        for f in Path(root).rglob('*.py'):
            add_file(f)
    files.sort(key=lambda d:d['path'])
    save(out/'ready.json',dict(contract=contract,files=files,no_training_started=True))
    run='''#!/usr/bin/env bash
set -euo pipefail
cd /home/ds/Workspace/DiT
PY=/home/ds/miniconda3/envs/vcp-embryo/bin/python
ROOT={root}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
"$PY" -u -m tools.style_matrix.runtime.serial --plan "$ROOT/plan_p0s.json" --execute
for D in B C1; do
  mkdir -p "$ROOT/execution/runs/$D/checkpoints"
  cp "$ROOT"/execution/runs/P0s/checkpoints/0030000.pt "$ROOT/execution/runs/$D/checkpoints/"
  cp "$ROOT"/execution/runs/P0s/checkpoints/0030000.pt.done "$ROOT/execution/runs/$D/checkpoints/"
  cp "$ROOT/execution/runs/P0s/identity.json" "$ROOT/execution/runs/$D/"
done
"$PY" -u -m tools.style_matrix.runtime.serial --plan "$ROOT/plan_b.json" --execute
"$PY" -u -m tools.style_matrix.runtime.serial --plan "$ROOT/plan_c.json" --execute
mkdir -p "$ROOT/execution/runs/A2/checkpoints"
cp "$ROOT"/execution/runs/A1/checkpoints/0030000.pt "$ROOT/execution/runs/A2/checkpoints/"
cp "$ROOT"/execution/runs/A1/checkpoints/0030000.pt.done "$ROOT/execution/runs/A2/checkpoints/"
cp "$ROOT/execution/runs/A1/identity.json" "$ROOT/execution/runs/A2/"
"$PY" -u -m tools.style_matrix.runtime.serial --plan "$ROOT/plan_a.json" --execute
echo ABC48-DONE
'''.format(root=str(out))
    (out/'run_abc48.sh').write_text(run);(out/'run_abc48.sh').chmod(0o755)
    print(json.dumps(dict(p0s_rows=len(p0s_rows),ft_rows=len(ft_rows),c1_rows=len(c1_rows),
        held_keys=len(held),slots='Top10 0-9 + null=10 (num_writers=10)',
        generic_writer_policy='null-only (mccd labels UNKNOWN)',
        plans=['plan_p0s','plan_b','plan_c','plan_a'])))
    json.dump(dict(p0s_rows=len(p0s_rows),ft_rows=len(ft_rows),held=len(held)),
        open(out/'abc48_summary.json','w'))


if __name__=='__main__':main()
