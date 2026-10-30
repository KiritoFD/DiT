"""S5-P125 serial plan builder: three chained serial invocations with runtime
materialization between them.

  serial_a: P0 100k scratch (FM only, 75% writer drop, 48 slots)
  materialize: init_p1_table on P0@100k EMA
  serial_b: P1 5k (70/30 mix, table LR 5e-5) -> P2 15k (Top10, 50/50 + pairs)
  materialize: copy P2@15000 run dirs -> P3A/P3B; run EG pilot gate
  serial_c: P3A (FM continue) + P3B (E-G assisted) to 20000 (125k cumulative)

The stage CSVs and shards_all are built by build_stage_csvs.py first.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
from ..style_matrix.runtime.common import save

OUT=Path("assets/p125_20261030")
REPO="/root/Workspace/xy/DiT"
PY="/opt/conda/envs/cu121/bin/python"
TOP10_CSV="assets/u100_p1p2_20261025/p2_train.csv"
VAL100="assets/u100_pretrain_20261024/validation_val100.csv"


def stage_config(base,stage):
    c=copy.deepcopy(base)
    c['num_writers']=47
    # Memory: global batch 512 stays one FM graph per microbatch choice. Measured
    # on this box: micro=512 -> 20.4GiB peak (+1.1GiB outside-container resident),
    # micro=256 -> 11.1GiB. 512 is the throughput default; if a foreign job grabs
    # the card and OOMs us, fix by dropping to 256, never by touching the allocator.
    c['microbatch']=512
    c['callig_remap_json']=str(OUT/'callig_remap_p125.json')
    c['latent_shards_dir']=str(OUT/'shards_all')
    c['std_shape_cache']='assets/u100_pretrain_20261024/std_shape_bank.pt'
    c['std_soft_root']='assets/generic48_kxl_top10_20261022/std_bbox64'
    c['null_writer_rows']=True
    c['char_remap_json']=str(OUT/'char_remap_p125.json')
    if stage=='p0':
        c.update(data_csv=str(OUT/'stage_p0.csv'),lr=7e-5,warmup=1500,
            lr_decay_steps=100000,max_steps=100000,drop_writer=0.75,
            ckpt_every=10000,eval_every=10000)
    elif stage=='p1':
        c.update(data_csv=str(OUT/'stage_p1.csv'),lr=1e-5,lr_backbone=1e-5,
            lr_style_repr=1e-5,lr_style_inj=1e-5,lr_writer_table=5e-5,
            warmup=200,lr_decay_steps=5000,max_steps=5000,drop_writer=0.25,
            ckpt_every=2500,eval_every=2500)
    elif stage=='p2':
        c.update(data_csv=str(OUT/'stage_p2.csv'),lr=1e-5,lr_backbone=1e-5,
            lr_style_repr=1e-5,lr_style_inj=1e-5,lr_writer_table=5e-5,
            warmup=200,lr_decay_steps=20000,max_steps=15000,drop_writer=0.1,
            pair_fraction=0.3,ckpt_every=5000,eval_every=5000)
    elif stage in ('p3a','p3b'):
        c.update(data_csv=str(OUT/'stage_p2.csv'),lr=1e-5,lr_backbone=1e-5,
            lr_style_repr=1e-5,lr_style_inj=1e-5,lr_writer_table=5e-5,
            warmup=200,lr_decay_steps=20000,max_steps=20000,drop_writer=0.1,
            pair_fraction=0.3,ckpt_every=5000,eval_every=5000)
    return c


def make_spec(arm,stage,base,contract,continues=None,source=None,joint=None):
    config=stage_config(base,stage)
    spec=dict(arm=arm,machine='4090',kind='joint' if joint else 'style',
        initialization='scratch' if stage=='p0' else 'stage_transfer',
        source=source,config=config,
        milestones=sorted(set([1000,5000]+([10000,50000,75000] if stage=='p0' else [])+[config['max_steps']])),
        identity=dict(arm=arm,contract=contract,config=config))
    if stage!='p0':
        # Carry the whole style branch, not just the shared backbone: P1/P2/P3 are
        # LR restarts of the same network, and a random style table would erase P0.
        spec['transfer_style']=True
    if joint:spec['joint']=joint
    if continues:spec['continues_identity']=continues
    return spec


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--base-spec',default='assets/style_matrix_20261027/matrix64/4090/specs/S8.json')
    p.add_argument('--eg-refs',default=str(OUT/'eg_refs.npz'))
    a=p.parse_args()
    base=json.loads(Path(a.base_spec).read_text())['config']
    contract=dict(p125='v1',global_batch=512,seed=90,compile=True,gradient_checkpointing=False,
        num_writers=47,latent_root=str(OUT/'shards_all'))
    out=OUT
    specs={}
    def emit(name,spec,stop=None):
        path=out/'specs'/f'{name}.json';save(path,spec)
        return dict(arm=name,spec=str(path),spec_sha256=sha_of(path),
            stop=stop or spec['config']['max_steps'],previous_stop=0)
    def sha_of(path):import hashlib;return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    (out/'specs').mkdir(parents=True,exist_ok=True)
    # P0
    specs['P0']=make_spec('P0','p0',base,contract)
    # P1: source = materialized table-init checkpoint (path fixed at plan time)
    runs=out/'execution'/'runs'
    p1_source=str(out/'p1_init.pt')
    specs['P1']=make_spec('P1','p1',base,contract,source=p1_source)
    # P2 continues P1's final weights; P3A/P3B continue P2@15k (they also get the
    # checkpoint copied into their own run dirs, in which case resume wins).
    p1_ckpt=str(runs/'P1'/'checkpoints'/'0005000.pt')
    p2_ckpt=str(runs/'P2'/'checkpoints'/'0015000.pt')
    specs['P2']=make_spec('P2','p2',base,contract,source=p1_ckpt)
    specs['P3A']=make_spec('P3A','p3a',base,contract,continues=specs['P2']['identity'],source=p2_ckpt)
    specs['P3B']=make_spec('P3B','p3b',base,contract,continues=specs['P2']['identity'],source=p2_ckpt,
        joint=dict(arm='EG',refs_npz=a.eg_refs,every=4,image_micro=4,target_ratio=0.02))
    jobs={}
    # 2026-10-30 user ruling: cut P0 at 50k. The CPU val100 trajectory plateaus
    # from ~20k (ssim 0.557/0.560/0.557 at 20/30/40k) while train-side fm keeps
    # dropping (-13%) — pure seen-side gain beyond that point. Spec stays at
    # max_steps=100000 so the run identity (and resume) is untouched; only the
    # serial stop boundary moves.
    jobs['P0']=emit('P0',specs['P0'],stop=50000)
    for name in ('P1','P2','P3A','P3B'):
        jobs[name]=emit(name,specs[name])
    def plan(stages,wall):
        return dict(machine='4090',root=str(out/'execution'),ready=str(out/'ready.json'),
            contract=contract,wall_hours=wall,estimated_upper_hours=0,
            promotion_limits={},stages={str(i):v for i,v in enumerate(stages,1)},
            evaluations=[dict(name='heldout',csv=VAL100,writer_space='top10-row')])
    save(out/'plan_a.json',plan([[jobs['P0']]],20.))
    save(out/'plan_b.json',plan([[jobs['P1'],jobs['P2']]],7.))
    save(out/'plan_c.json',plan([[jobs['P3A'],jobs['P3B']]],4.5))
    run=f'''#!/usr/bin/env bash
set -euo pipefail
cd {REPO}
export TORCHINDUCTOR_CACHE_DIR=/root/.cache/torch/inductor
# 铁律: PYTORCH_CUDA_ALLOC_CONF=expandable_segments 一律禁用。显存不够只能调小
# microbatch（见 specs 里的 256），不许动分配器。
PY={PY}
ROOT={out}
# The ~1.1GiB reservation lives outside this container with an unstable PID; the
# serial runner needs a nonzero --resident-pid to tolerate it rotating.
RESIDENT_PID=${{RESIDENT_PID:-4045410}}
"$PY" -u -m tools.style_matrix.runtime.serial --plan "$ROOT/plan_a.json" --execute --resident-pid "$RESIDENT_PID"
# P0 cut-over boundary (50k, see plan_a stage stop).
P0CK=$(ls "$ROOT"/execution/runs/P0/checkpoints/0050000.pt)
"$PY" -u -m tools.p125.init_p1_table --source "$P0CK" --out "$ROOT/p1_init.pt"
# serial pins clock.json to the first plan's sha256, so each plan segment must
# reset the (per-segment) wall clock before starting its own runner.
rm -f "$ROOT"/execution/clock.json
"$PY" -u -m tools.style_matrix.runtime.serial --plan "$ROOT/plan_b.json" --execute --resident-pid "$RESIDENT_PID"
for ARM in P3A P3B; do
  mkdir -p "$ROOT/execution/runs/$ARM/checkpoints"
  cp "$ROOT"/execution/runs/P2/checkpoints/0015000.pt "$ROOT/execution/runs/$ARM/checkpoints/"
  cp "$ROOT"/execution/runs/P2/checkpoints/0015000.pt.done "$ROOT/execution/runs/$ARM/checkpoints/"
  cp "$ROOT/execution/runs/P2/identity.json" "$ROOT/execution/runs/$ARM/"
done
"$PY" -u -m tools.p125.pilot_eg --plan "$ROOT/plan_c.json" --out "$ROOT/pilot_eg.json"
"$PY" -u -c "import json;d=json.load(open('$ROOT/pilot_eg.json'));assert d['passed'],d"
rm -f "$ROOT"/execution/clock.json
"$PY" -u -m tools.style_matrix.runtime.serial --plan "$ROOT/plan_c.json" --execute --resident-pid "$RESIDENT_PID"
'''
    (out/'run_p125.sh').write_text(run)
    (out/'run_p125.sh').chmod(0o755)
    print(json.dumps(dict(plans=['plan_a.json','plan_b.json','plan_c.json'],
        specs={k:v['spec'] for k,v in jobs.items()},
        p3a_continues_p2=True,p3b_eg=True),indent=1))


if __name__=='__main__':main()
