"""Build the 64h extension-round plans (S8, J3, paired/12k extensions) from the
sealed 20261027 plans. Reuses prior spec files byte-identically so existing
checkpoints resume, and seals everything including the extended-round sources.

Phase 1 (default): write specs, plan.json skeleton, run_stage.sh, migrate old runs.
Phase 2 (--seal --rates RATES.json): patch measured estimates, then seal ready.json.
"""
import argparse
import copy
import csv
import json
import shutil
from pathlib import Path
from ..runtime.common import save,sha

S_ARMS={'S0':(0,0),'S1':(0,1),'S2':(0,2),'S3':(1,2),'S4':(2,1),'S5':(3,1),'S6':(0,3),'S7':(3,3),'S8':(2,3)}
S_PORT={'4090':['S0','S1','S4','S6','S8'],'48':['S2','S3','S5','S7']}
S_PAIR_CANDIDATES={'4090':['S1','S4','S6','S8']}
PY={'4090':'/opt/conda/envs/cu121/bin/python','48':'/home/ds/miniconda3/envs/vcp-embryo/bin/python'}
REPO={'4090':'/root/Workspace/xy/DiT','48':'/home/ds/Workspace/DiT'}
RESIDENT={'4090':' --resident-module directuv_surface.scripts.prepare_object','48':''}


def load_spec(path):
    return json.loads(Path(path).read_text())


def job(arm,spec_path,stop,previous_stop=0):
    return dict(arm=arm,spec=str(spec_path),spec_sha256=sha(spec_path),stop=stop,previous_stop=previous_stop)


def copy_run(old_root,new_root,arm,manifest):
    """Migrate a previous plan's run dir only when it holds a real checkpoint."""
    old=Path(old_root)/'runs'/arm
    if not old.exists():return 0
    if not list(old.glob('checkpoints/*.pt')):return 0
    new=Path(new_root)/'runs'/arm
    if new.exists():raise FileExistsError('Migration target exists: '+str(new))
    shutil.copytree(old,new)
    latest=sorted(new.glob('checkpoints/*.pt'))[-1].stem
    manifest.append(dict(arm=arm,from_plan=str(old),step=int(latest)))
    return int(latest)


def build_s_spec(base_spec,arm,machine,out,contract):
    spec=copy.deepcopy(load_spec(base_spec))
    r,i=S_ARMS[arm]
    spec['config']['style_matrix_repr']=r
    spec['config']['style_matrix_injection']=i
    spec['config']['style_matrix_vector_dim']=32
    spec['config']['callig_remap_json']=str(out/'callig_remap.json')
    spec.update(arm=arm,machine=machine,kind='style',initialization='scratch',source=None)
    spec['identity']=dict(arm=arm,contract=contract,config=spec['config'])
    return spec


def build_ext_spec(spec,*,stop,schedule,arm=None):
    """Extension spec: same science, longer clock, explicit controlled LR restart."""
    ext=copy.deepcopy(spec)
    ext['config']['max_steps']=stop
    ext['config']['lr_decay_steps']=stop
    ext['milestones']=sorted(set(ext['milestones'])|{stop})
    ext['identity']=dict(spec['identity'],config=ext['config'])
    ext['continues_identity']=spec['identity']
    ext['schedule_restart']=schedule
    if arm:ext['arm']=arm
    return ext


def estimate(jobs,migrated,rates,protocols,eval_min,compile_min,reserve_h,stage2_s_limit=None):
    seconds=0.;evals=0
    ext=[j for j in jobs if j['previous_stop']>0 and j['arm'].startswith('S')]
    if stage2_s_limit and len(ext)>stage2_s_limit:
        ext=sorted(ext,key=lambda j:-rates[j['arm']])[:stage2_s_limit]
        jobs=[j for j in jobs if not (j['previous_stop']>0 and j['arm'].startswith('S'))]+ext
    for j in jobs:
        spec=load_spec(j['spec'])
        rate=rates.get(j['arm'])
        if not rate:raise ValueError('Missing measured rate for '+j['arm'])
        start=migrated.get(j['arm'],0) if j['previous_stop']==0 else j['previous_stop']
        seconds+=(j['stop']-start)/rate
        evals+=sum(1 for p in protocols if not p.get('arms') or j['arm'] in p['arms'])
    hours=seconds/3600+evals*eval_min/60+len(jobs)*compile_min/60+reserve_h
    return round(hours,2)


def seal(out,plan_path,rates_path,eval_min,compile_min,reserve_h):
    plan=json.loads(Path(plan_path).read_text())
    rates=json.loads(Path(rates_path).read_text())
    migrated={m['arm']:m['step'] for m in json.loads((out/'migrations.json').read_text())}
    flat=[j for stage in plan['stages'].values() for j in stage]
    plan['estimated_upper_hours']=estimate(flat,migrated,rates,plan['evaluations'],eval_min,compile_min,reserve_h,
        stage2_s_limit=plan.get('promotion_limits',{}).get('2',{}).get('S'))
    plan['rate_basis']='Probe-measured compiled step/s per arm on this machine; estimate incl. eval/compile overhead'
    save(plan_path,plan)
    for j in flat:
        if sha(j['spec'])!=j['spec_sha256']:raise ValueError('Spec changed before seal: '+j['arm'])
    for arm,step in migrated.items():
        runs=Path(plan['root'])/'runs'/arm
        if not list(runs.glob('checkpoints/*.pt')):raise ValueError('Migrated run lost its checkpoints: '+arm)
    fresh=[p for p in (Path(plan['root'])/'runs').glob('*/checkpoints/*.pt')
           if p.parent.parent.name not in migrated]
    if fresh:raise ValueError('Training started before seal: '+str(fresh[:2]))
    paths={str(out/'callig_remap.json'),*[j['spec'] for j in flat]}
    seen_cfg=set()
    for j in flat:
        spec=load_spec(j['spec']);cfg=spec['config']
        key=json.dumps(cfg,sort_keys=True)
        if key in seen_cfg:continue
        seen_cfg.add(key)
        for key_ in ('data_csv','eval_csv','std_shape_cache','char_remap_json','callig_remap_json','font_remap_json'):
            paths.add(cfg[key_])
        for key_ in ('latent_shards_dir','std_soft_root','vae_path'):
            files=[str(f) for f in Path(cfg[key_]).rglob('*') if f.is_file()]
            if not files:raise FileNotFoundError(cfg[key_])
            paths.update(files)
        if spec['kind']=='joint':
            cal=spec['joint']['calibration'];paths.add(cal)
            paths.update(r['image_path'] for r in csv.DictReader(open(cfg['data_csv'],encoding='utf-8-sig')))
    for root in ('src','tools/style_matrix'):
        paths.update(str(f) for f in Path(root).rglob('*.py'))
    if (out/'ready.json').exists():raise FileExistsError('Already sealed')
    # Seal discipline: hash small files (code/specs/CSV headers) to catch edits;
    # record path+size only for large assets — a 64h run does not re-hash GBs.
    files=[]
    for f in sorted(paths):
        size=Path(f).stat().st_size
        files.append(dict(path=f,sha256=None,size=size) if size>1<<20 else dict(path=f,sha256=sha(f)))
    save(out/'ready.json',dict(contract=plan['contract'],files=files,no_training_started=True))
    print('SEALED',out,'estimated_upper_hours',plan['estimated_upper_hours'],flush=True)


def write_run_stage(machine,out):
    repo=REPO[machine];py=PY[machine];root=str(out)
    if machine=='4090':
        script=f'''#!/usr/bin/env bash
set -euo pipefail
cd {repo}
export TORCHINDUCTOR_CACHE_DIR=/root/.cache/torch/inductor
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY={py}
ROOT={root}
"$PY" -u -m tools.style_matrix.runtime.serial --plan "$ROOT/plan.json" --execute{RESIDENT[machine]}
SELECTION="$ROOT/selection.json"
echo "stage 1 complete; waiting up to 24h for stage-2 pair selection"
for i in $(seq 1 288); do [ -f "$SELECTION" ] && break; sleep 300; done
[ -f "$SELECTION" ] || {{ echo "no selection.json after 24h; stopping cleanly"; exit 0; }}
exec "$PY" -u -m tools.style_matrix.runtime.serial --plan "$ROOT/plan.json" --stage 2 --execute --selection "$SELECTION"{RESIDENT[machine]}
'''
    else:
        script=f'''#!/usr/bin/env bash
set -euo pipefail
cd {repo}
export TORCHINDUCTOR_CACHE_DIR=/home/ds/.cache/torch/inductor
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec {py} -u -m tools.style_matrix.runtime.serial --plan {root}/plan.json --execute
'''
    path=out/'run_stage.sh';path.write_text(script)
    path.chmod(0o755)


def main():
    p=argparse.ArgumentParser();p.add_argument('--machine',choices=('4090','48'),required=True)
    p.add_argument('--out',required=True);p.add_argument('--old-plan',required=True)
    p.add_argument('--s-base-spec',default='',help='48 only: path to the scratch S0 spec json')
    p.add_argument('--steps',type=int,default=50000);p.add_argument('--joint-steps',type=int,default=5000)
    p.add_argument('--ext-s-steps',type=int,default=75000);p.add_argument('--ext-j-steps',type=int,default=12000)
    p.add_argument('--wall-hours',type=float,default=64.)
    p.add_argument('--seal',action='store_true');p.add_argument('--rates',default='')
    p.add_argument('--eval-min',type=float,default=12.);p.add_argument('--compile-min',type=float,default=2.5)
    p.add_argument('--reserve-h',type=float,default=2.)
    a=p.parse_args();out=Path(a.out)
    if a.seal:
        if not a.rates:raise ValueError('--seal needs --rates')
        return seal(out,out/'plan.json',a.rates,a.eval_min,a.compile_min,a.reserve_h)
    if out.exists():raise FileExistsError('Use a new immutable plan directory')
    old=Path(a.old_plan);old_plan=json.loads((old/'plan.json').read_text())
    out.mkdir(parents=True)
    save(out/'callig_remap.json',{str(i):i for i in range(10)})
    manifest=[];migrated={}
    specs=out/'specs';specs.mkdir()
    if a.machine=='4090':
        contract=old_plan['contract'];eval_protocol=old_plan['evaluations'][0]
        stage1=[];ext_specs=[]
        for arm in S_PORT['4090']:
            src=old/'specs'/f'{arm}.json'
            if arm=='S8':
                spec=build_s_spec(old/'specs'/'S0.json','S8','4090',out,contract)
                path=specs/'S8.json';save(path,spec)
            else:
                if not src.exists():raise FileNotFoundError(src)
                path=specs/f'{arm}.json';shutil.copyfile(src,path)
            stage1.append(job(arm,path,a.steps))
            step=copy_run(old/'execution',out/'execution',arm,manifest)
            if step:migrated[arm]=step
        for arm in S_PAIR_CANDIDATES['4090']:
            ext=build_ext_spec(load_spec(specs/f'{arm}.json'),stop=a.ext_s_steps,
                schedule=dict(horizon=a.ext_s_steps-a.steps,warmup=200,peak_scale=.5))
            ext_specs.append(job(arm,save(specs/f'{arm}_ext75.json',ext) or specs/f'{arm}_ext75.json',a.ext_s_steps,a.steps))
        plan=dict(machine='4090',root=str(out/'execution'),ready=str(out/'ready.json'),contract=contract,
            wall_hours=a.wall_hours,estimated_upper_hours=0,
            promotion_limits={'2':dict(S=2,J=0)},stages={'1':stage1,'2':ext_specs},
            evaluations=[dict(name=eval_protocol['name'],csv=eval_protocol['csv'],writer_space=eval_protocol['writer_space'])],
            pair_note='Stage 2 extends ONE reviewed pair to 75k; candidates S1/S4/S6/S8; choose after 50k heldout+writer metrics')
    else:
        if not a.s_base_spec:raise ValueError('48 needs --s-base-spec (scratch S0 spec)')
        j_contract=old_plan['contract'];j_eval=old_plan['evaluations'][0]
        stage1=[]
        for arm in ('J0','J1','J2'):
            src=old/'specs'/f'{arm}.json'
            if not src.exists():raise FileNotFoundError(src)
            path=specs/f'{arm}.json';shutil.copyfile(src,path)
            stage1.append(job(arm,path,a.joint_steps))
            step=copy_run(old/'execution',out/'execution',arm,manifest)
            if step:migrated[arm]=step
        j2=load_spec(old/'specs'/'J2.json')
        j3=copy.deepcopy(j2)
        j3['arm']='J3';j3['joint']=dict(arm='J4',image_micro=j2['joint']['image_micro'],
            image_batch=j2['joint']['image_batch'],calibration=j2['joint']['calibration'])
        j3['identity']=dict(arm='J3',internal_objective='J4',contract=j_contract,config=j2['config'])
        path=specs/'J3.json';save(path,j3)
        stage1.append(job('J3',path,a.joint_steps))
        s_contract=dict(initialization='scratch',global_batch=512,seed=90,compile=True,
            gradient_checkpointing=False,train_csv_sha256=sha(json.loads(Path(a.s_base_spec).read_text())['config']['data_csv']),
            eval_csv_sha256=sha(json.loads(Path(a.s_base_spec).read_text())['config']['eval_csv']),steps=a.steps)
        for arm in S_PORT['48']:
            spec=build_s_spec(a.s_base_spec,arm,'48',out,s_contract)
            spec['milestones']=[5000,10000,25000,a.steps]
            # 48 disk headroom: 4 S arms at the 4090 cadence would write ~143GB.
            spec['config']['ckpt_every']=25000;spec['config']['eval_every']=25000
            path=specs/f'{arm}.json';save(path,spec)
            stage1.append(job(arm,path,a.steps))
        for arm in ('J0','J1','J2','J3'):
            base=load_spec(specs/f'{arm}.json')
            ext=build_ext_spec(base,stop=a.ext_j_steps,
                schedule=dict(horizon=a.ext_j_steps-a.joint_steps,warmup=200,peak_scale=.5))
            ext['joint']=base['joint']
            stage1.append(job(arm,save(specs/f'{arm}_ext12.json',ext) or specs/f'{arm}_ext12.json',a.ext_j_steps,a.joint_steps))
        s_eval_csv=json.loads(Path(a.s_base_spec).read_text())['config']['eval_csv']
        plan=dict(machine='48',root=str(out/'execution'),ready=str(out/'ready.json'),contract=j_contract,
            wall_hours=a.wall_hours,estimated_upper_hours=0,
            promotion_limits={'2':dict(S=0,J=4)},stages={'1':stage1},
            evaluations=[dict(name=j_eval['name'],csv=j_eval['csv'],writer_space=j_eval['writer_space'],
                              arms=['J0','J1','J2','J3']),
                         dict(name='heldout_s',csv=s_eval_csv,writer_space='top10-row',arms=['S2','S3','S5','S7'])])
    save(out/'plan.json',plan)
    save(out/'migrations.json',manifest)
    write_run_stage(a.machine,out)
    print(json.dumps(dict(plan=str(out/'plan.json'),migrated=migrated,
        stage1=[j['arm'] for j in plan['stages']['1']],
        stage2=[j['arm'] for j in plan['stages'].get('2',[])]),indent=2))


if __name__=='__main__':main()
