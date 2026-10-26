"""Generate S/J specs and a conservative two-machine 64h proposal; no training."""
import argparse
import copy
import hashlib
from dataclasses import fields
import json
from pathlib import Path
from src.unet.io import Config
from ..runtime.common import save,sha

ARMS={'S0':(0,0),'S1':(1,0),'S2':(0,2),'S3':(1,2),'S4':(0,1),
      'S5':(2,1),'S6':(3,1),'S7':(0,3),'S8':(3,3),'S2-large':(0,2)}
HOSTS={'4090':['S0','S2','S3','S4','S5','S7','S2-large'],'48':['S1','S6','S8','J0','J1','J2','J3']}


def main():
    p=argparse.ArgumentParser();p.add_argument('--base-config',required=True);p.add_argument('--p1',required=True)
    p.add_argument('--p2',required=True);p.add_argument('--measurements',required=True);p.add_argument('--joint-results',required=True)
    p.add_argument('--out',required=True);p.add_argument('--eval-csv',required=True)
    p.add_argument('--writer-space',choices=['raw','top10-row'],required=True)
    p.add_argument('--wall-hours',type=float,default=64.);p.add_argument('--reserve-hours',type=float,default=8.)
    a=p.parse_args();out=Path(a.out)
    if out.exists():raise FileExistsError('Use a new immutable plan directory')
    base=json.loads(Path(a.base_config).read_text());allowed={f.name for f in fields(Config)}
    base={k:v for k,v in base.items() if k in allowed}
    base.update(batch=512,compile=False,freeze_writer_table=False,lr=2e-5,weight_decay=.01,warmup=300,
        ema_decay=.999,ema_interval=1,pair_fraction=.3,initialize_from='',std_shape_step_offset=0,
        checkpoint_steps=[],stop_at_step=0,ckpt_every=5000,eval_every=5000,seed=90)
    for legacy in ('study_spec','semantic_writer_rows'):base.pop(legacy,None)
    measured={host:{v['injection']:v['chosen'] for v in json.loads((Path(a.measurements)/f'{host}_fastest.json').read_text())}
              for host in HOSTS}
    joint=json.loads(Path(a.joint_results).read_text())['arms']
    if set(joint)!=set(('J0','J1','J2','J3')):raise ValueError('Need all four actual J preflights')
    if any(v['global_batch']!=512 or v['gradient_checkpointing'] for v in joint.values()):raise ValueError('Probe protocol changed')
    rates={host:{arm:(joint[arm]['step_s'] if arm.startswith('J') else measured[host][ARMS[arm][1]]['step_s'])
                 for arm in arms} for host,arms in HOSTS.items()}
    # Upper bound for two local S survivors, then one; all J arms reach phase2, two reach phase3.
    def phases(host,ssteps,jsteps):
        sr=sorted([1/r for arm,r in rates[host].items() if arm.startswith('S')],reverse=True)
        jr=sorted([1/r for arm,r in rates[host].items() if arm.startswith('J')],reverse=True)
        return [(ssteps[0]*sum(sr)+jsteps[0]*sum(jr))/3600,
            ((ssteps[1]-ssteps[0])*sum(sr[:2])+(jsteps[1]-jsteps[0])*sum(jr))/3600,
            ((ssteps[2]-ssteps[1])*sum(sr[:1])+(jsteps[2]-jsteps[1])*sum(jr[:2]))/3600]
    def cost(host,ssteps,jsteps):return sum(phases(host,ssteps,jsteps))+a.reserve_hours
    def wall(ssteps,jsteps):
        return sum(max(phases(h,ssteps,jsteps)[i] for h in HOSTS) for i in range(3))+a.reserve_hours
    ssteps=[10000,30000,45000];jsteps=[4000,12000,20000]
    # Preserve first-stage minima; uniformly shorten extensions if the upper bound exceeds64h.
    for percent in range(100,-1,-1):
        ss=[10000]+[10000+int((v-10000)*percent/100/500)*500 for v in ssteps[1:]]
        jj=[4000]+[4000+int((v-4000)*percent/100/500)*500 for v in jsteps[1:]]
        if wall(ss,jj)<=a.wall_hours:ssteps,jsteps=ss,jj;break
    else:raise ValueError('First-stage minimum cannot fit wall budget')
    contract=dict(p1_sha256=sha(a.p1),p2_sha256=sha(a.p2),train_csv_sha256=sha(base['data_csv']),
        eval_csv_sha256=sha(a.eval_csv),global_batch=512,seed=90,gradient_checkpointing=False)
    for host,arms in HOSTS.items():
        directory=out/host;stages={str(i):[] for i in (1,2,3)}
        for arm in arms:
            config=copy.deepcopy(base);ladder=jsteps if arm.startswith('J') else ssteps
            config.update(max_steps=ladder[-1],lr_decay_steps=ladder[-1],eval_csv=a.eval_csv)
            if arm.startswith('S'):
                r,i=ARMS[arm];config.update(style_matrix_repr=r,style_matrix_injection=i,
                    style_matrix_vector_dim=512 if arm=='S2-large' else 32,microbatch=measured[host][i]['microbatch'])
            else:config.update(style_matrix_repr=-1,microbatch=joint[arm]['fm_microbatch'],std_shape_step_offset=20000)
            spec=dict(arm=arm,machine=host,kind='joint' if arm.startswith('J') else 'style',config=config,
                source=a.p2 if arm.startswith('J') else a.p1,milestones=sorted(set([1000,3000]+ladder)),
                identity=dict(arm=arm,contract=contract,horizon=ladder[-1],std_offset=config['std_shape_step_offset'],
                    config_sha256=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()))
            if arm.startswith('J'):
                spec['joint']=dict(arm=arm,image_micro=joint[arm]['image_microbatch'],image_batch=joint[arm]['image_batch'],
                                   calibration=str(out/'joint_calibration.json'))
            path=directory/'specs'/f'{arm}.json';save(path,spec)
            for i,stop in enumerate(ladder,1):stages[str(i)].append(dict(arm=arm,spec=str(path),spec_sha256=sha(path),
                stop=stop,previous_stop=0 if i==1 else ladder[i-2]))
        plan=dict(machine=host,root=str(directory/'execution'),ready=str(directory/'ready.json'),contract=contract,
            wall_hours=a.wall_hours,estimated_upper_hours=wall(ssteps,jsteps),reserve_hours=a.reserve_hours,
            promotion_limits={'2':dict(S=2,J=4 if host=='48' else 0),'3':dict(S=1,J=2 if host=='48' else 0)},
            stages=stages,evaluations=[dict(name='heldout',csv=a.eval_csv,writer_space=a.writer_space)])
        save(directory/'plan.json',plan)
        python='/opt/conda/envs/cu121/bin/python' if host=='4090' else '/home/ds/miniconda3/envs/vcp-embryo/bin/python'
        repo='/root/Workspace/xy/DiT' if host=='4090' else '/home/ds/Workspace/DiT'
        script=directory/'run_stage.sh'
        script.write_text('#!/usr/bin/env bash\nset -euo pipefail\ncd '+repo+'\n'+
            'export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True\n'+
            'exec '+python+' -u -m tools.style_matrix.runtime.serial --plan '+str(directory/'plan.json')+' "$@"\n')
    save(out/'budget.json',dict(S_steps=ssteps,J_steps=jsteps,effective_batch=512,wall_hours=a.wall_hours,
        upper_hours={h:cost(h,ssteps,jsteps) for h in HOSTS},natural_wall_upper_hours=wall(ssteps,jsteps),
        phase_training_hours={h:phases(h,ssteps,jsteps) for h in HOSTS},reserve_hours=a.reserve_hours,
        rates=rates,quality_selection_required=True,launchable=False,
        pending=['common asset ready manifests','J gradient calibration','real GT pairing','runner resume preflight','S2-large capacity']))
    print(json.dumps(json.loads((out/'budget.json').read_text()),indent=2))


if __name__=='__main__':main()
