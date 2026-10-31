"""2026-10-31 [A/B/C] 生成 spec / plan / ready 并校验。

A：P3A 续训 +25k（旧配方延长，新 LR 表）——counter 20000→45000
B：P0@50k 直接 Top10 适配 +25k——counter 50000→75000
C1：P0@50k Writer 强化 5k——counter 50000→55000
C2：C1 联合适配 20k——counter 55000→75000
所有 arm 均以 schedule_restart 取得全新 cosine（200 warmup），峰值 LR 取自本 spec。
"""
import hashlib, json, sys
from pathlib import Path

R=Path('/root/Workspace/xy/DiT')
ROOT=R/'assets/abc_20261031'
(SPEC:=ROOT/'specs').mkdir(parents=True,exist_ok=True)

def sha(p):return hashlib.sha256((R/p).read_bytes()).hexdigest()

p0=json.loads((R/'assets/p125_20261030/specs/P0.json').read_text())
p3=json.loads((R/'assets/p125_20261030/specs/P3A.json').read_text())
base=p3['config']
contract=p3['identity']['contract']

def cfg_for(lrs,max_steps,decay):
    c=dict(base)
    c.update(lr_backbone=lrs[0],lr_style_repr=lrs[1],lr_style_inj=lrs[2],
             lr_writer_table=lrs[3],max_steps=max_steps,lr_decay_steps=decay,
             output='assets/abc_20261031/execution/runs')
    return c

LR_ABC=[1e-5,5e-5,5e-5,2.5e-5]      # A / C2: Backbone 1e-5, Style 5e-5, Writer Table 2.5e-5
LR_C1=[2e-6,5e-5,5e-5,1e-4]        # C1: Backbone 2e-6, Style 5e-5, Writer Table 1e-4

arms={
 'A':dict(cfg=cfg_for(LR_ABC,45000,25000),
          restart=dict(origin=20000,horizon=25000,warmup=200,peak_scale=1.0),
          milestones=[25000,35000,45000],continues=p3['identity'],
          source='assets/p125_20261030/execution/runs/P3A/checkpoints/0020000.pt'),
 'B':dict(cfg=cfg_for(LR_ABC,75000,25000),
          restart=dict(origin=50000,horizon=25000,warmup=200,peak_scale=1.0),
          milestones=[75000],continues=p0['identity'],
          source='assets/p125_20261030/execution/runs/P0/checkpoints/0050000.pt'),
 'C1':dict(cfg=cfg_for(LR_C1,55000,5000),
          restart=dict(origin=50000,horizon=5000,warmup=200,peak_scale=1.0),
          milestones=[55000],continues=p0['identity'],
          source='assets/p125_20261030/execution/runs/P0/checkpoints/0050000.pt'),
 'C2':dict(cfg=cfg_for(LR_ABC,75000,20000),
          restart=dict(origin=55000,horizon=20000,warmup=200,peak_scale=1.0),
          milestones=[75000],continues=None,  # 由 C1 identity 填充
          source='assets/abc_20261031/execution/runs/C1/checkpoints/0055000.pt'),
}

specs={}
for name,spec in arms.items():
    s=dict(arm=name,machine='4090',kind='style',initialization='stage_transfer',
           source=spec['source'],config=spec['cfg'],
           milestones=spec['milestones'],
           identity=dict(arm=name,contract=contract,config=spec['cfg']),
           schedule_restart=spec['restart'])
    if spec['continues'] is not None:s['continues_identity']=spec['continues']
    specs[name]=s

specs['C2']['continues_identity']=dict(specs['C1']['identity'])
for name,s in specs.items():
    (SPEC/f'{name}.json').write_text(json.dumps(s,ensure_ascii=False,indent=1),encoding='utf-8')

# 断言:resume 身份链闭合
assert specs['A']['continues_identity']['arm']=='P3A'
assert specs['B']['continues_identity']['arm']=='P0'
assert specs['C1']['continues_identity']['arm']=='P0'
assert specs['C2']['continues_identity']['arm']=='C1'

proto='assets/style_focus_85k_20261029/4090/protocols'
S4='assets/style_focus_85k_20261029/4090/protocols'
plan=dict(machine='4090',root='assets/abc_20261031/execution',
    ready='assets/abc_20261031/ready.json',contract=contract,
    wall_hours=40,estimated_upper_hours=0,promotion_limits={},
    stages={'1':[
        dict(arm='A',spec='assets/abc_20261031/specs/A.json',spec_sha256=sha('assets/abc_20261031/specs/A.json'),
             stop=25000,previous_stop=0,fork_from='assets/p125_20261030/execution/runs/P3A/checkpoints/0020000.pt'),
        dict(arm='A',spec='assets/abc_20261031/specs/A.json',spec_sha256=sha('assets/abc_20261031/specs/A.json'),
             stop=35000,previous_stop=25000),
        dict(arm='A',spec='assets/abc_20261031/specs/A.json',spec_sha256=sha('assets/abc_20261031/specs/A.json'),
             stop=45000,previous_stop=35000),
        dict(arm='B',spec='assets/abc_20261031/specs/B.json',spec_sha256=sha('assets/abc_20261031/specs/B.json'),
             stop=75000,previous_stop=0,fork_from='assets/p125_20261030/execution/runs/P0/checkpoints/0050000.pt'),
        dict(arm='C1',spec='assets/abc_20261031/specs/C1.json',spec_sha256=sha('assets/abc_20261031/specs/C1.json'),
             stop=55000,previous_stop=0,fork_from='assets/p125_20261030/execution/runs/P0/checkpoints/0050000.pt'),
        dict(arm='C2',spec='assets/abc_20261031/specs/C2.json',spec_sha256=sha('assets/abc_20261031/specs/C2.json'),
             stop=75000,previous_stop=55000),
    ]},
    evaluations=[
        dict(name='heldout187_seed0',csv=f'{proto}/S_187_seed0.csv',writer_space='top10-row',arms=['A','B','C2']),
        dict(name='heldout187_seed1',csv=f'{proto}/S_187_seed1.csv',writer_space='top10-row',arms=['A','B','C2']),
        dict(name='heldout187_seed2',csv=f'{proto}/S_187_seed2.csv',writer_space='top10-row',arms=['A','B','C2']),
        dict(name='heldout',csv='assets/u100_pretrain_20261024/validation_val100.csv',writer_space='top10-row',arms=['A','B','C2']),
    ])
(ROOT/'plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=1),encoding='utf-8')

# ready.json:继承 p125 封存清单,重算 sha(train.py 已改),加入新 spec
old=json.loads((R/'assets/p125_20261030/ready.json').read_text())
files=[]
seen=set()
for item in old['files']:
    p=item['path']
    if p in seen:continue
    seen.add(p)
    if (R/p).is_dir():
        files.append(dict(item));continue
    entry=dict(path=p,size=(R/p).stat().st_size)
    if (R/p).stat().st_size<=(1<<20):entry['sha256']=sha(p)
    else:entry['sha256']=item.get('sha256','')
    files.append(entry)
for p in ['assets/abc_20261031/specs/A.json','assets/abc_20261031/specs/B.json',
          'assets/abc_20261031/specs/C1.json','assets/abc_20261031/specs/C2.json']:
    files.append(dict(path=p,size=(R/p).stat().st_size,sha256=sha(p)))
ready=dict(contract=contract,files=files,
           total_bytes=sum(f['size'] for f in files),
           built_from='assets/p125_20261030/ready.json + A/B/C specs (2026-10-31)')
(ROOT/'ready.json').write_text(json.dumps(ready,ensure_ascii=False,indent=1),encoding='utf-8')

# 关键路径存在性校验
for p in [arms['A']['source'],arms['B']['source'],
          f'{proto}/S_187_seed0.csv',f'{proto}/S_187_seed1.csv',f'{proto}/S_187_seed2.csv',
          'assets/u100_pretrain_20261024/validation_val100.csv',
          'assets/p125_20261030/stage_p2.csv',
          'assets/p125_20261030/shards_all']:
    assert (R/p).exists(),p
print('BUILT OK:',[f'assets/abc_20261031/specs/{n}.json' for n in specs],'plan.json ready.json')
