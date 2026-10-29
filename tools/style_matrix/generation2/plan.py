"""Versioned generation-two recipes, semantic eval mapping and local forks."""
import copy
import csv
import hashlib
import json
from pathlib import Path
from tools.style_matrix.runtime.common import save,sha


def read(path):return json.loads(Path(path).read_text())


def csv_write(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def protocols(root,spec,family):
    cfg=spec['config'];rows=list(csv.DictReader(open('exp-std/csv/eval200_fixed.csv',encoding='utf-8-sig')))
    train=list(csv.DictReader(open(cfg['data_csv'],encoding='utf-8-sig')))
    if family=='S':
        writers={r['calligrapher']:r['calligrapher_id'] for r in train}
        scripts={r['script']:r['script_id'] for r in train}
        for r in rows:
            char=r.get('char_used','')
            if len(char)!=1:char=r['character']
            if len(char)!=1:raise ValueError('Evaluation target is not a single Unicode character')
            r.update(character=char,character_id=str(ord(char)),glyph_id=str(ord(char)),
                     calligrapher_id=writers[r['calligrapher']],script_id=scripts[r['script']])
    results=[]
    for seed in (0,1,2):
        rr=copy.deepcopy(rows)
        for r in rr:
            # Shared character/script => shared noise across Writer, too.
            key=f"{seed}:{r['script']}:{r['character']}".encode()
            r['eval_noise_seed']=str(int.from_bytes(hashlib.sha256(key).digest()[:8],'big')%(2**63))
        path=root/'protocols'/f'{family}_187_seed{seed}.csv';csv_write(path,rr)
        results.append(dict(name=f'heldout187_seed{seed}',csv=str(path),writer_space='raw'))
    # Diagnostic shared-character panel: training GT, explicitly not a heldout score.
    cells={}
    for r in train:
        if (r.get('aug') or '').strip():continue
        cells.setdefault((r['script'],r['character']),{}).setdefault(r['calligrapher'],r)
    chosen=[]
    for script in sorted({k[0] for k in cells}):
        keys=sorted((k for k,v in cells.items() if k[0]==script and len(v)>=3),key=lambda k:(-len(cells[k]),k))[:2]
        for key in keys:chosen.extend(list(cells[key].values())[:5])
    if not chosen:raise ValueError('No shared-character diagnostic cells')
    fields=set().union(*(r.keys() for r in chosen));panel=[]
    for r in chosen:
        rr={k:r.get(k,'') for k in sorted(fields)}
        rr['eval_noise_seed']=str(int.from_bytes(hashlib.sha256(f"panel:{r['script']}:{r['character']}".encode()).digest()[:8],'big')%(2**63))
        panel.append(rr)
    path=root/'protocols'/f'{family}_shared_train_panel.csv';csv_write(path,panel)
    results.append(dict(name='shared_train_panel',csv=str(path),writer_space='raw'))
    return results


def recipes(root,machine):
    old=Path('assets/style_matrix_20261027/matrix64')/machine
    first=Path('assets/style_matrix_20261028/generation2')/machine
    specs={}
    if machine=='4090':
        source=read(old/'specs/S0.json');base=copy.deepcopy(source)
        base.update(arm='S9',kind='velocity',initialization='scratch',source=None)
        base['config'].update(style_matrix_repr=-1,writer_velocity='cnn',max_steps=50000,lr_decay_steps=50000,
                              batch=512,microbatch=256,compile=True,ckpt_every=5000,eval_every=5000)
        base['identity']=dict(generation=2,arm='S9',config=base['config'],source=None,joint=None,
                             parent_contract=source['identity']['contract'])
        specs['S9']=base
    else:
        for arm in ('J4','J5','J6','J7'):specs[arm]=read(first/'specs'/f'{arm}.json')
        source=read(old/'specs/S2.json');base=copy.deepcopy(source)
        base.update(arm='S11',kind='velocity',initialization='scratch',source=None)
        base['config'].update(style_matrix_repr=-1,writer_velocity='attention',max_steps=50000,lr_decay_steps=50000,
                              batch=512,microbatch=256,compile=True,ckpt_every=5000,eval_every=5000)
        base['identity']=dict(generation=2,arm='S11',config=base['config'],source=None,joint=None,
                             parent_contract=source['identity']['contract'])
        specs['S11']=base
    if machine=='4090':
        for arm,mode,seed in (('S10','geometry',90),('S9_seed91','cnn',91),('S10_seed91','geometry',91)):
            spec=copy.deepcopy(base);spec['arm']=arm;spec['config'].update(writer_velocity=mode,seed=seed)
            spec['identity']=dict(generation=2,arm=arm,config=spec['config']);specs[arm]=spec
        for arm in ('S9FM','S9Pair'):
            spec=continuation(base,arm,50000,60000)
            if arm=='S9Pair':
                spec['velocity_pair_weight']=1.;spec['velocity_pair_gradient_ratio']=.25
                spec['identity']['velocity_pair_gradient_ratio']=.25
            specs[arm]=spec
        for arm in ('S9FM','S9Pair','S10'):
            specs[arm+'_75']=continuation(specs[arm],arm,60000 if arm!='S10' else 50000,75000)
        for arm in ('S0','S4'):
            specs[arm]=read(old/'specs'/f'{arm}.json')
            specs[arm+'_75']=continuation(specs[arm],arm,50000,75000)
    else:
        for arm in ('J4','J5','J6','J7'):specs[arm+'_12']=continuation(specs[arm],arm,5000,12000)
        specs['S11_75']=continuation(specs['S11'],'S11',50000,75000)
        spec=copy.deepcopy(specs['S11']);spec['arm']='S11_seed91';spec['config']['seed']=91
        spec['identity']=dict(generation=2,arm=spec['arm'],config=spec['config']);specs[spec['arm']]=spec
    for key,spec in specs.items():
        spec['milestones']=[s for s in (1000,5000,10000,25000,50000,60000,75000) if s<=spec['config']['max_steps']]
        save(root/'specs'/f'{key}.json',spec)
    return specs


def continuation(base,arm,start,stop):
    spec=copy.deepcopy(base);spec['arm']=arm;spec['continues_identity']=base['identity']
    spec['config'].update(max_steps=stop,lr_decay_steps=stop)
    spec['schedule_restart']=dict(origin=start,horizon=stop-start,warmup=200,peak_scale=.5)
    spec['identity']=dict(generation=2,arm=arm,config=spec['config'],parent=base['identity'],restart=spec['schedule_restart'],
                          velocity_pair_gradient_ratio=spec.get('velocity_pair_gradient_ratio',0.))
    return spec


def seal(root,machine):
    old=Path('assets/style_matrix_20261027/matrix64')/machine/'ready.json'
    files={item['path']:item for item in read(old)['files']}
    for directory in ('src','tools/style_matrix'):
        for path in Path(directory).rglob('*.py'):files[str(path)]=dict(path=str(path),sha256=sha(path))
    path=Path('tools/eval_gen_with_disc.py');files[str(path)]=dict(path=str(path),sha256=sha(path))
    for path in Path('assets/style_matrix_20261027').rglob('writer_proto.pt'):
        files[str(path)]=dict(path=str(path),sha256=sha(path))
    path=Path('assets/style_disc_ce_holdout.pt')
    files[str(path)]=dict(path=str(path),size=path.stat().st_size,sha256=None)
    for directory in ('specs','protocols'):
        for path in (root/directory).rglob('*'):
            if path.is_file():files[str(path)]=dict(path=str(path),sha256=sha(path))
    save(root/'ready.json',dict(contract=dict(generation=2,machine=machine,batch=512),files=list(files.values())))
