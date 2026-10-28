"""Generate independent, sealed first-24h plans from verified local assets."""
import argparse
import copy
import json
from pathlib import Path
from tools.style_matrix.runtime.common import save,sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--machine',choices=['4090','48'],required=True);a=p.parse_args()
    old=Path('assets/style_matrix_20261027/matrix64')/a.machine
    root=Path('assets/style_matrix_20261028/generation2')/a.machine
    if (root/'plan.json').exists():raise FileExistsError('Preserve existing experiment identity')
    jobs=[];spec_paths=[]
    arms=('S9',) if a.machine=='4090' else ('J4','J5','J6','J7')
    for arm in arms:
        source=json.loads((old/'specs'/('S0.json' if arm=='S9' else 'J3.json')).read_text())
        spec=copy.deepcopy(source);cfg=spec['config'];spec['arm']=arm
        cfg.update(compile=True,batch=512,microbatch=256,ckpt_every=5000,eval_every=5000)
        if arm=='S9':
            cfg.update(style_matrix_repr=-1,writer_velocity='cnn',max_steps=50000,lr_decay_steps=50000)
            spec.update(kind='velocity',initialization='scratch',source=None)
        else:
            spec['joint']['arm']='G2'+arm
            spec['joint'].update(image_micro=8,image_batch=16)
            cfg.update(max_steps=5000,lr_decay_steps=5000)
        spec['identity']=dict(generation=2,arm=arm,config=cfg,source=spec.get('source'),
                              joint=spec.get('joint'),parent_contract=source['identity']['contract'])
        spec['milestones']=[1000,5000]+([10000,25000,50000] if arm=='S9' else [])
        path=root/'specs'/f'{arm}.json';save(path,spec);spec_paths.append(path)
        jobs.append(dict(arm=arm,spec=str(path),spec_sha256=sha(path),stop=cfg['max_steps'],previous_stop=0))
    contract=dict(generation=2,machine=a.machine,effective_batch=512,gradient_checkpointing=False)
    plan=dict(machine=a.machine,root=str(root/'execution'),ready=str(root/'ready.json'),contract=contract,
              wall_hours=24.,estimated_upper_hours=None,stages={'1':jobs},evaluations=json.loads((old/'plan.json').read_text())['evaluations'])
    plan['evaluations']=[dict(v,arms=list(arms)) for v in plan['evaluations'][:1]]
    save(root/'plan.json',plan)
    files={item['path']:item for item in json.loads((old/'ready.json').read_text())['files']}
    for folder in ('src','tools/style_matrix'):
        for path in Path(folder).rglob('*.py'):files[str(path)]=dict(path=str(path),sha256=sha(path))
    for path in spec_paths:files[str(path)]=dict(path=str(path),sha256=sha(path))
    save(root/'ready.json',dict(contract=contract,files=list(files.values())))
    print(root/'plan.json')


if __name__=='__main__':main()
