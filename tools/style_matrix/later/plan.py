"""Additional post-24h recipes; existing identities and input files are immutable."""
from pathlib import Path
from tools.style_matrix.generation2.plan import read,continuation,seal
from tools.style_matrix.runtime.common import save


def matched(base,arm):
    middle=continuation(base,arm,50000,60000)
    mature=continuation(middle,arm,60000,75000)
    middle['milestones']=[55000,60000];mature['milestones']=[65000,70000,75000]
    return middle,mature


def prepare(root,machine):
    root=Path(root);arms=('S10','S0','S4') if machine=='4090' else ('S11',)
    for arm in arms:
        base=read(root/'specs'/f'{arm}.json');middle,mature=matched(base,arm)
        identity_path=root/'runs'/arm/'identity.json'
        completed=sorted((root/'runs'/arm/'checkpoints').glob('*.pt.done'))
        if completed and int(completed[-1].name.split('.')[0])>50000:
            identity=read(identity_path)
            if identity not in (middle['identity'],mature['identity']):
                raise ValueError(f'{arm} already entered a different post-50k LR trajectory; preserve the active experiment')
        for suffix,spec in (('60_matched',middle),('75_matched',mature)):
            path=root/'specs'/f'{arm}_{suffix}.json'
            if path.exists() and read(path)!=spec:raise ValueError('Immutable future recipe changed: '+str(path))
            if not path.exists():save(path,spec)
    seal(root,machine)
    save(root/'later_plan.json',dict(version=1,phase1_unchanged=True,post50_schedule=[
        dict(origin=50000,horizon=10000,warmup=200,peak_scale=.5),
        dict(origin=60000,horizon=15000,warmup=200,peak_scale=.5)],
        machine=machine,arms=list(arms),deadline_preserved=True))
