"""5k P2 controls: only explicit declared factors change; supervision arms require real gates."""
from dataclasses import asdict
import json
from pathlib import Path
import sys
import torch
from tools.three_day.contracts import save,sha


def run(queue):
    from src.unet.io import checkpoint_config,Config
    from tools.three_day.contracts import TOP10
    root=queue.root;source=Path('assets/u100_p1p2_20261025/p2_runs/20261025-152857-Unet-flow/checkpoints/0020000.pt')
    ck=torch.load(source,map_location='cpu',weights_only=False,mmap=True);c=checkpoint_config(ck);base=asdict(c)
    base.update(initialize_from=str(source),initialize_state='ema',max_steps=5000,stop_at_step=5000,lr_decay_steps=5000,
        batch=512,seed=90,lr=5e-6,writer_cond_lr=2e-5,warmup=100,ema_start_step=0,
        std_soft_eval_sigma=0.,std_shape_step_offset=20000,ckpt_every=2500,eval_every=2500,eval_before_training=False,
        semantic_writer_rows={str(k):v for k,v in TOP10.items()})
    day=root/'day2';day.mkdir(exist_ok=True)
    from .difficulty import build
    weights=build(base,day/'difficulty.npz')
    specs={
        'B0':dict(objective='fm',external_dropout=True),
        'B1':dict(objective='fm',external_dropout=True,sampling='writer_script'),
        'B2':dict(objective='fm',external_dropout=True,sampling='complexity',weights=weights),
        'B3':dict(objective='writer_time',external_dropout=True)}
    save(day/'manifest.json',dict(source=str(source),source_sha256=sha(source),specs=specs,
        external_dropout='Matched B0/B1/B2/B3 explicitly sample native masks before model call; B3 weights only actual non-null writers.',
        B0_reuse=False,reason='New matched control includes dropout visibility adapter; do not mix with old strong run.'))
    for arm,spec in specs.items():
        path=day/arm/'study.json';save(path,spec)
        config=dict(base,output=str(day/arm/'runs'),study_spec=str(path))
        Config(**config).validate();p=day/arm/'config.json';save(p,config)
        queue.command(arm,[sys.executable,'-u','-m','src.unet.train','--config',str(p)])
        finals=list((day/arm/'runs').glob('*/checkpoints/0005000.pt.done'))
        if len(finals)!=1:raise FileNotFoundError('Missing finished Day2 checkpoint '+arm)
    gates=root/'remote48'/'style_gates.json'
    # Deliberate dependency state, not a fictitious completed auxiliary experiment.
    save(day/'supervision_pending.json',dict(state='awaiting_audited_teacher_and_content_gradient_gates',
        gates_path=str(gates),arms=['B4','B5','B6'],no_implicit_approval=True))
    queue.status.update(state='day2_sampling_complete',current='B4_B5_B6_gates');queue.flush()
