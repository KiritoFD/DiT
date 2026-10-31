"""2026-10-31 [A/B/C] train.py restart/resume 补丁。

两处语义修正（A/B/C 实验的前提）：
1. 峰值 LR 必须来自本 spec 的 config 分组（lr_backbone/lr_style_repr/lr_style_inj/lr_writer_table），
   而不是被 resume opt.load_state_dict 覆盖后的旧 phase 衰减 LR；
2. 跨 phase 边界 optimizer 拓扑变化（如 P0 无 writer_table 组 → B/C1 4 组）时，
   用新 phase 的全新 moments，而不是抛错；同 arm 续跑的组数不匹配仍然报错。
"""
from pathlib import Path

P=Path('/root/Workspace/xy/DiT/tools/style_matrix/runtime/train.py')
src=P.read_text(encoding='utf-8')

old1="""    opt=optimizer(raw,cfg,auxiliary.vae.decoder if auxiliary else None,scratch=spec.get('initialization')=='scratch')
    restart=spec.get('schedule_restart')"""
new1="""    opt=optimizer(raw,cfg,auxiliary.vae.decoder if auxiliary else None,scratch=spec.get('initialization')=='scratch')
    config_lrs=[g['lr'] for g in opt.param_groups]
    restart=spec.get('schedule_restart')"""
assert src.count(old1)==1
src=src.replace(old1,new1)

old2="""    if resume:
        ema.load_state_dict(resume['ema'],strict=True);opt.load_state_dict(resume['opt'])
    if restart:"""
new2="""    if resume:
        ema.load_state_dict(resume['ema'],strict=True)
        if len(resume['opt']['param_groups'])!=len(opt.param_groups):
            print(f'[resume] optimizer topology changed '
                  f'({len(resume["opt"]["param_groups"])} -> {len(opt.param_groups)} groups); '
                  'fresh moments for this phase',flush=True)
        else:
            opt.load_state_dict(resume['opt'])
    if restart:"""
assert src.count(old2)==1
src=src.replace(old2,new2)

old3="""        scheduler.last_epoch=start;assert scheduler.last_epoch==start
        if start>origin and resume['matrix_identity']==spec['identity']:"""
new3="""        scheduler.last_epoch=start;assert scheduler.last_epoch==start
        # 2026-10-31 [A/B/C]: 峰值 LR 取自本 spec 的 config 分组;
        # resume 已把 param_groups 覆盖成旧 phase 的衰减 LR / 旧拓扑, 不能作 base。
        scheduler.base_lrs=config_lrs
        for group,lr in zip(opt.param_groups,config_lrs):group['initial_lr']=lr
        if start>origin and resume['matrix_identity']==spec['identity']:"""
assert src.count(old3)==1
src=src.replace(old3,new3)

old4="""        if previous_lrs is None:
            raise KeyError('Checkpoint scheduler state has neither _last_lr nor last_lr')
        if len(previous_lrs)!=len(opt.param_groups):
            raise ValueError(f'Checkpoint LR groups ({len(previous_lrs)}) do not match optimizer groups ({len(opt.param_groups)})')
        for group,lr in zip(opt.param_groups,previous_lrs):group['lr']=lr"""
new4="""        if previous_lrs is None:
            raise KeyError('Checkpoint scheduler state has neither _last_lr nor last_lr')
        if len(previous_lrs)!=len(opt.param_groups):
            if resume['matrix_identity']==spec['identity']:
                raise ValueError(f'Checkpoint LR groups ({len(previous_lrs)}) do not match optimizer groups ({len(opt.param_groups)})')
            print(f'[restart] LR-group topology changed across phase boundary '
                  f'({len(previous_lrs)} -> {len(opt.param_groups)}); skip first-step LR continuity',flush=True)
        else:
            for group,lr in zip(opt.param_groups,previous_lrs):group['lr']=lr"""
assert src.count(old4)==1
src=src.replace(old4,new4)

P.write_text(src,encoding='utf-8')
import py_compile
py_compile.compile(str(P),doraise=True)
print('PATCHED OK')
