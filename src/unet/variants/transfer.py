"""Audited variant loading; old tensors are mandatory, only declared new tensors may initialize."""
from dataclasses import asdict

def load_expansion(model,checkpoint,source_config,target_config,*,state='raw'):
    """Return an exact inventory of newly initialized state keys, never silently skip old weights."""
    if state not in ('raw','ema'):raise ValueError(state)
    if state=='ema' and 'architecture' not in checkpoint:
        raise ValueError('Legacy EMA is invalid; use raw')
    for key in ('channels','num_res_blocks','num_writers','num_scripts'):
        a,b=getattr(source_config,key),getattr(target_config,key)
        if key=='channels':a,b=tuple(a),tuple(b)
        if a!=b:raise ValueError(f'Expansion may not change {key}; full-depth model must train from scratch')
    old_depth=source_config.decoder_res_blocks or (source_config.num_res_blocks,)*3
    new_depth=target_config.decoder_res_blocks or (target_config.num_res_blocks,)*3
    if any(new<old for old,new in zip(old_depth,new_depth)):
        raise ValueError('Cannot drop trained decoder blocks')
    prefixes=[f'decoder.{i}.blocks.{j}.' for i,(old,new) in enumerate(zip(old_depth,new_depth)) for j in range(old,new)]
    old_rank,new_rank=source_config.writer_film_rank,target_config.writer_film_rank
    if old_rank and old_rank!=new_rank:raise ValueError('Cannot remove or resize trained Writer FiLM')
    if not old_rank and new_rank:
        prefixes.extend(name+'.writer_film.' for name,block in model.named_modules()
                        if getattr(block,'writer_film',None) is not None)
    if source_config.residual_cond_gate and not target_config.residual_cond_gate:
        raise ValueError('Cannot remove trained conditional residual gates')
    if not source_config.residual_cond_gate and target_config.residual_cond_gate:
        prefixes.extend(name+'.residual_gate.' for name,block in model.named_modules()
                        if getattr(block,'residual_gate',None) is not None)
    for key,prefix in [('skip_gate_scales','skip_gates.'),('std_gate_scales','std_gates.')]:
        old,new=getattr(source_config,key),getattr(target_config,key)
        if old and not new:raise ValueError(f'Cannot remove {key}')
        if not old and new:prefixes.append(prefix)
    source=checkpoint['delta' if state=='raw' else 'ema']
    source={k.removeprefix('_orig_mod.'):v for k,v in source.items()}
    target=model.state_dict()
    expected_new={k for k in target if any(k.startswith(p) for p in prefixes)}
    missing=set(target)-set(source)
    unexpected=set(source)-set(target)
    if missing!=expected_new or unexpected:
        raise ValueError(f'Unapproved checkpoint keys: missing={sorted(missing-expected_new)}, unexpected={sorted(unexpected)}')
    for key,value in source.items():
        if value.shape!=target[key].shape:raise ValueError(f'Checkpoint shape mismatch: {key}')
    # This is still strict loading: defaults fill exactly the reviewed extension, nothing else.
    model.load_state_dict({**target,**source},strict=True)
    return dict(state=state,loaded_keys=len(source),new_keys=sorted(expected_new),
                source_config=asdict(source_config),target_config=asdict(target_config))


def set_scheduler_position(scheduler,optimizer,step):
    """Retain global exposure clock when A/B/C all deliberately restart their optimizer."""
    scheduler.last_epoch=step
    rates=[base*fn(step) for base,fn in zip(scheduler.base_lrs,scheduler.lr_lambdas)]
    for group,rate in zip(optimizer.param_groups,rates):group['lr']=rate
    scheduler._last_lr=rates
    return rates


def restore_expanded_optimizer(optimizer,model,source_model,saved):
    """Keep old Adam slots by parameter NAME; appended branches start with no slots.

    The U-Net trainer owns exactly one optimizer group. Positional loading after
    inserting modules is unsafe because parameter order changes inside blocks.
    """
    current=optimizer.state_dict()
    if len(saved['param_groups'])!=1 or len(current['param_groups'])!=1:
        raise ValueError('Expanded resume requires the U-Net single optimizer group')
    old_params=dict(source_model.named_parameters())
    new_params=dict(model.named_parameters())
    old_ids=saved['param_groups'][0]['params']
    new_ids=current['param_groups'][0]['params']
    if len(old_params)!=len(old_ids) or len(new_params)!=len(new_ids):
        raise ValueError('Optimizer parameter inventory mismatch')
    if not old_params.keys()<=new_params.keys():raise ValueError('Old optimizer parameters disappeared')
    destination=dict(zip(new_params,new_ids))
    states={}
    for (name,parameter),old_id in zip(old_params.items(),old_ids):
        if parameter.shape!=new_params[name].shape:raise ValueError('Optimizer shape changed: '+name)
        if old_id not in saved['state']:continue
        slots=saved['state'][old_id]
        for key in ('exp_avg','exp_avg_sq','max_exp_avg_sq'):
            if key in slots and slots[key].shape!=parameter.shape:
                raise ValueError('Optimizer slot shape mismatch: '+name)
        states[destination[name]]=slots
    group=dict(saved['param_groups'][0],params=new_ids)
    optimizer.load_state_dict(dict(state=states,param_groups=[group]))
    return dict(restored_slots=len(states),old_parameters=len(old_params),
                new_parameters=sorted(new_params.keys()-old_params.keys()))
