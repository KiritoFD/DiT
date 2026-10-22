"""Explicit frozen independently-pretrained writer-table contract."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


@torch.no_grad()
def install_writer_table(model,ema,optimizer,config):
    path=Path(config.writer_table_file)
    if not config.freeze_writer_table:
        raise ValueError('Independent writer-table replacement must stay frozen')
    mapping=json.loads(path.with_name('callig_remap.json').read_text())
    expected=json.loads(Path(config.callig_remap_json).read_text())
    expected={k:v for k,v in expected.items() if int(v)!=model.writer.weight.shape[0]-1}
    if mapping!=expected:raise ValueError('Writer-table row mapping differs from model mapping')
    values=torch.from_numpy(np.load(path)).to(model.writer.weight)
    if values.shape!=model.writer.weight[:-1].shape or not torch.isfinite(values).all():
        raise ValueError('Writer-table shape/values invalid')
    model.writer.weight[:-1].copy_(values)
    ema.writer.weight.copy_(model.writer.weight)  # preserve source CFG-null row, not randomize it
    model.writer.weight.requires_grad_(False)
    if optimizer is not None:optimizer.state.pop(model.writer.weight,None)
    return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        shape=list(values.shape),frozen=True,optimizer_writer_state_cleared=True,
        null_row='retained from source raw checkpoint',mapping_verified=True)
