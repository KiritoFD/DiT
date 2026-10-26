"""Strict common-content transfer and R4 token initialization."""
import hashlib
from pathlib import Path
import torch


def initialize(model,checkpoint,state='ema',table=''):
    ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
    source={k.removeprefix('_orig_mod.'):v for k,v in ck[state if state=='ema' else 'delta'].items()}
    model.backbone.load_state_dict(source,strict=True)
    assert all(torch.equal(v,source[k].to(v)) for k,v in model.backbone.state_dict().items())
    if model.style_representation.kind==4:
        if not table:raise ValueError('R4 requires independently pretrained tokens')
        data=torch.load(table,map_location='cpu',weights_only=True)
        tokens=data['tokens']
        if tokens.shape!=(11,8,64):raise ValueError('R4 table must contain 10 writers + null')
        with torch.no_grad():model.style_representation.table.weight.copy_(tokens.reshape(11,512))
    return dict(source=str(checkpoint),source_step=int(ck['train_steps']),
                copied=len(source),strict_backbone_equal=True,r4_table=table,
                r4_sha256=hashlib.sha256(Path(table).read_bytes()).hexdigest() if table else None)
