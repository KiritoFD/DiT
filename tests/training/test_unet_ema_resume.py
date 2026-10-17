"""Small CPU regressions for the actual U-Net EMA incident."""
import copy
from types import SimpleNamespace

import pytest
import torch

from src.train.ema import update_ema, effective_decay
from src.unet.io import ARCHITECTURE, Config, checkpoint_config, eval_args


@pytest.mark.parametrize('source_compiled,target_compiled',[(False,False),(True,False),(True,True)])
def test_full_ema_pairing(source_compiled,target_compiled):
    raw=torch.nn.Sequential(torch.nn.Linear(2,2),torch.nn.BatchNorm1d(2))
    target=copy.deepcopy(raw)
    source=torch.compile(raw,backend='eager') if source_compiled else raw
    ema=torch.compile(target,backend='eager') if target_compiled else target
    with torch.no_grad():
        raw[0].weight.add_(1)
        raw[1].running_mean.add_(2)
        raw[1].num_batches_tracked.add_(3)
    update_ema(ema,source,decay=0.)
    assert len(ema._ema_pairs[0])==len(list(raw.parameters()))
    for key,value in raw.state_dict().items():assert torch.equal(value,target.state_dict()[key])
    # A later update must read the current parameters, not a stale copied tensor.
    with torch.no_grad():raw[0].weight.add_(2)
    update_ema(ema,source,decay=.5)
    assert torch.allclose(target[0].weight,raw[0].weight-1)


def test_ema_mismatch_crashes():
    with pytest.raises(ValueError,match='keys differ'):
        update_ema(torch.nn.Linear(2,2),torch.nn.Sequential(torch.nn.Linear(2,2)),.9)


def test_reset_warmup_and_legacy_identity():
    args=SimpleNamespace(ema_decay=.999,ema_interval=2,ema_warmup=True,ema_start_step=20000)
    assert effective_decay(20000,args)==pytest.approx(.1**2)
    assert effective_decay(20001,args)==pytest.approx((2/11)**2)
    values=Config(max_steps=200000).__dict__
    old={'args':SimpleNamespace(architecture=ARCHITECTURE,architecture_config=values)}
    assert checkpoint_config(old).max_steps==200000
    assert eval_args(Config(max_steps=100000)).eval_vae_path=='pretrained_models/sd-vae-ft-ema'
    with pytest.raises(ValueError):checkpoint_config({'args':SimpleNamespace(architecture='DiT')})
