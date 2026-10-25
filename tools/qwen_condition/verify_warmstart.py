"""CPU verification of the arm-C warm start: exact strict load + zero-init stem channel."""
import json
import sys

sys.path.insert(0, '/home/ds/Workspace/DiT')
import torch  # noqa: E402

from src.unet.io import Config, build, checkpoint_config  # noqa: E402
from src.unet.variants.transfer import load_expansion  # noqa: E402


def to_cfg(d):
    d = dict(d)
    for k in ('channels', 'skip_gate_scales', 'std_gate_scales', 'checkpoint_steps'):
        if k in d and isinstance(d[k], list):
            d[k] = tuple(d[k])
    d['decoder_res_blocks'] = tuple(d['decoder_res_blocks']) if d.get('decoder_res_blocks') else None
    return Config(**d)


new = to_cfg(json.load(open('/home/ds/Workspace/DiT/tools/qwen_condition/arm_c.json')))
ck = torch.load('/home/ds/Workspace/DiT/assets/results/unet_deep_sym4_114m_60m/'
                '20261019-011647-Unet-flow/checkpoints/0085000.pt',
                map_location='cpu', weights_only=False)
old = checkpoint_config(ck)
model = build(new)
report = load_expansion(model, ck, old, new)
orig_stem = ck['delta']['_orig_mod.std_stem.0.weight']
print('loaded', report['loaded_keys'], 'new', report['new_keys'])
print('stem ch0==orig:', torch.equal(model.std_stem[0].weight[:, 0:1], orig_stem))
print('stem ch1 zero:', bool((model.std_stem[0].weight[:, 1] == 0).all()))
