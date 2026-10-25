"""Minimal repro of the offline-eval 16-vs-0 crash (arm A ckpt, 24-row eval set)."""
import json
import sys

sys.path.insert(0, '/home/ds/Workspace/DiT')
import torch  # noqa: E402

from src.unet.io import Config, build, checkpoint_config  # noqa: E402

CKPT = '/home/ds/Workspace/DiT/assets/results/qwen_cond_arm_a/20261025-033733-Unet-flow/checkpoints/0090000.pt'
ck = torch.load(CKPT, map_location='cpu', weights_only=False)


def to_cfg(d):
    d = dict(d)
    for k in ('channels', 'skip_gate_scales', 'std_gate_scales', 'checkpoint_steps'):
        if k in d and isinstance(d[k], list):
            d[k] = tuple(d[k])
    d['decoder_res_blocks'] = tuple(d['decoder_res_blocks']) if d.get('decoder_res_blocks') else None
    return Config(**d)


cfg = to_cfg(json.loads(json.dumps(checkpoint_config(ck), default=lambda o: dict(o.__dict__) if hasattr(o, '__dict__') else list(o))))
model = build(cfg).cuda()
w = {k.removeprefix('_orig_mod.'): v for k, v in ck['ema'].items()}
model.load_state_dict(w, strict=True)
model.eval().cuda()

from src.utils.std_soft_glyph import StdSoftGlyphStore  # noqa: E402
store = StdSoftGlyphStore('/home/ds/Workspace/DiT/exp-std/data/std_soft_glyphs_v1',
                          is_train=False, eval_sigma=.65, preload=True)
from src.eval.sampler import sample_latents  # noqa: E402
from src.loss.flow_matching import FlowMatching  # noqa: E402
import csv  # noqa: E402

rows = list(csv.DictReader(open('/home/ds/Workspace/DiT/assets/results/qwen_cond_eval/'
                                'qwen_cond_arm_a_20261025-033733_0090000/sets/eval200fix.csv',
                                encoding='utf-8-sig')))[:24]
SID = {'楷': 0, '行': 3, '隶': 4}
n = len(rows)
noise = torch.randn(n, 4, 32, 32, generator=torch.Generator().manual_seed(0))
conds = [(int(r['calligrapher_id']), int(r['character_id'])) for r in rows]
raw = [int(r['calligrapher_id']) for r in rows]
pair = [int(r['calligrapher_id']) for r in rows]
script = [int(r['script_id']) for r in rows]
skel = torch.stack([store.get(SID[r['script']], r['character']) for r in rows])
diff = FlowMatching(num_steps=50, sampler='heun')
print('setup ok n=', n, flush=True)
lat = sample_latents(model, diff, noise, conds, cfg_scale=1., batch=16,
                     device=torch.device('cuda'), skel=skel, seed=0,
                     hier_conds=(raw, pair, script))
print('SAMPLE_OK', lat.shape)
