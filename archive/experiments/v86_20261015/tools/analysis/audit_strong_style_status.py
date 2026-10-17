"""Read-only mapping and training-probe audit; no checkpoint mutation."""
import csv
import json
import re
from pathlib import Path
import numpy as np
import sys
import torch

root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root))
cfg = json.loads((root/'src/train/configs/v86_strong_A60.json').read_text())
rows = list(csv.DictReader(open(root/cfg['data_csv'], encoding='utf-8-sig')))
cr = json.loads((root/cfg['callig_remap_json']).read_text())
sr = json.loads((root/cfg['font_remap_json']).read_text())
writers, scripts = [], []
for r in rows:
    if r.get('aug', '').strip(): continue
    if r['calligrapher'] not in writers: writers.append(r['calligrapher'])
    if r['script'] not in scripts: scripts.append(r['script'])
mismatch = sorted({(r['calligrapher'], int(cr[str(r['calligrapher_id'])]), writers.index(r['calligrapher']))
                   for r in rows if r['calligrapher'] in writers
                   and int(cr[str(r['calligrapher_id'])]) != writers.index(r['calligrapher'])})
sm = sorted({(r['script'], int(sr[str(r['script_id'])]), scripts.index(r['script']))
             for r in rows if int(sr[str(r['script_id'])]) != scripts.index(r['script'])})
result = dict(writer_mapping_mismatch=mismatch, script_mapping_mismatch=sm, runs={})
from src.loss.critic_loss import CriticLoss
critic = CriticLoss.__new__(CriticLoss)
torch.nn.Module.__init__(critic)
critic.slots = torch.load(root/cfg['critic_ckpt'], map_location='cpu', weights_only=False)['slots']
lut = critic.build_id2slot(root/cfg['data_csv'], str(root/cfg['callig_remap_json']), str(root/cfg['font_remap_json']))
bad = {(r['calligrapher'], r['script']) for r in rows
       if lut.get((int(cr[str(r['calligrapher_id'])]), int(sr[str(r['script_id'])])))
       != r['calligrapher'] + '·' + r['script']}
assert not bad, bad
result['corrected_mapping_mismatches'] = sorted(bad)
result['corrected_slots'] = len(lut)
for arm in ('A30', 'A60', 'A100'):
    files = sorted((root/'logs').glob('strong_style_'+arm+'_*.log'))
    if not files: continue
    text = files[-1].read_text(errors='replace')
    probes = re.findall(r'step=(\d+) active=(\d+) margin=([-\d.]+) writer=([\deE+.-]+) weight=([\deE+.-]+) ratio=([\deE+.-]+)', text)
    p = np.array(probes, dtype=float)
    result['runs'][arm] = dict(n_probes=len(p), first100_mean=p[:100].mean(0).tolist(),
        last100_mean=p[-100:].mean(0).tolist(), weight_quantiles=np.quantile(p[:,4], [.1,.5,.9]).tolist(),
        nan_skips=text.count('skipping'), underpowered=text.count('strong-style-underpowered'))
print(json.dumps(result, ensure_ascii=False, indent=2))
