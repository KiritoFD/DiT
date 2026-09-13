#!/bin/bash
# Recreate pending markers properly with correct JSON
cd /root/Workspace/xy/DiT/5script/results/s10_b4_grey_clear
rm -f */checkpoints/eval_auto_*.json
rm -f */checkpoints/eval_pending_*.json
rm -f */checkpoints/eval_samples/eval_pending_*.json

python3 -c "
import os, json, glob
base = '.'
for d in sorted(glob.glob(os.path.join(base, '*/checkpoints/eval_samples/step*'))):
    if not os.path.isdir(d):
        continue
    step = int(os.path.basename(d).replace('step',''))
    ckpt_dir = os.path.dirname(os.path.dirname(d))
    n = len(glob.glob(os.path.join(d, 'sample*.png')))
    rel = 'eval_samples/' + os.path.basename(d)
    pend = os.path.join(ckpt_dir, f'eval_pending_{step:07d}.json')
    info = {'step': step, 'n': n, 'dir': rel, 'elapsed_gpu': 0}
    with open(pend, 'w') as f:
        json.dump(info, f)
    print(f'wrote {pend} (n={n})')
"
