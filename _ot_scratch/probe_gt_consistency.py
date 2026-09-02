import os, json, numpy as np
from PIL import Image

os.chdir('/root/Workspace/xy/DiT')

# changed ids
changed = set()
for l in open('/tmp/fame_v7_report.jsonl', encoding='utf-8'):
    r = json.loads(l)
    if 'error' in r:
        continue
    iid = int(os.path.basename(r['path']).split('.')[0])
    if r.get('bars', 0) > 0 or r.get('removed_frac', 0) > 0 or r.get('inverted'):
        changed.add(iid)
try:
    for i in json.load(open('/tmp/encode_changed_ids.json', encoding='utf-8')):
        changed.add(int(i))
except Exception:
    pass
changed = sorted(changed)

# 对比 final_imgs_fame_clean vs final_imgs_256 (changed 抽样 + 非 changed 抽样)
import random
rng = random.Random(3)
ch_s = rng.sample(changed, 8)
un_s = rng.sample([i for i in range(1, 60000) if i not in changed], 8)
for iid in ch_s + un_s:
    f1 = 'final_imgs_fame_clean/%d.png' % iid
    f2 = 'final_imgs_256/%d.png' % iid
    e1, e2 = os.path.exists(f1), os.path.exists(f2)
    same = None
    if e1 and e2:
        im1 = np.asarray(Image.open(f1).convert('L'))
        im2 = np.asarray(Image.open(f2).convert('L'))
        same = np.array_equal(im1, im2)
    print('id %6d %s changed=%s | clean_exists=%s p256_exists=%s same=%s' % (
        iid, 'CH' if iid in changed else 'un', iid in changed, e1, e2, same))

# 非 changed 但 csv 指向 final_imgs_256: 它们是否与 fame_clean 一致?
print()
print('total changed:', len(changed))