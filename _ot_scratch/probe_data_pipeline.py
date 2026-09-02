import os, csv, json, numpy as np

os.chdir('/root/Workspace/xy/DiT')

# 1) train_fame_clean.csv 的 image_path 指向哪里?
rows = list(csv.DictReader(open('5script/train_fame_clean.csv', encoding='utf-8')))
print('train_fame_clean rows:', len(rows))
print('cols:', list(rows[0].keys()))
p = rows[0]['image_path']
print('sample image_path:', p, '| exists:', os.path.exists(p))
paths = set(r['image_path'] for r in rows[:5000])
prefixes = {}
for p in paths:
    d = p.split('/')[0]
    prefixes[d] = prefixes.get(d, 0) + 1
print('path dirs (sample 5000):', prefixes)

# 2) final_latents_fame_clean vs final_latents_fame 内容是否一致?
a = np.load('final_latents_fame_clean/shard_0000.npz')
b = np.load('final_latents_fame/shard_0000.npz')
print('clean==plain shard0 ids equal:', np.array_equal(a['img_ids'], b['img_ids']))
print('clean==plain shard0 latents equal:', np.array_equal(a['latents'], b['latents']))
if not np.array_equal(a['latents'], b['latents']):
    diff = np.abs(a['latents'].astype(np.float32) - b['latents'].astype(np.float32))
    print('max abs diff shard0:', float(diff.max()), '| nonzero:', int((diff > 0).sum()))

# 3) changed ids 与 clean shards 的关系
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
print('changed ids:', len(changed))

# 4) final_imgs_fame_clean vs final_imgs_256: 训练 csv 用哪个? img 文件是否已更新?
# 检查 changed 中一个 id 的两个目录文件是否一致
sample = sorted(changed)[:3]
from PIL import Image
for iid in sample:
    f_clean = 'final_imgs_fame_clean/%d.png' % iid
    f_plain = 'final_imgs_256/%d.png' % iid
    ex1, ex2 = os.path.exists(f_clean), os.path.exists(f_plain)
    same = None
    if ex1 and ex2:
        im1 = np.asarray(Image.open(f_clean).convert('L'))
        im2 = np.asarray(Image.open(f_plain).convert('L'))
        same = np.array_equal(im1, im2)
    print('id %d | clean exists=%s plain exists=%s same=%s' % (iid, ex1, ex2, same))