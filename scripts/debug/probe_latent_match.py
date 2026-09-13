import os, glob, csv, json, numpy as np, torch
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

# 找 8 个 changed id 在两个 latent 目录中都在的
def load_latent(ids, name):
    out = {}
    for f in sorted(glob.glob('%s/shard_*.npz' % name)):
        d = np.load(f)
        for j, iid in enumerate(d['img_ids'].tolist()):
            if iid in ids:
                out[iid] = d['latents'][j]
    return out

sample_ids = sorted(changed)[:8]
print('sample ids:', sample_ids)

lat_clean = load_latent(sample_ids, 'final_latents_fame_clean')
lat_plain = load_latent(sample_ids, 'final_latents_fame')
print('found in clean:', len(lat_clean), 'in plain:', len(lat_plain))

# 用干净图 encode 作对比
import torchvision.transforms as T
tf = T.Compose([T.Resize((256, 256)), T.ToTensor(), T.Normalize([0.5]*3, [0.5]*3)])
from diffusers.models import AutoencoderKL
vae = AutoencoderKL.from_pretrained('pretrained_models/sd-vae-ft-ema').eval()

def enc_ref(iid):
    for root in ['final_imgs_fame_clean', 'final_imgs_256', 'final_imgs_256_clean']:
        p = '%s/%d.png' % (root, iid)
        if os.path.exists(p):
            x = tf(Image.open(p).convert('RGB'))[None]
            with torch.no_grad():
                z = (vae.encode(x).latent_dist.mode() * 0.18215).to(torch.float16)[0]
            return root, z.numpy()
    return None, None

for iid in sample_ids:
    root, zr = enc_ref(iid)
    if zr is None:
        print('id %d: no ref image' % iid)
        continue
    d_c = float(np.abs(lat_clean.get(iid, np.zeros((4, 32, 32), np.float16)).astype(np.float32) - zr.astype(np.float32)).max()) if iid in lat_clean else None
    d_p = float(np.abs(lat_plain.get(iid, np.zeros((4, 32, 32), np.float16)).astype(np.float32) - zr.astype(np.float32)).max()) if iid in lat_plain else None
    print('id %6d refimg=%s | maxdiff vs clean=%.4f vs plain=%.4f' % (iid, root, d_c, d_p))