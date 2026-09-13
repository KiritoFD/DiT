import os, glob, random, numpy as np, torch
from PIL import Image

os.chdir('/root/Workspace/xy/DiT')
import torchvision.transforms as T
tf = T.Compose([T.Resize((256, 256)), T.ToTensor(), T.Normalize([0.5]*3, [0.5]*3)])
from diffusers.models import AutoencoderKL
vae = AutoencoderKL.from_pretrained('pretrained_models/sd-vae-ft-ema').eval().cuda()

def enc(img):
    with torch.no_grad():
        return (vae.encode(img.unsqueeze(0).cuda()).latent_dist.mode() * 0.18215).to(torch.float16)[0].cpu().numpy()

def find_in(dirname, iid):
    """真实查询: 扫 shards 找 id 的 latent"""
    for sh in sorted(glob.glob(dirname + '/shard_*.npz')):
        d = np.load(sh)
        ids = d['img_ids']
        if iid in ids:
            j = list(ids).index(iid)
            return d['latents'][j]
    return None

changed = set()
import json
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

rng = random.Random(7)
sample = rng.sample(changed, min(25, len(changed)))
print('validating %d changed ids ...' % len(sample))
bad_img = bad_s1 = bad_s3 = 0
for iid in sample:
    # img latent
    img = tf(Image.open('final_imgs_fame_v8/%d.png' % iid).convert('RGB'))
    z_img = enc(img)
    st_img = find_in('final_latents_fame_v8', iid)
    if st_img is None or float(np.abs(st_img.astype(np.float32) - z_img.astype(np.float32)).max()) > 0.05:
        bad_img += 1
    # skel1
    img1 = tf(Image.open('final_skel1_fame_v8/%d.png' % iid).convert('RGB'))
    z1 = enc(img1)
    st1 = find_in('final_skel_latents_fame_1px_v8', iid)
    if st1 is None or float(np.abs(st1.astype(np.float32) - z1.astype(np.float32)).max()) > 0.05:
        bad_s1 += 1
    # skel3
    sk3 = np.asarray(Image.open('final_skel3_fame_v8/%d.png' % iid).convert('L')).astype(np.float32) / 255.0 * 2 - 1
    x3 = torch.from_numpy(np.repeat(sk3[None, :, :], 3, axis=0))  # (3,256,256)
    z3 = enc(x3)
    st3 = find_in('final_skel_latents_fame_v8', iid)
    if st3 is None or float(np.abs(st3.astype(np.float32) - z3.astype(np.float32)).max()) > 0.05:
        bad_s3 += 1
print('img  consistency: %d/%d OK' % (len(sample) - bad_img, len(sample)))
print('skel1 consistency: %d/%d OK' % (len(sample) - bad_s1, len(sample)))
print('skel3 consistency: %d/%d OK' % (len(sample) - bad_s3, len(sample)))
print('V8 VALIDATION DONE')