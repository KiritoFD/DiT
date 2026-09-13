# -*- coding: utf-8 -*-
"""gpu_encode_v8.py — 建 v8 新资产集 (不覆盖旧文件):
  - copy 旧资产 -> *_v8 (GT 图 / img latents / skel latents / skel PNG)
  - 仅对 v7 变更 ids (21050) 用 GPU 重编码: img / skel3 / skel1 latent + 重算 skel PNG
  - 打补丁写入新 npz (fame_clean_v8.npz) 与新 shards
  - 生成 v8 csv: image_path 指向 final_imgs_fame_v8/
用法: python gpu_encode_v8.py --gpu 0
"""
import os, sys, csv, glob, json, time, shutil, argparse
import numpy as np
from PIL import Image

BASE = '/root/Workspace/xy/DiT'
os.chdir(BASE); sys.path.insert(0, BASE)
sys.stdout.reconfigure(encoding='utf-8')

from scipy.ndimage import binary_dilation, generate_binary_structure
try:
    from skimage.morphology import skeletonize
except ImportError:
    from scipy.ndimage import binary_erosion
    def skeletonize(b):
        skel = np.zeros_like(b); img = b.copy(); st = generate_binary_structure(2, 2)
        while img.any():
            er = binary_erosion(img, structure=st); skel |= img & ~er; img = er
        return skel

def dil3(b):
    return binary_dilation(b, structure=generate_binary_structure(2, 2), iterations=3)

V = '_v8'  # 新资产后缀

def load_changed():
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
    return sorted(changed)

def copy_tree(src, dst):
    """copy 旧资产到新目录 (保留文件名)"""
    if not os.path.isdir(src):
        print('SKIP copy (no src):', src)
        return
    os.makedirs(dst, exist_ok=True)
    n = 0
    for f in sorted(glob.glob(os.path.join(src, '*'))):
        base = os.path.basename(f)
        t = os.path.join(dst, base)
        if os.path.isdir(f):
            continue
        if not os.path.exists(t):
            shutil.copy2(f, t)
            n += 1
    print('copied %d files %s -> %s' % (n, src, dst))

def patch_shards(shard_dir, key, tag):
    """key: {iid: latent}。按 img_ids 打补丁写入 (原地覆盖 shard 文件)."""
    n = 0
    for shard in sorted(glob.glob(os.path.join(shard_dir, 'shard_*.npz'))):
        d = np.load(shard)
        m = np.isin(d['img_ids'], list(key.keys()))
        if not m.any():
            continue
        lat = d['latents'].copy()
        ids = d['img_ids']
        for j in np.where(m)[0]:
            lat[j] = key[int(ids[j])]
        np.savez_compressed(shard, latents=lat, img_ids=ids)
        n += int(m.sum())
    print('patched %s %d' % (tag, n))
    return n

def patch_npz(fname, key, tag):
    if not os.path.exists(fname):
        print('SKIP patch (no npz):', fname)
        return 0
    d = np.load(fname)
    lat = d['latents'].copy(); ids = d['img_ids']
    n = 0
    for iid, l in key.items():
        idx = np.where(ids == iid)[0]
        if len(idx):
            lat[idx[0]] = l
            n += 1
    np.savez_compressed(fname, latents=lat, img_ids=ids)
    print('patched %s (npz) %d' % (tag, n))
    return n


def make_v8_csv():
    # 生成 v8 csv: image_path 全部指向 final_imgs_fame_v8/
    for src_csv, dst_csv in [
        ('5script/train_fame_clean.csv', '5script/train_fame_clean_v8.csv'),
        ('5script/eval_fame_strict_clean.csv', '5script/eval_fame_strict_clean_v8.csv'),
    ]:
        rows = list(csv.DictReader(open(src_csv, encoding='utf-8')))
        with open(dst_csv, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                r = dict(r)
                iid = int(r['image_path'].split('/')[-1][:-4])
                r['image_path'] = 'final_imgs_fame_v8/%d.png' % iid
                w.writerow(r)
        print('made %s (%d rows)' % (dst_csv, len(rows)))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--copy-only', action='store_true', help='仅 copy 旧资产+新csv, 不重编码')
    args = ap.parse_args()

    # ---------- 1) copy 旧资产 -> v8 ----------
    copy_tree('final_imgs_fame_clean', 'final_imgs_fame_v8')
    copy_tree('final_latents_fame_clean', 'final_latents_fame_v8')
    copy_tree('final_skel1_fame', 'final_skel1_fame_v8')
    copy_tree('final_skel3_fame', 'final_skel3_fame_v8')
    copy_tree('final_skel_latents_fame', 'final_skel_latents_fame_v8')
    copy_tree('final_skel_latents_fame_1px', 'final_skel_latents_fame_1px_v8')
    # 新 npz: copy 旧 fame.npz
    if not os.path.exists('fame_clean_v8.npz') and os.path.exists('fame.npz'):
        shutil.copy2('fame.npz', 'fame_clean_v8.npz')
        print('copied fame.npz -> fame_clean_v8.npz')
    make_v8_csv()
    if args.copy_only:
        print('COPY-ONLY DONE')

    # ---------- 2) 变更 ids ----------
    changed = load_changed()
    print('changed ids:', len(changed))
    if changed[0] not in [int(x.split('/')[-1][:-4]) for x in os.listdir('final_imgs_fame_v8')]:
        pass  # 目录存在性检查
    missing = [i for i in changed if not os.path.exists('final_imgs_fame_v8/%d.png' % i)]
    print('changed missing in v8 GT:', len(missing))
    if missing:
        # 从最终清洗图补
        for i in missing:
            if os.path.exists('final_imgs_fame_clean/%d.png' % i):
                shutil.copy2('final_imgs_fame_clean/%d.png' % i, 'final_imgs_fame_v8/%d.png' % i)
        print('backfilled missing from final_imgs_fame_clean')

    # ---------- 3) GPU 重编码 ----------
    import torch
    import torchvision.transforms as T
    device = 'cuda:%d' % args.gpu
    print('cuda avail:', torch.cuda.is_available(), '| device:', device)
    tf = T.Compose([T.Resize((256, 256)), T.ToTensor(), T.Normalize([0.5]*3, [0.5]*3)])
    from diffusers.models import AutoencoderKL
    vae = AutoencoderKL.from_pretrained('pretrained_models/sd-vae-ft-ema').eval().to(device)

    def enc(t):
        with torch.no_grad():
            return (vae.encode(t.to(device)).latent_dist.mode() * 0.18215).to(torch.float16).cpu().numpy()

    batch = args.batch
    img_lat = {}; sk3_lat = {}; sk1_lat = {}
    t_start = time.time()
    for idx in range(0, len(changed), batch):
        ids_b = changed[idx:idx + batch]
        # GT 图 (v8 目录)
        xb = torch.stack([tf(Image.open('final_imgs_fame_v8/%d.png' % i).convert('RGB')) for i in ids_b])
        z_img = enc(xb)
        # skel PNG 重算 + 写 v8 目录
        sk3_arr = np.zeros((len(ids_b), 256, 256), np.uint8)
        sk1_arr = np.zeros((len(ids_b), 256, 256), np.uint8)
        for k, iid in enumerate(ids_b):
            p = 'final_imgs_fame_v8/%d.png' % iid
            a = np.asarray(Image.open(p).convert('L'))
            if a.shape != (256, 256):
                a = np.asarray(Image.open(p).convert('L').resize((256, 256)))
            sk = skeletonize(a < 127)
            s3 = np.where(dil3(sk), 0, 255).astype('uint8')
            s1 = np.where(sk, 0, 255).astype('uint8')
            Image.fromarray(s3, 'L').save('final_skel3_fame_v8/%d.png' % iid)
            Image.fromarray(s1, 'L').save('final_skel1_fame_v8/%d.png' % iid)
            sk3_arr[k] = s3; sk1_arr[k] = s1
        # skel3 latent: 灰度 [-1,1] 3ch
        x3 = torch.from_numpy(np.repeat((sk3_arr.astype(np.float32) / 255.0 * 2 - 1)[:, None, :, :], 3, axis=1))
        z3 = enc(x3)
        # skel1 latent: PNG RGB
        x1 = torch.stack([tf(Image.open('final_skel1_fame_v8/%d.png' % i).convert('RGB')) for i in ids_b])
        z1 = enc(x1)
        for k, iid in enumerate(ids_b):
            img_lat[iid] = z_img[k]
            sk3_lat[iid] = z3[k]
            sk1_lat[iid] = z1[k]
        if (idx // batch) % 10 == 0:
            el = time.time() - t_start
            print('encoded %d/%d in %.0fs (%.1f img/s)' % (idx + len(ids_b), len(changed), el, (idx + len(ids_b)) / el), flush=True)
    print('encode done: img=%d sk3=%d sk1=%d in %.0fs' % (len(img_lat), len(sk3_lat), len(sk1_lat), time.time() - t_start))

    # ---------- 4) 打补丁写入 v8 ----------
    n = patch_npz('fame_clean_v8.npz', img_lat, 'img')
    n += patch_shards('final_latents_fame_v8', img_lat, 'img shards')
    n += patch_shards('final_skel_latents_fame_v8', sk3_lat, 'skel3 shards')
    n += patch_shards('final_skel_latents_fame_1px_v8', sk1_lat, 'skel1 shards')
    print('total patched:', n)

    # ---------- 5) 校验抽样 ----------
    import random
    rng = random.Random(7)
    sample = rng.sample(changed, min(20, len(changed)))
    bad = 0
    for iid in sample:
        # img latent 一致性: 重编码 vs shard
        x = tf(Image.open('final_imgs_fame_v8/%d.png' % iid).convert('RGB'))[None]
        z = enc(x)[0]
        d = np.load('final_latents_fame_v8/shard_%04d.npz' % (iid // 3000))  # 近似 shard 分桶, 下面用真实查询
        # 真查询: 扫 shards 找 id
        for sh in sorted(glob.glob('final_latents_fame_v8/shard_*.npz')):
            dd = np.load(sh)
            if iid in dd['img_ids']:
                idx = list(dd['img_ids']).index(iid)
                stored = dd['latents'][idx]
                md = float(np.abs(stored.astype(np.float32) - z.astype(np.float32)).max())
                if md > 0.05:
                    bad += 1
                break
    print('latent consistency (img): %d/%d OK' % (len(sample) - bad, len(sample)))
    # skel1 一致性
    bad1 = 0
    for iid in sample:
        z1n = enc(tf(Image.open('final_skel1_fame_v8/%d.png' % iid).convert('RGB'))[None])[0]
        for sh in sorted(glob.glob('final_skel_latents_fame_1px_v8/shard_*.npz')):
            dd = np.load(sh)
            if iid in dd['img_ids']:
                idx = list(dd['img_ids']).index(iid)
                md = float(np.abs(dd['latents'][idx].astype(np.float32) - z1n.astype(np.float32)).max())
                if md > 0.05:
                    bad1 += 1
                break
    print('latent consistency (skel1): %d/%d OK' % (len(sample) - bad1, len(sample)))
    print('V8 ENCODE ALL DONE')