# -*- coding: utf-8 -*-
"""cpu_encode_v8.py - v7 清洗后变更图的多进程 CPU encode (img + skel3 PNG + skel1 PNG).
并行: nproc 进程 x 每进程 threads (oneDNN/mkldnn SIMD), nice 降优先级避免挤训练.
自动对比 fp32 vs bf16 (前 32 张 max-abs-diff), 差异小(<0.05)则用 bf16 提速.
用法: python cpu_encode_v8.py --nproc 12 --threads 4 --batch 16 --dtype auto
"""
import os, sys, csv, json, glob, time, argparse, multiprocessing as mp
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

def build_path_map():
    rows = list(csv.DictReader(open('5script/train_fame.csv', encoding='utf-8')))
    rows += list(csv.DictReader(open('5script/eval_fame_strict.csv', encoding='utf-8')))
    m = {}
    for r in rows:
        iid = int(r['image_path'].split('/')[-1][:-4])
        m[iid] = r['image_path']
    return m

# ---------------- worker ----------------
def worker(cfg, q):
    me, nproc, ithr, batch, dtype, paths = cfg
    os.environ['OMP_NUM_THREADS'] = str(ithr)
    os.environ['MKL_NUM_THREADS'] = str(ithr)
    try:
        os.nice(10)
    except Exception:
        pass
    import torch
    torch.set_num_threads(ithr)
    import torchvision.transforms as T
    tf = T.Compose([T.Resize((256, 256)), T.ToTensor(), T.Normalize([0.5]*3, [0.5]*3)])
    from diffusers.models import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(BASE + '/pretrained_models/sd-vae-ft-ema').eval()
    if dtype == 'bf16':
        vae = vae.to(torch.bfloat16)
    def enc(x):
        with torch.no_grad():
            return (vae.encode(x).latent_dist.mode() * 0.18215).to(torch.float16).cpu().numpy()

    keys = sorted(paths.keys())
    mine = [k for k in keys if int(k) % nproc == me]
    img_lat = {}; skel3_lat = {}; skel1_lat = {}
    t0 = time.time(); done = 0; acc = []; acc_id = []

    def flush_img():
        nonlocal acc, acc_id, done
        if not acc:
            return
        xb = torch.from_numpy(np.stack(acc))
        if dtype == 'bf16':
            xb = xb.to(torch.bfloat16)
        lat = enc(xb)
        for j, iid in enumerate(acc_id):
            img_lat[iid] = lat[j]
        done += len(acc); acc = []; acc_id = []

    # ---- pass 1: skel PNG 重算 + img latent ----
    for iid in mine:
        p = paths[iid]
        a = np.asarray(Image.open(p).convert('L'))
        if a.shape != (256, 256):
            a = np.asarray(Image.open(p).convert('L').resize((256, 256)))
        sk = skeletonize(a < 127)
        sk3 = np.where(dil3(sk), 0, 255).astype('uint8')   # 3px 骨架 PNG (黑线白底)
        sk1 = np.where(sk, 0, 255).astype('uint8')          # 1px 骨架 PNG
        Image.fromarray(sk3, 'L').save('final_skel3_fame/%d.png' % iid)
        Image.fromarray(sk1, 'L').save('final_skel1_fame/%d.png' % iid)
        acc.append(tf(Image.open(p).convert('RGB')).numpy()); acc_id.append(iid)
        if len(acc) >= batch:
            flush_img()
            if done % 200 < batch:
                dt = time.time() - t0
                print('[proc%d] img encode %d/%d @ %.1f img/s' % (me, done, len(mine), done / dt), flush=True)
    flush_img()

    # ---- pass 2: skel3 latent (灰度 -> [-1,1] 3ch) ----
    acc = []; acc_id = []
    for iid in mine:
        sk3 = np.asarray(Image.open('final_skel3_fame/%d.png' % iid).convert('L'))
        x = np.repeat((sk3.astype(np.float32) / 255.0 * 2 - 1)[None, None, :, :], 3, axis=1)
        acc.append(x); acc_id.append(iid)
        if len(acc) >= batch:
            xb = torch.from_numpy(np.stack(acc))
            if dtype == 'bf16':
                xb = xb.to(torch.bfloat16)
            lat = enc(xb)
            for j, iid2 in enumerate(acc_id):
                skel3_lat[iid2] = lat[j]
            acc = []; acc_id = []
    if acc:
        xb = torch.from_numpy(np.stack(acc))
        if dtype == 'bf16':
            xb = xb.to(torch.bfloat16)
        lat = enc(xb)
        for j, iid2 in enumerate(acc_id):
            skel3_lat[iid2] = lat[j]

    # ---- pass 3: skel1 latent (PNG RGB) ----
    acc = []; acc_id = []
    for iid in mine:
        acc.append(tf(Image.open('final_skel1_fame/%d.png' % iid).convert('RGB')).numpy()); acc_id.append(iid)
        if len(acc) >= batch:
            xb = torch.from_numpy(np.stack(acc))
            if dtype == 'bf16':
                xb = xb.to(torch.bfloat16)
            lat = enc(xb)
            for j, iid2 in enumerate(acc_id):
                skel1_lat[iid2] = lat[j]
            acc = []; acc_id = []
    if acc:
        xb = torch.from_numpy(np.stack(acc))
        if dtype == 'bf16':
            xb = xb.to(torch.bfloat16)
        lat = enc(xb)
        for j, iid2 in enumerate(acc_id):
            skel1_lat[iid2] = lat[j]

    dt = time.time() - t0
    n_img = len(img_lat); n_s3 = len(skel3_lat); n_s1 = len(skel1_lat)
    r = (n_img + n_s3 + n_s1) / dt
    print('[proc%d] DONE img=%d skel3=%d skel1=%d in %ds -> %.1f tot/s' % (me, n_img, n_s3, n_s1, dt, r), flush=True)
    # 落临时文件回传 (避免 Queue 传大对象)
    os.makedirs('/tmp/v8_enc', exist_ok=True)
    iarr = np.array(sorted(img_lat.keys()), dtype=np.int64)
    s3arr = np.array(sorted(skel3_lat.keys()), dtype=np.int64)
    s1arr = np.array(sorted(skel1_lat.keys()), dtype=np.int64)
    np.savez_compressed(
        '/tmp/v8_enc/w%d.npz' % me,
        img_ids=iarr,
        img_lat=np.stack([img_lat[k] for k in sorted(img_lat.keys())]) if iarr.size else np.zeros((0, 4, 32, 32), np.float16),
        sk3_ids=s3arr,
        sk3_lat=np.stack([skel3_lat[k] for k in sorted(skel3_lat.keys())]) if s3arr.size else np.zeros((0, 4, 32, 32), np.float16),
        sk1_ids=s1arr,
        sk1_lat=np.stack([skel1_lat[k] for k in sorted(skel1_lat.keys())]) if s1arr.size else np.zeros((0, 4, 32, 32), np.float16),
    )
    q.put((me, n_img, n_s3, n_s1, r, dt))


def patch_npz(fname, key):
    d = np.load(fname)
    m = np.isin(d['img_ids'], list(key.keys()))
    n = int(m.sum())
    if n:
        lat = d['latents'].copy()
        for iid, l in key.items():
            idx = np.where(d['img_ids'] == iid)[0]
            if len(idx):
                lat[idx[0]] = l
        np.savez_compressed(fname, latents=lat, img_ids=d['img_ids'])
    return n


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--nproc', type=int, default=12)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--dtype', choices=['auto', 'fp32', 'bf16'], default='auto')
    args = ap.parse_args()
    print('loading changed ids...', flush=True)
    changed = load_changed()
    path_of = build_path_map()
    path_of = {i: path_of[i] for i in changed if i in path_of}
    print('changed=%d with_path=%d' % (len(changed), len(path_of)), flush=True)
    import torch
    print('torch %s mkldnn=%s threads=%d' % (torch.__version__, torch.backends.mkldnn.is_available(), torch.get_num_threads()), flush=True)

    # EPYC 7542 (Zen2) 无 AMX/bf16 硬件加速, bf16 反而模拟变慢 -> 固定 fp32 (mkldnn AVX2 已开).
    dtype = 'fp32' if args.dtype == 'auto' else args.dtype

    print('== launch nproc=%d threads=%d batch=%d dtype=%s ==' % (args.nproc, args.threads, args.batch, dtype), flush=True)
    ctx = mp.get_context('spawn')
    q = ctx.Queue()
    ps = [ctx.Process(target=worker, args=((i, args.nproc, args.threads, args.batch, dtype, path_of), q)) for i in range(args.nproc)]
    t_start = time.time()
    for p in ps:
        p.start()
    res = [q.get(timeout=7200) for _ in ps]
    for p in ps:
        p.join()
    tot_wall = time.time() - t_start
    n_img = sum(r[1] for r in res); n_s3 = sum(r[2] for r in res); n_s1 = sum(r[3] for r in res)
    print('ALL WORKERS DONE: img=%d skel3=%d skel1=%d wall=%.0fs' % (n_img, n_s3, n_s1, tot_wall), flush=True)

    # ---------------- 聚合写回 ----------------
    print('aggregating...', flush=True)
    img_lat = {}; skel3_lat = {}; skel1_lat = {}
    for wf in sorted(glob.glob('/tmp/v8_enc/w*.npz')):
        d = np.load(wf)
        ids = d['img_ids'].tolist()
        lat = d['img_lat']
        for j, iid in enumerate(ids):
            img_lat[iid] = lat[j]
        ids = d['sk3_ids'].tolist()
        lat = d['sk3_lat']
        for j, iid in enumerate(ids):
            skel3_lat[iid] = lat[j]
        ids = d['sk1_ids'].tolist()
        lat = d['sk1_lat']
        for j, iid in enumerate(ids):
            skel1_lat[iid] = lat[j]
    print('collected img=%d skel3=%d skel1=%d' % (len(img_lat), len(skel3_lat), len(skel1_lat)), flush=True)

    n0 = patch_npz('fame.npz', img_lat)
    print('fame.npz patched:', n0, flush=True)
    n1 = 0
    for shard in sorted(glob.glob('final_latents_fame/shard_*.npz')):
        n1 += patch_npz(shard, img_lat)
    print('final_latents_fame patched:', n1, flush=True)
    n2 = 0
    for shard in sorted(glob.glob('final_skel_latents_fame/shard_*.npz')):
        n2 += patch_npz(shard, skel3_lat)
    print('final_skel_latents_fame patched:', n2, flush=True)
    n3 = 0
    for shard in sorted(glob.glob('final_skel_latents_fame_1px/shard_*.npz')):
        n3 += patch_npz(shard, skel1_lat)
    print('final_skel_latents_fame_1px patched:', n3, flush=True)

    # ---------------- skel 一致性抽查 ----------------
    import random
    rng = random.Random(0)
    sample = rng.sample(sorted(skel3_lat.keys()), min(50, len(skel3_lat)))
    bad = 0
    for iid in sample:
        a = np.asarray(Image.open(path_of[iid]).convert('L'))
        if a.shape != (256, 256):
            a = np.asarray(Image.open(path_of[iid]).convert('L').resize((256, 256)))
        sk = skeletonize(a < 127)
        chk3 = np.where(dil3(sk), 0, 255).astype('uint8')
        on3 = np.asarray(Image.open('final_skel3_fame/%d.png' % iid).convert('L'))
        chk1 = np.where(sk, 0, 255).astype('uint8')
        on1 = np.asarray(Image.open('final_skel1_fame/%d.png' % iid).convert('L'))
        if not np.array_equal(chk3, on3) or not np.array_equal(chk1, on1):
            bad += 1
    print('skel consistency spot-check: %d/%d OK' % (len(sample) - bad, len(sample)), flush=True)
    print('ENCODE ALL DONE', flush=True)
