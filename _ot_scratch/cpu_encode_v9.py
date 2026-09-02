# -*- coding: utf-8 -*-
"""cpu_encode_v9.py - v7 清洗后变更图的多进程 CPU encode (分阶段, 幂等可断点续跑).
并行: nproc 进程 x 每进程 threads (oneDNN/mkldnn AVX2 SIMD), nice 降优先级避免挤训练.
阶段:
  img     - 重算 skel3/skel1 PNG + 编码 img latent            -> /tmp/v9_enc/w{me}_img.npz
  skel3   - 编码 skel3 latent (灰度*0.18215 前已是 VAE 输出; 输入 3ch 灰度扩展) -> w{me}_sk3.npz
  skel1   - 编码 skel1 latent (PNG RGB)                        -> w{me}_sk1.npz
  writeback - 聚合所有 w*.npz 写回 fame.npz + 各 shards + skel 一致性抽查
每个阶段若产物已存在则跳过 (可断点续跑).
用法: python cpu_encode_v9.py --nproc 16 --threads 4 --batch 16 --phase img
      python cpu_encode_v9.py --nproc 16 --threads 4 --batch 16 --phase skel3
      python cpu_encode_v9.py --nproc 16 --threads 4 --batch 16 --phase skel1
      python cpu_encode_v9.py --phase writeback
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

OUT = '/tmp/v9_enc'

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

def skel_from(iid, paths):
    """返回 (skel3_arr, skel1_arr) 从原图计算."""
    p = paths[iid]
    a = np.asarray(Image.open(p).convert('L'))
    if a.shape != (256, 256):
        a = np.asarray(Image.open(p).convert('L').resize((256, 256)))
    sk = skeletonize(a < 127)
    sk3 = np.where(dil3(sk), 0, 255).astype('uint8')
    sk1 = np.where(sk, 0, 255).astype('uint8')
    return sk3, sk1

# ---------------- worker ----------------
def worker(me, nproc, ithr, batch, paths, tag, q):
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
    def enc(x):
        with torch.no_grad():
            return (vae.encode(x).latent_dist.mode() * 0.18215).to(torch.float16).cpu().numpy()

    keys = sorted(paths.keys())
    mine = [k for k in keys if int(k) % nproc == me]
    out_f = '%s/w%d_%s.npz' % (OUT, me, tag)
    if os.path.exists(out_f):
        print('[proc%d %s] skip (exists)' % (me, tag), flush=True)
        q.put((me, 0))
        return

    t0 = time.time()
    if tag == 'img':
        os.makedirs('final_skel3_fame', exist_ok=True)
        os.makedirs('final_skel1_fame', exist_ok=True)
        ids = []; lat = []
        acc = []; acc_id = []
        def flush():
            if not acc:
                return
            xb = torch.from_numpy(np.stack(acc))
            z = enc(xb)
            for j, iid in enumerate(acc_id):
                ids.append(iid); lat.append(z[j])
            acc.clear(); acc_id.clear()
        done = 0
        for iid in mine:
            sk3, sk1 = skel_from(iid, paths)
            Image.fromarray(sk3, 'L').save('final_skel3_fame/%d.png' % iid)
            Image.fromarray(sk1, 'L').save('final_skel1_fame/%d.png' % iid)
            acc.append(tf(Image.open(paths[iid]).convert('RGB')).numpy()); acc_id.append(iid)
            if len(acc) >= batch:
                flush(); done += batch
                print('[proc%d img] %d/%d' % (me, done, len(mine)), flush=True)
        flush()
        a_ids = np.array(ids, dtype=np.int64)
        lat = np.stack(lat) if a_ids.size else np.zeros((0, 4, 32, 32), np.float16)
        np.savez_compressed(out_f, ids=a_ids, lat=lat)
    elif tag == 'skel3':
        ids = []; lat = []
        acc = []; acc_id = []
        def flush():
            if not acc:
                return
            xb = torch.from_numpy(np.stack(acc))[:, None].repeat(1, 3, 1, 1)  # (B,H,W)->(B,3,H,W)
            z = enc(xb)
            for j, iid in enumerate(acc_id):
                ids.append(iid); lat.append(z[j])
            acc.clear(); acc_id.clear()
        done = 0
        for iid in mine:
            sk3, _ = skel_from(iid, paths)
            acc.append((sk3.astype(np.float32) / 255.0 * 2 - 1)); acc_id.append(iid)
            if len(acc) >= batch:
                flush(); done += batch
                print('[proc%d skel3] %d/%d' % (me, done, len(mine)), flush=True)
        flush()
        a_ids = np.array(ids, dtype=np.int64)
        lat = np.stack(lat) if a_ids.size else np.zeros((0, 4, 32, 32), np.float16)
        np.savez_compressed(out_f, ids=a_ids, lat=lat)
    elif tag == 'skel1':
        ids = []; lat = []
        acc = []; acc_id = []
        def flush():
            if not acc:
                return
            xb = torch.from_numpy(np.stack(acc))
            z = enc(xb)
            for j, iid in enumerate(acc_id):
                ids.append(iid); lat.append(z[j])
            acc.clear(); acc_id.clear()
        done = 0
        for iid in mine:
            _, sk1 = skel_from(iid, paths)
            Image.fromarray(sk1, 'L').save('final_skel1_fame/%d.png' % iid)
            acc.append(tf(Image.fromarray(sk1, 'L').convert('RGB')).numpy()); acc_id.append(iid)
            if len(acc) >= batch:
                flush(); done += batch
                print('[proc%d skel1] %d/%d' % (me, done, len(mine)), flush=True)
        flush()
        a_ids = np.array(ids, dtype=np.int64)
        lat = np.stack(lat) if a_ids.size else np.zeros((0, 4, 32, 32), np.float16)
        np.savez_compressed(out_f, ids=a_ids, lat=lat)
    dt = time.time() - t0
    print('[proc%d %s] DONE %d in %.0fs (%.1f img/s)' % (me, tag, len(ids), dt, len(ids) / dt), flush=True)
    q.put((me, len(ids)))


def run_phase(phase, nproc, ithr, batch, path_of):
    os.makedirs(OUT, exist_ok=True)
    print('== phase=%s nproc=%d threads=%d batch=%d ==' % (phase, nproc, ithr, batch), flush=True)
    ctx = mp.get_context('spawn')
    q = ctx.Queue()
    ps = [ctx.Process(target=worker, args=(i, nproc, ithr, batch, path_of, phase, q)) for i in range(nproc)]
    t_start = time.time()
    for p in ps:
        p.start()
    res = [q.get(timeout=7200) for _ in ps]
    for p in ps:
        p.join()
    print('PHASE %s DONE wall=%.0fs total=%d' % (phase, time.time() - t_start, sum(r[1] for r in res)), flush=True)


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


def writeback(path_of):
    print('aggregating...', flush=True)
    img_lat = {}; skel3_lat = {}; skel1_lat = {}
    for wf in sorted(glob.glob(OUT + '/w*_img.npz')):
        d = np.load(wf)
        for j, iid in enumerate(d['ids'].tolist()):
            img_lat[iid] = d['lat'][j]
    for wf in sorted(glob.glob(OUT + '/w*_sk3.npz')):
        d = np.load(wf)
        for j, iid in enumerate(d['ids'].tolist()):
            skel3_lat[iid] = d['lat'][j]
    for wf in sorted(glob.glob(OUT + '/w*_sk1.npz')):
        d = np.load(wf)
        for j, iid in enumerate(d['ids'].tolist()):
            skel1_lat[iid] = d['lat'][j]
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

    import random
    rng = random.Random(0)
    sample = rng.sample(sorted(skel3_lat.keys()), min(50, len(skel3_lat)))
    bad = 0
    for iid in sample:
        sk3, sk1 = skel_from(iid, path_of)
        on3 = np.asarray(Image.open('final_skel3_fame/%d.png' % iid).convert('L'))
        on1 = np.asarray(Image.open('final_skel1_fame/%d.png' % iid).convert('L'))
        if not np.array_equal(sk3, on3) or not np.array_equal(sk1, on1):
            bad += 1
    print('skel consistency spot-check: %d/%d OK' % (len(sample) - bad, len(sample)), flush=True)
    print('WRITEBACK DONE', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--nproc', type=int, default=16)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--phase', choices=['img', 'skel3', 'skel1', 'writeback'], required=True)
    args = ap.parse_args()
    import torch
    print('torch %s mkldnn=%s' % (torch.__version__, torch.backends.mkldnn.is_available()), flush=True)
    print('loading changed ids...', flush=True)
    changed = load_changed()
    path_of = build_path_map()
    path_of = {i: path_of[i] for i in changed if i in path_of}
    print('changed=%d with_path=%d' % (len(changed), len(path_of)), flush=True)
    if args.phase == 'writeback':
        writeback(path_of)
    else:
        run_phase(args.phase, args.nproc, args.threads, args.batch, path_of)
