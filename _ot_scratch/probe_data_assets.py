import os, glob, numpy as np

os.chdir('/root/Workspace/xy/DiT')

def summarize(name):
    fs = sorted(glob.glob(name + '/shard_*.npz'))
    print('== %s ==' % name)
    print('  shard files:', len(fs))
    if not fs:
        print('  (empty)')
        return None
    d = np.load(fs[0])
    print('  keys:', list(d.keys()))
    print('  shard0 latents dtype/shape:', d['latents'].dtype, d['latents'].shape)
    print('  shard0 img_ids[:3]:', d['img_ids'][:3])
    tot = 0
    idset = set()
    for f in fs[:5]:
        dd = np.load(f)
        tot += dd['latents'].shape[0]
        idset.update(dd['img_ids'].tolist())
    print('  first %d shards: %d imgs, unique ids %d' % (min(5, len(fs)), tot, len(idset)))
    return fs

for name in ['final_latents_fame_clean', 'final_latents_fame',
             'final_skel_latents_fame_1px', 'final_skel_latents_fame',
             'final_latents_f4', 'final_skel_latents_eval_1px']:
    try:
        summarize(name)
    except Exception as e:
        print('== %s == ERR %r' % (name, e))

# inode check: is final_latents_fame_clean a symlink or hardlink of final_latents_fame?
for a, b in [('final_latents_fame_clean/shard_0000.npz', 'final_latents_fame/shard_0000.npz')]:
    if os.path.exists(a) and os.path.exists(b):
        print(a, '==', b, 'hardlink?', os.stat(a).st_ino == os.stat(b).st_ino,
              'size', os.path.getsize(a), os.path.getsize(b))
    print('symlink?', os.path.islink(a), os.path.islink(b))

# fame.npz img_ids coverage vs clean shards
npz = np.load('fame.npz')
print('fame.npz ids:', npz['img_ids'].shape, 'latents:', npz['latents'].shape, npz['latents'].dtype)
d0 = np.load('final_latents_fame_clean/shard_0000.npz')
print('clean shard0 ids[:5]:', d0['img_ids'][:5])