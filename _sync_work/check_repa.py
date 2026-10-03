import os, sys, numpy as np

base = "data/dino_cache"
for d in sorted(os.listdir(base)):
    p = os.path.join(base, d)
    if not os.path.isdir(p):
        continue
    print("=" * 60)
    print(p)
    for f in sorted(os.listdir(p)):
        fp = os.path.join(p, f)
        sz = os.path.getsize(fp) / 1e6
        print(f"  {f}  {sz:.1f} MB")
        if f.endswith(".npy"):
            try:
                a = np.load(fp, mmap_mode="r")
                print(f"     shape={a.shape} dtype={a.dtype}")
            except Exception as e:
                print(f"     (load err) {e}")
        elif f.endswith(".npz"):
            try:
                z = np.load(fp)
                for k in list(z.keys())[:3]:
                    print(f"     key={k} shape={z[k].shape}")
            except Exception as e:
                print(f"     (load err) {e}")

# how does train.py key the cache?
print("\n" + "=" * 60)
print("grep repa_cache usage in src/loss/repa.py / src/model/repa.py")
