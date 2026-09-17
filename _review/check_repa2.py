import os, re, csv, numpy as np

# px60 image ids actually used by the training csv
rows = list(csv.DictReader(open("assets/train_fame-kxl-tj-px60.csv", encoding="utf-8")))
px = [int(re.search(r"(\d+)\.png", r["image_path"]).group(1)) for r in rows]
px_s = set(px)
print(f"px60 rows={len(rows)}  distinct ids={len(px_s)}  min={min(px)} max={max(px)}")

# files present on disk
d = "data/fame-kxl-tj-px60/imgs"
disk = []
for f in os.listdir(d):
    m = re.match(r"(\d+)\.png", f)
    if m:
        disk.append(int(m.group(1)))
disk_s = set(disk)
print(f"px60 imgs on disk={len(disk)}  min={min(disk)} max={max(disk)}")

for name in ["base_v1", "base_sym_v1", "v11_sym_full"]:
    p = f"data/dino_cache/{name}/ids.npy"
    if not os.path.exists(p):
        print(f"\n{name}: MISSING")
        continue
    ids = np.load(p)
    s = set(ids.tolist())
    inter = px_s & s
    print(f"\n{name}: n={len(s)} min={ids.min()} max={ids.max()}")
    print(f"   overlap with px60 train ids: {len(inter)} / {len(px_s)} "
          f"({len(inter)/len(px_s):.1%})")
    if len(inter) == 0:
        print("   >>> NO OVERLAP -> REPA cache unusable for this dataset (silently no-op?)")
    elif len(inter) < len(px_s):
        print(f"   >>> PARTIAL -> {len(px_s)-len(inter)} samples would miss teacher feats")
    else:
        print("   >>> FULL COVERAGE -> cache is valid for this dataset")
