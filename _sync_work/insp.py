import torch, glob, os
ck = sorted(glob.glob("assets/results/v12_pretrain_S_cat_fame_kxl_tj_px60/*/checkpoints/*.pt"))
ck = [p for p in ck if os.path.getsize(p) > 1e6][-1]
print("ckpt:", ck)
d = torch.load(ck, map_location="cpu", weights_only=False)
print("top keys:", list(d.keys())[:10])
for k in d:
    v = d[k]
    if isinstance(v, dict):
        ks = list(v.keys())[:6]
        print(f"  {k}: dict with {len(v)} keys, sample={ks}")
    else:
        print(f"  {k}: {type(v).__name__}")
