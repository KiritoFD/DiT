import numpy as np

z1 = np.load("/root/Workspace/xy/DiT/data/50k_v2_glyph15k/shards_std/shard_00000.npz")
z2 = np.load("/root/Workspace/xy/DiT/data/top10_style23/predskel_std84_e2e/shard_00000.npz")

print("z1 (official 50k_v2 shards_std) IDs:", z1["img_ids"][:5], "mean:", float(z1["latents"].mean()), "std:", float(z1["latents"].std()))
print("z2 (predskel_std84_e2e) IDs:", z2["img_ids"][:5], "mean:", float(z2["latents"].mean()), "std:", float(z2["latents"].std()))

id0 = int(z2["img_ids"][0])
m = np.where(z1["img_ids"] == id0)[0]
if len(m) > 0:
    lat1 = z1["latents"][m[0]]
    lat2 = z2["latents"][0]
    print(f"Match ID {id0}: lat1 mean={lat1.mean():.4f}, lat2 mean={lat2.mean():.4f}, diff={float(np.abs(lat1 - lat2).mean()):.4f}")
else:
    print(f"ID {id0} NOT FOUND in z1! IDs in z1 are completely different!")
