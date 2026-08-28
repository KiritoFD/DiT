import sys, numpy as np
d = np.load(sys.argv[1])
print("latents", d["latents"].shape, d["latents"].dtype)
print("img_ids", d["img_ids"].shape, d["img_ids"].dtype)
print("first ids", d["img_ids"][:5])
d.close()
