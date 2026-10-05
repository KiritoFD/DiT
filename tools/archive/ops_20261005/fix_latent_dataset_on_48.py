p = "/home/ds/Workspace/DiT/src/utils/latent_dataset.py"
with open(p, "r", encoding="utf-8") as f:
    s = f.read()

s = s.replace("np.load(sp)", "np.load(sp, allow_pickle=True)")
s = s.replace("np.load(shards[0])", "np.load(shards[0], allow_pickle=True)")
s = s.replace(
    "np.load(_sk_shards[0])", "np.load(_sk_shards[0], allow_pickle=True)"
)

with open(p, "w", encoding="utf-8") as f:
    f.write(s)

print("latent_dataset.py successfully patched with allow_pickle=True!")
