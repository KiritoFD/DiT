import json, numpy as np, os
os.chdir("/root/Workspace/xy/DiT")
emb768 = np.load("pretrained_models/dino_embeddings/glyph_dino_embeddings.npy")
emb384 = np.load("pretrained_models/dino_embeddings/glyph_dino_embeddings_384.npy")
idx = json.load(open("pretrained_models/dino_embeddings/glyph_dino_index.json"))
glyphs = idx.get("glyphs", idx)
print("emb768:", emb768.shape, "emb384:", emb384.shape, "index entries:", len(glyphs))
print("index[0]:", glyphs[0], "index[-1]:", glyphs[-1])
print("first elem type:", type(glyphs[0]))
# Check the table injection would work: gid = sid*7026+cid, need gid < 35130
max_gid = max(int(s)*7026+int(c) for s,c in glyphs[:20000])
print("max_gid (first 20k):", max_gid, "table size 35130")
