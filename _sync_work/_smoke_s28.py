# -*- coding: utf-8 -*-
"""s28 smoke test v2: 标准字形 DINO embedder + ln_only 直通 (零可训练投影)。"""
import os, sys, io
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.chdir("/root/Workspace/xy/DiT")
sys.path.insert(0, os.getcwd())
import torch
import json

from src.model import DiT_2Cond_models

cfg = json.load(open("src/train/configs/s28_std_dino_pretrain.json", encoding="utf-8"))
print("=== build model ===")
model = DiT_2Cond_models[cfg["model"]](
    input_size=32, in_channels=4, learn_sigma=False,
    num_calligraphers=cfg["num_calligraphers"], num_characters=cfg["num_characters"],
    condition_fusion=cfg["condition_fusion"],
    callig_embed_dim=cfg["callig_embed_dim"], char_embed_dim=cfg["char_embed_dim"],
    char_proj_mode=cfg.get("char_proj_mode", "full"),
    cond_drop_all_prob=cfg["cond_drop_all_prob"], cond_drop_one_prob=cfg["cond_drop_one_prob"],
    cond_drop_which_glyph_prob=cfg["cond_drop_which_glyph_prob"],
    use_std_dino_char_embedder=cfg.get("use_std_dino_char_embedder", False),
    std_dino_table_path=cfg.get("std_dino_table_path"),
    norm_type=cfg.get("norm_type","rms"), mlp_type=cfg.get("mlp_type","swiglu"),
    qk_norm=bool(cfg.get("qk_norm",1)), rope=bool(cfg.get("rope",1)),
    rope_theta=cfg.get("rope_theta",100.0), attn_impl=cfg.get("attn_impl","sdpa"),
)
model.eval().cuda()

emb = model.y_char_embedder
print("char_embedder:", type(emb).__name__)
print("  table:", tuple(emb.char_table.shape), "dtype", emb.char_table.dtype, "| downsample:", emb.downsample)
print("  embedder trainable params:", sum(p.numel() for p in emb.parameters() if p.requires_grad))
print("  char_proj:", type(model.char_proj).__name__, getattr(model.char_proj, 'weight', None) is not None)
total_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("  TOTAL trainable params:", f"{total_train:,}")

# forward
B, NCH = 8, 7026
x = torch.randn(B, 4, 32, 32).cuda()
t = torch.rand(B).cuda()
y_callig = torch.randint(0, cfg["num_calligraphers"], (B,)).cuda()
script = torch.randint(0, 5, (B,)).cuda()
char = torch.randint(0, NCH, (B,)).cuda()
y_char = script * NCH + char
with torch.no_grad():
    out = model(x, t, y_callig, y_char)
print("  out:", tuple(out.shape))

# 形近字直通后余弦（应保持高，且不经过可训练投影）
import csv
charid2char = {}
for r in csv.DictReader(open("assets/train_fame.csv", encoding="utf-8")):
    charid2char[int(r["character_id"])] = r["character"]
def cid(ch):
    for c, cc in charid2char.items():
        if cc == ch: return c
    return None
with torch.no_grad():
    cp = model.char_proj  # LayerNorm (直通)
    for a, b in [("土","士"),("大","太"),("王","玉"),("水","火")]:
        ca, cb = cid(a), cid(b)
        if ca is None or cb is None: continue
        ea = emb.char_table[ca]; eb = emb.char_table[cb]
        cos_raw = float(torch.cosine_similarity(ea, eb, dim=0))
        pa = cp(ea.unsqueeze(0)).squeeze(); pb = cp(eb.unsqueeze(0)).squeeze()
        cos_proj = float(torch.cosine_similarity(pa, pb, dim=0))
        print(f"  {a}/{b}: 插值DINO cos={cos_raw:.3f}  LayerNorm后 cos={cos_proj:.3f}")
print("\nOK")
