import csv, glob, os, sys
from collections import Counter
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, "/root/Workspace/xy/DiT")
os.chdir("/root/Workspace/xy/DiT")
import torch

# 活跃书家 id
rows = list(csv.DictReader(open('assets/train_fame3_clean_v8.csv', encoding='utf-8')))
c = Counter(int(r['calligrapher_id']) for r in rows)
active = sorted(c)
print(f"活跃书家数 = {len(active)}  id = {active}")

ck_path = sorted(glob.glob("assets/results/v10b_stdskel_fame3_sp2/*/checkpoints/*.pt"),
                 key=lambda p: int(os.path.basename(p).split(".")[0]))[-1]
ck = torch.load(ck_path, map_location="cpu", weights_only=False)
a = ck.get("args", {}) or {}
if not isinstance(a, dict):
    a = vars(a) if hasattr(a, "__dict__") else {}

from src.model import DiT_2Cond_models
arch = dict(norm_type=a.get("norm_type","rms"), mlp_type=a.get("mlp_type","swiglu"),
            qk_norm=bool(a.get("qk_norm",1)), rope=bool(a.get("rope",1)),
            rope_theta=float(a.get("rope_theta",100.0)), attn_impl="sdpa")
model = DiT_2Cond_models[a.get("model","DiT-2Cond-Sp/2")](
    num_calligraphers=int(a.get("num_calligraphers",1013)),
    num_characters=int(a.get("num_characters",35130)),
    condition_fusion=a.get("condition_fusion","factorized_add"),
    callig_embed_dim=int(a.get("callig_embed_dim",128)),
    char_embed_dim=int(a.get("char_embed_dim",384)),
    char_proj_mode=a.get("char_proj_mode","mlp"),
    freeze_char_table=bool(a.get("freeze_char_table",False)),
    cond_drop_all_prob=0.1, cond_drop_one_prob=0.4,
    cond_drop_which_glyph_prob=0.85, use_checkpoint=False, learn_sigma=False,
    use_glyph_cond=True, use_char_cond=not bool(a.get("no_char_cond",False)),
    glyph_scale_init=float(a.get("glyph_scale_init",0.6)),
    glyph_drop_prob=float(a.get("glyph_drop_prob",0.1)),
    glyph_inject_layers=int(a.get("glyph_inject_layers",4)),
    glyph_embedder_depth=int(a.get("glyph_embedder_depth",2)), **arch)
sd = ck.get("ema") or ck.get("model") or ck
sd = {(k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v for k, v in sd.items()}
model.load_state_dict(sd, strict=False)

# 找 callig embedding
emb = None
for name, p in model.named_parameters():
    if "callig" in name and "embed" in name and "weight" in name:
        emb = p.detach().float().clone(); break
print(f"embedding 表形状 = {tuple(emb.shape)}  (num_calligraphers={a.get('num_calligraphers')})")

# 只取活跃书家行
act = emb[active]
nc = act.shape[0]
e = act / (act.norm(dim=1, keepdim=True) + 1e-8)
sim = e @ e.T
off = sim[~torch.eye(nc, dtype=torch.bool)]
print(f"\n[活跃 {nc} 书家 embedding 塌缩检查]")
print(f"  归一化 pairwise 余弦 = mean {off.mean():.4f} / std {off.std():.4f}")
print(f"  离均值距离 std/mean = {((act-act.mean(0)).norm(dim=1)).std():.4f}/{((act-act.mean(0)).norm(dim=1)).mean():.4f}")
print(f"  ||emb|| 分布 = mean {act.norm(dim=1).mean():.3f} ± {act.norm(dim=1).std():.3f}")

# 对比: 死行 vs 活跃行的分布差异 (活跃行是否被训练"拉开")
dead = emb[[i for i in range(emb.shape[0]) if i not in active]]
print(f"\n[死行 vs 活跃行]")
print(f"  死行数 = {dead.shape[0]}, 活跃行数 = {act.shape[0]}")
print(f"  死行 ||emb|| = mean {dead.norm(dim=1).mean():.3f} ± {dead.norm(dim=1).std():.3f}")
print(f"  活跃行 ||emb|| = mean {act.norm(dim=1).mean():.3f} ± {act.norm(dim=1).std():.3f}")
print(f"  (训练不会改变 embedding 模长明显; 关键看活跃行 pairwise cos 是否趋近 1 = 塌缩)")

# proj 后: 两两活跃书家距离 (是否可区分)
with torch.no_grad():
    y = torch.tensor(active)
    ec = model.y_callig_embedder(y, False)
    cp = model.callig_proj(ec)
    d = (cp.unsqueeze(0) - cp.unsqueeze(1)).norm(dim=-1)
    off_d = d[~torch.eye(nc, dtype=torch.bool)]
    print(f"\n[proj 后活跃书家两两距离]")
    print(f"  mean {off_d.mean():.3f} / min {off_d.min():.3f} / ||proj后|| mean {cp.norm(dim=-1).mean():.3f}")
    print(f"  最接近的两书家距离 = {off_d.min():.3f} (若≈0 则这两书家风格无法区分)")