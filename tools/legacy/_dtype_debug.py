import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, "/root/Workspace/xy/DiT")
import torch
from models import DiT_2Cond_models

m = DiT_2Cond_models["DiT-2Cond-B/4"](
    input_size=64, in_channels=3,
    num_calligraphers=1011, num_characters=35130,
    condition_fusion="factorized_add", callig_embed_dim=128, char_embed_dim=768,
    learn_sigma=True, cond_drop_all_prob=0.05, cond_drop_one_prob=0.25,
).cuda().half().eval()

# Find all non-fp16 params
bad = []
for name, p in m.named_parameters():
    if p.dtype != torch.float16:
        bad.append((name, p.dtype))
for name, buf in m.named_buffers():
    if buf.dtype != torch.float16:
        bad.append((name, str(buf.dtype)+" (buffer)"))
print(f"non-fp16 params/buffers: {len(bad)}")
for name, dt in bad[:20]:
    print(f"  {name}: {dt}")

# Try forward step by step
z = torch.randn(4, 3, 64, 64, device="cuda", dtype=torch.float16)
t = torch.tensor([0,0,0,0], device="cuda", dtype=torch.long)
yc = torch.tensor([0,0,0,0], device="cuda", dtype=torch.long)
ych = torch.tensor([0,0,0,0], device="cuda", dtype=torch.long)

with torch.no_grad():
    # x_embedder
    x = m.x_embedder(z)
    print(f"x_embedder: {x.dtype} {x.shape}")
    # t_embedder
    t_emb = m.t_embedder(t)
    print(f"t_embedder: {t_emb.dtype} {t_emb.shape}")
    # cond
    try:
        e_callig = m.y_callig_embedder(yc, False)
        print(f"y_callig: {e_callig.dtype} {e_callig.shape}")
    except Exception as e:
        print(f"y_callig error: {e}")
    try:
        e_char = m.y_char_embedder(ych, False)
        print(f"y_char: {e_char.dtype} {e_char.shape}")
    except Exception as e:
        print(f"y_char error: {e}")
    try:
        e_callig = m.y_callig_embedder(yc, False)
        e_char = m.y_char_embedder(ych, False)
        c = (m.callig_proj(e_callig) + m.char_proj(e_char)) / (2 ** 0.5)
        print(f"cond_fused: {c.dtype} {c.shape}")
    except Exception as e:
        print(f"cond_fused error: {e}")
