import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, "/root/Workspace/xy/DiT")
import torch
from models import DiT_2Cond_models

# Build model, check dtype of key components
m = DiT_2Cond_models["DiT-2Cond-B/4"](
    input_size=64, in_channels=3,
    num_calligraphers=1011, num_characters=35130,
    condition_fusion="factorized_add", callig_embed_dim=128, char_embed_dim=768,
    learn_sigma=True, cond_drop_all_prob=0.05, cond_drop_one_prob=0.25,
).cuda().half().eval()

# Check embedder dtype
print("y_char_embedder table dtype:", m.y_char_embedder.embedding_table.weight.dtype)
print("y_char_embedder table device:", m.y_char_embedder.embedding_table.weight.device)

# Test forward
z = torch.randn(4, 3, 64, 64, device="cuda", dtype=torch.float16)
t = torch.tensor([0,0,0,0], device="cuda", dtype=torch.long)
yc = torch.tensor([0,0,0,0], device="cuda", dtype=torch.long)
ych = torch.tensor([0,0,0,0], device="cuda", dtype=torch.long)

with torch.no_grad():
    try:
        out = m(z, t, yc, ych)
        print("forward OK:", out.shape, out.dtype)
    except Exception as e:
        print("forward error:", e)
        # Check where dtype mismatch is
        for name, p in m.named_parameters():
            if p.dtype != torch.float16:
                print(f"  NOT fp16: {name} dtype={p.dtype}")
                break
