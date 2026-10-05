import sys, os, json
base = "/home/ds/Workspace/moyi/ref/moyi"
sys.path.insert(0, base)
sys.path.insert(0, os.path.join(base, "moyun"))

import torch
from moyun_2 import DiT_models
from diffusers.models import AutoencoderKL
from torchvision.utils import save_image

device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. Load Vocab
vocab_path = "/home/ds/Workspace/moyi/results/moyi_top10_rf/vocab.json"
with open(vocab_path, "r", encoding="utf-8") as f:
    vocabs = json.load(f)
callig_vocab = vocabs["callig_vocab"]
script_vocab = vocabs["script_vocab"]
char_vocab = vocabs["char_vocab"]

# 2. Prompts to generate
prompts = [
    {"idx": 10, "callig": "王羲之", "script": "楷", "char": "旨"},
    {"idx": 11, "callig": "王羲之", "script": "行", "char": "好"},
    {"idx": 21, "callig": "颜真卿", "script": "楷", "char": "其"},
    {"idx": 22, "callig": "颜真卿", "script": "行", "char": "憫"},
    {"idx": 13, "callig": "米芾", "script": "行", "char": "墟"},
    {"idx": 19, "callig": "赵孟頫", "script": "行", "char": "匠"},
    {"idx": 20, "callig": "赵孟頫", "script": "隶", "char": "盤"},
    {"idx": 6, "callig": "柳公权", "script": "楷", "char": "連"},
]

# 3. Load Model
ckpt_path = "/home/ds/Workspace/moyi/results/moyi_top10_rf/checkpoints/moyi_0020000.pt"
print(f"Loading Moyi checkpoint from {ckpt_path}...")
model = DiT_models["moyun-12channel-B"](
    input_size=32,
    num_classes=6000,
    learn_sigma=False,
    if_rope=False
).to(device)

ckpt = torch.load(ckpt_path, map_location="cpu")
state_dict = ckpt.get("ema", ckpt.get("model"))
model.load_state_dict(state_dict)
model.eval()

# 4. Load VAE
vae_path = "/home/ds/Workspace/moyi/models/sd-vae-ft-ema"
vae = AutoencoderKL.from_pretrained(vae_path).to(device)
vae.eval()

out_dir = "/home/ds/Workspace/moyi/results/moyi_top10_rf/eval_samples_step20000"
os.makedirs(out_dir, exist_ok=True)

# 5. Sampling
torch.manual_seed(42)
steps = 50
for p in prompts:
    idx = p["idx"]
    c_name = p["callig"]
    s_name = p["script"]
    ch = p["char"]
    
    cid = callig_vocab.get(c_name, 0)
    sid = script_vocab.get(s_name, 0)
    chid = char_vocab.get(ch, 0)
    
    y = torch.tensor([[cid, sid, chid]], device=device)
    stroke = torch.zeros(1, dtype=torch.long, device=device)
    
    z = torch.randn(1, 12, 32, 32, device=device)
    ts = torch.linspace(0, 1, steps + 1, device=device)
    
    with torch.no_grad():
        for i in range(steps):
            t = ts[i].expand(1)
            dt = ts[i + 1] - ts[i]
            v_pred = model(z, t, y, stroke)
            if isinstance(v_pred, (tuple, list)):
                v_pred = v_pred[0]
            z = z + v_pred * dt
            
        img_latent = z[:, :4, :, :]
        decoded = vae.decode(img_latent / 0.18215).sample
        decoded = torch.clamp((decoded + 1.0) / 2.0, 0.0, 1.0)
        
        save_path = os.path.join(out_dir, f"moyi_g{idx}_{c_name}_{s_name}_{ch}.png")
        save_image(decoded, save_path)
        print(f"✓ Saved {save_path}")

print("All Moyi eval samples generated successfully!")
