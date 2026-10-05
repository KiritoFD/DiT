import os, sys, json
import torch as th

ROOT = "/root/Workspace/xy/DiT"
os.chdir(ROOT)
sys.path.insert(0, ROOT)

from src.eval import model_io
from src.eval.in_mem_eval import run_in_mem_eval
from src.model.joint_skel2img import JointSkel2Img
from types import SimpleNamespace

dev = th.device("cuda")

ckpt_path = "exp/v35_union/20261002-003600-v35-union-1step/checkpoints/0002000.pt"
ck = th.load(ckpt_path, map_location="cpu")
print("Loading Stage 1 & Stage 2 from step 2,000 ckpt...")

gen_sd = ck["gen_ema"]
gen_ckpt = ck["gen_ckpt"]
bak_ckpt = ck["bak_ckpt"]

gen, ga = model_io.load_model_from_ckpt(gen_ckpt, device=dev, use_ema=True)
bak, ba = model_io.load_model_from_ckpt(bak_ckpt, device=dev, use_ema=True)

# Load updated weights
gen.load_state_dict({(k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v for k, v in gen_sd.items()})
gen.eval()
bak.eval()

wrapper = JointSkel2Img(gen, bak, gen_steps=25).to(dev).eval()

ev_args = SimpleNamespace(**{k: v for k, v in vars(ba).items() if not k.startswith("_")})
ev_args.eval_blend_alpha, ev_args.eval_cfg, ev_args.eval_steps = 0.0, 1.0, 50
ev_args.eval_self_cond, ev_args.img_root = False, None

# ★ USE THE OFFICIAL 50k_v2 SHARDS FOR STRICT84!
ev_args.eval_skel_latent_shards_dir = "data/50k_v2_glyph15k/shards_std"
ev_args.eval_skel_latent_shards_dir_pred = ""

run_dir = "exp/v35_union/eval_step2000_official"
os.makedirs(run_dir, exist_ok=True)

print("\n--- Running Evaluation with OFFICIAL shards_std on Strict84 ---")
res = run_in_mem_eval(
    wrapper, ev_args, 2000, dev, run_dir,
    sets=[("strict84_official_std", "assets/eval_top10_strict_subset84.csv", 84)]
)

print("\n=== Strict84 真实真实端到端生成成绩 ===")
for k, v in res.items():
    print(f"[{k}]: {v}")
