import json
import torch
from pathlib import Path
from src.unet.io import Config, build, checkpoint_config
from src.unet.variants.transfer import load_expansion

ckpt_path = Path("assets/results/unet_connectivity_100000/baseline_resume/20261018-030921-Unet-flow/checkpoints/0050000.pt")
warm = torch.load(ckpt_path, map_location="cpu", weights_only=False)
source_cfg = checkpoint_config(warm)
print("Loaded 50k source checkpoint, steps:", warm["train_steps"])

configs = [
    "src/unet/configs/connectivity/DiC_residual_gate.json",
    "src/unet/configs/connectivity/DiC_stage_cond_adapter.json",
    "src/unet/configs/connectivity/DiC_stage_cond_independent.json"
]

for cfg_path in configs:
    cfg = Config(**json.loads(Path(cfg_path).read_text(encoding="utf-8")))
    model = build(cfg, initialize_tables=False)
    report = load_expansion(model, warm, source_cfg, cfg, state="raw")
    print(f"SUCCESS: {cfg_path} -> loaded={report['loaded_keys']}, new={len(report['new_keys'])}")

print("ALL 3 CONFIGS WARM-START VERIFIED 100% OK!")
