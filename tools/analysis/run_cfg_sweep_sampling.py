import os, sys, json, csv
from pathlib import Path
import torch
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.unet.io import load, eval_args, diffusion, Config
from src.unet.model import LatentUNet
from src.eval.in_mem_eval import run_in_mem_eval

def run_cfg_sweep(name, ckpt_path, cfg_scales, out_base, device="cuda", is_4090=False):
    if is_4090:
        # Load 4090 config
        conf_dict = json.loads(open(ROOT / "src/unet/configs/connectivity/DiC_U100_b768.json").read())
        config = Config(**conf_dict)
        model = LatentUNet(tuple(config.channels), config.num_writers, config.num_scripts,
                           num_res_blocks=2, residual_cond_gate=True,
                           stage_specific_cond=True, stage_cond_mode='adapter', act_fn='gelu').to(device)
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        weights = {k.removeprefix("_orig_mod."): v for k, v in ck["ema"].items()}
        model.load_state_dict(weights, strict=True)
        model.eval()
        step = 50000
    else:
        model, config, ck = load(str(ckpt_path), device, use_ema=False)
        step = int(ck.get('train_steps', 75000))
        del ck

    for cfg in cfg_scales:
        print(f"\n>>> Sampling {name} at CFG={cfg} ...", flush=True)
        eval_conf = eval_args(config)
        eval_conf.eval_cfg = float(cfg)
        eval_conf.eval_steps = 50
        eval_conf.in_mem_eval_batch = 32
        eval_conf.in_mem_eval_poster = False
        
        target_dir = Path(out_base) / f"{name}_cfg{cfg:.1f}"
        target_dir.mkdir(parents=True, exist_ok=True)
        
        with torch.no_grad():
            values = run_in_mem_eval(
                model, eval_conf, step, device,
                results_dir=str(target_dir),
                diffusion_override=diffusion(config)
            )
        print(f"[Done {name} CFG={cfg:.1f}]", flush=True)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--is-4090", action="store_true")
    parser.add_argument("--cfgs", nargs="+", type=float, default=[1.0, 1.2, 1.4, 1.6])
    parser.add_argument("--out", default="assets/results/cfg_sweep")
    args = parser.parse_args()
    
    run_cfg_sweep(args.model, args.ckpt, args.cfgs, args.out, is_4090=args.is_4090)
