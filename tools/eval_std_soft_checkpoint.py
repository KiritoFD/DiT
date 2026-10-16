"""Strict std-soft checkpoint evaluation, independent of a live training process.

Use a new output directory: historical v83-A inline metrics had missing g.
The default is the intended ODE baseline (gamma=0), not the training CLI default.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eval.in_mem_eval import _get_cache, run_in_mem_eval
from src.eval.model_io import load_model_from_ckpt


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_eval_args(args, gamma, steps, cfg, batch, sets=None):
    if not bool(getattr(args, "std_soft_cond", False)):
        raise ValueError("This evaluator requires a std_soft_cond checkpoint")
    if gamma < 0 or steps <= 0 or batch <= 0 or cfg < 0:
        raise ValueError("gamma/cfg must be nonnegative; steps/batch must be positive")
    # ★ 2026-10-16: C2OT 已删除（实验证伪 ✓ 用户指示直接删 ✓）⇒ 判据去掉 `use_c2ot` ✓
    #   ★★ **但守卫本身必须保留** ✗ —— `use_ot`（朴素全局 OT ✓）**同样**会置换 x0↔ε 配对 ✗
    #     ⇒ "corrected score SDE 要求独立高斯配对" 的理由对 `use_ot` **依然成立** ✓
    #     ⇒ 只改判据 ✓ **不删守卫** ✗（删了会让 SDE+OT 静默给出错误结果 ✗✗）
    if gamma > 0 and getattr(args, "use_ot", False):
        raise ValueError("Corrected score SDE requires independent Gaussian pairing; use gamma=0 for OT checkpoints")
    args.sde_gamma = gamma
    args.flow_sampler = "heun"
    args.eval_steps = steps
    args.eval_cfg = cfg
    args.in_mem_eval_batch = batch
    args.in_mem_eval_vae_batch = batch
    args.in_mem_eval_save_samples = True
    args.in_mem_eval_lpips = True
    if sets is not None:
        args.in_mem_eval_sets = sets
    return args


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--gamma", type=float, default=0.0)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--cfg", type=float, default=1.0)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--sets", default=None, help="Default: checkpoint's recorded eval sets")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--vae-path", default=None, help="Explicit local image VAE; never silently fall back")
    parser.add_argument("--preflight-only", action="store_true")
    cli = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    # Artifact isolation prevents (step,set) dedup from reusing invalid old scores.
    out = Path(cli.out)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Use a new empty --out directory; do not mix evaluation protocols")
    model, args = load_model_from_ckpt(
        cli.ckpt, device="cpu" if cli.preflight_only else cli.device, use_ema=True)
    prepare_eval_args(args, cli.gamma, cli.steps, cli.cfg, cli.batch, cli.sets)
    vae_path = cli.vae_path or getattr(args, "eval_vae_path", "") or getattr(args, "vae_path", "")
    if not vae_path or not (Path(vae_path) / "config.json").is_file():
        raise ValueError("Provide --vae-path pointing to the actual local image VAE")
    args.eval_vae_path = str(Path(vae_path).resolve())
    specs = [s.split(":") for s in str(getattr(args, "in_mem_eval_sets", "")).split(",") if s.strip()]
    if not specs:
        raise ValueError("No evaluation sets configured")
    caches = {}
    for name, csv_path, n in specs:
        cache = _get_cache(csv_path, int(n), None, "", args)
        caches[name] = dict(n=cache["n"], g_kind=cache["g_kind"],
                            g_shape=list(cache["skels_latent"].shape), csv=csv_path,
                            csv_sha256=sha256(csv_path))
        print(f"[preflight] {name}: {caches[name]}", flush=True)
    if cli.preflight_only:
        print("PREFLIGHT_OK strict EMA + std-soft coverage/shape/range/nonblank", flush=True)
        return
    out.mkdir(parents=True, exist_ok=True)
    manifest = dict(checkpoint=str(Path(cli.ckpt).resolve()), checkpoint_sha256=sha256(cli.ckpt),
                    weights="ema", gamma=cli.gamma, cfg=cli.cfg, steps=cli.steps,
                    sampler=args.flow_sampler, batch=cli.batch, seed=0, caches=caches,
                    vae_path=args.eval_vae_path, vae_config_sha256=sha256(Path(vae_path) / "config.json"),
                    vae_scaling_factor=getattr(args, "vae_scaling_factor", None),
                    vae_shift_factor=getattr(args, "vae_shift_factor", 0.0),
                    torch_version=torch.__version__,
                    source_sha256={str(p): sha256(ROOT / p) for p in (
                        "src/model/dit.py", "src/eval/inference.py", "src/eval/in_mem_eval.py",
                        "src/eval/model_io.py", "src/loss/flow_matching.py", "tools/eval_std_soft_checkpoint.py")})
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    step = int(Path(cli.ckpt).stem)
    audit = dict(calls=0, first_inputs=[])

    def record_encoder_input(module, inputs):
        audit["calls"] += 1
        if len(audit["first_inputs"]) < 2:
            g = inputs[0]
            audit["first_inputs"].append(dict(shape=list(g.shape), mean=float(g.float().mean())))

    hook = model.glyph_embedder.register_forward_pre_hook(record_encoder_input)
    try:
        result = run_in_mem_eval(model, args, step, torch.device(cli.device), str(out), logger=print)
    finally:
        hook.remove()
    (out / "encoder_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    if not any(v["mean"] > 0 for v in audit["first_inputs"]):
        raise RuntimeError("Nonempty std-soft never reached the actual glyph encoder")
    if set(result) != set(caches) or any(v.get("lpips") is None for v in result.values()):
        raise RuntimeError("Incomplete evaluation or unavailable LPIPS; results are not accepted")
    (out / "metrics.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print("EVAL_COMPLETE " + json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
