"""Strict TABLE-conditioned checkpoint evaluation — the char-table counterpart of
``tools/eval_std_soft_checkpoint.py``.

Why this file exists
--------------------
v68 (char/callig/script embedding tables, **no glyph branch**) and v83-A (same tables
**plus** a std-soft glyph branch) must be compared **under one ruler**. The historical
in-run numbers are unusable for that: the in-run evaluation silently dropped the glyph
condition for std-soft runs (``g=None``), which understated ink by ~2.5x
(v83-A 25k: in-run ink_iou 0.09-0.11 vs corrected 0.2360).

For a table-only run ``g=None`` is **correct** (that is how it was trained, so there is
no condition to drop) — hence this evaluator deliberately does not require std-soft, and
it asserts that no std-soft path is active instead.

Protocol is pinned to match ``eval_std_soft_checkpoint.py`` exactly:
ODE (gamma=0), heun sampler, 50 steps, cfg=1.0, EMA weights, same eval sets, LPIPS on.
Anything else would make the two families incomparable.
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--gamma", type=float, default=0.0)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--cfg", type=float, default=1.0)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--sets", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--vae-path", required=True, help="Explicit local image VAE")
    parser.add_argument("--preflight-only", action="store_true")
    cli = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)

    out = Path(cli.out)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Use a new empty --out directory; do not mix evaluation protocols")

    model, args = load_model_from_ckpt(
        cli.ckpt, device="cpu" if cli.preflight_only else cli.device, use_ema=True)

    # ── 分支判定（写死、不猜）: 这个评测器只服务"表条件"那条路 ──
    use_glyph = bool(getattr(model, "use_glyph_cond", False)) or bool(getattr(model, "std_soft_cond", False))
    if use_glyph:
        raise ValueError(
            "该 ckpt 带字形分支 -> 属于 **std 分支**，请用 tools/eval_std_soft_checkpoint.py"
            "（或统一入口 tools/eval_checkpoint.py，它会自动判分支）。"
            "用表分支去评 std run 会静默把条件拔掉 —— 正是 2026-10-12 白化误判的成因 ✗")
    if not use_glyph:
        for _n in ("glyph_embedder",):
            if hasattr(model, _n):
                raise RuntimeError("Table-only evaluation expected, but %s exists" % _n)

    # ── 协议钉死（与 std-soft 评测器逐项一致）──
    if gamma_bad := (cli.gamma < 0 or cli.steps <= 0 or cli.batch <= 0 or cli.cfg < 0):
        raise ValueError("gamma/cfg must be nonnegative; steps/batch must be positive")
    args.sde_gamma = cli.gamma
    args.flow_sampler = "heun"
    args.eval_steps = cli.steps
    args.eval_cfg = cli.cfg
    args.in_mem_eval_batch = cli.batch
    args.in_mem_eval_vae_batch = cli.batch
    args.in_mem_eval_save_samples = True
    args.in_mem_eval_lpips = True
    if cli.sets is not None:
        args.in_mem_eval_sets = cli.sets

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
        g_kind = cache.get("g_kind")
        if str(g_kind).strip() not in ("", "none", "None"):
            raise RuntimeError("[%s] expected g=None for a table-only run, got g_kind=%r" % (name, g_kind))
        caches[name] = dict(n=cache["n"], g_kind=g_kind, csv=csv_path, csv_sha256=sha256(csv_path))
        print("[preflight] %s: %s" % (name, caches[name]), flush=True)
    if cli.preflight_only:
        print("PREFLIGHT_OK table-only strict EMA + g=None + sets covered", flush=True)
        return

    out.mkdir(parents=True, exist_ok=True)
    manifest = dict(checkpoint=str(Path(cli.ckpt).resolve()), checkpoint_sha256=sha256(cli.ckpt),
                    weights="ema", condition="table-only (g=None)",
                    gamma=cli.gamma, cfg=cli.cfg, steps=cli.steps, sampler=args.flow_sampler,
                    batch=cli.batch, seed=0, caches=caches,
                    vae_path=args.eval_vae_path, vae_config_sha256=sha256(Path(vae_path) / "config.json"),
                    vae_scaling_factor=getattr(args, "vae_scaling_factor", None),
                    vae_shift_factor=getattr(args, "vae_shift_factor", 0.0),
                    torch_version=torch.__version__,
                    source_sha256={str(p): sha256(ROOT / p) for p in (
                        "src/model/dit.py", "src/eval/inference.py", "src/eval/in_mem_eval.py",
                        "src/eval/model_io.py", "src/loss/flow_matching.py",
                        "tools/eval_table_checkpoint.py")})
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    step = int(Path(cli.ckpt).stem)
    result = run_in_mem_eval(model, args, step, torch.device(cli.device), str(out), logger=print)
    if set(result) != set(caches) or any(v.get("lpips") is None for v in result.values()):
        raise RuntimeError("Incomplete evaluation or unavailable LPIPS; results are not accepted")
    (out / "metrics.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print("EVAL_COMPLETE " + json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
