# -*- coding: utf-8 -*-
"""Repo audit: inventory, duplicates (md5), import map, entrypoints."""
import os, sys, re, hashlib
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKIP = {"__pycache__", ".git", "docs", "5script", "pretrained_models", "node_modules"}
STDLIB = set(sys.stdlib_module_names) if hasattr(sys, "stdlib_module_names") else set()


def walk_py(base):
    out = []
    for dp, dns, fns in os.walk(base):
        dns[:] = [d for d in dns if d not in SKIP and not d.endswith("__pycache__")]
        for fn in fns:
            if fn.endswith(".py"):
                out.append(os.path.join(dp, fn))
    return out


def main():
    all_py = walk_py(ROOT)
    rels = [os.path.relpath(p, ROOT) for p in all_py]
    print(f"TOTAL .py: {len(rels)}")
    groups = defaultdict(list)
    for r in rels:
        groups[r.split(os.sep)[0]].append(r)
    for k in sorted(groups):
        print(f"  [{k}] {len(groups[k])}")

    # duplicates
    print("\n=== DUPLICATES (md5) ===")
    hashes = defaultdict(list)
    for r in rels:
        h = hashlib.md5(open(os.path.join(ROOT, r), "rb").read()).hexdigest()
        hashes[h].append(r)
    for h, v in hashes.items():
        if len(v) > 1:
            print(f"  {len(v)}x: {v}")

    # imports of the core modules
    core = ["models", "diffusion", "latent_dataset", "losses", "samplers",
            "glyph_latent", "lora", "train", "controlnet_dit", "in_process_eval",
            "in_process_ctrl_eval", "eval_ctrl_metrics_daemon", "auto_eval_ctrl_flow"]
    print("\n=== WHO IMPORTS CORE ===")
    for r in rels:
        src = open(os.path.join(ROOT, r), encoding="utf-8", errors="ignore").read()
        for c in core:
            pat = rf"(?:from {c} import|import {c}\b|from {c}\.)"
            if re.search(pat, src):
                print(f"  {r} -> {c}")

    # sys.path hacks
    print("\n=== FILES WITH sys.path.insert ===")
    for r in rels:
        src = open(os.path.join(ROOT, r), encoding="utf-8", errors="ignore").read()
        if "sys.path.insert" in src:
            print(f"  {r}")

    # main entrypoints with argparse at root/src/tools top level
    print("\n=== ENTRYPOINTS (has __main__ + parse_args) ===")
    for r in sorted(rels):
        if r.startswith(("_", "tests")) or os.sep in r and r.split(os.sep)[0] in ("docs",):
            continue
        src = open(os.path.join(ROOT, r), encoding="utf-8", errors="ignore").read()
        if "__main__" in src and ("parse_args" in src or "ArgumentParser" in src):
            print(f"  {r}")


if __name__ == "__main__":
    main()