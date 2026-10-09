#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""prepare_data.py — 为消融臂准备数据面（图像 + CSV + REPA 缓存 + latent shards）。

幂等：每步都可在中断后重跑（图像/ shard 级跳过）。

步骤
----
  A. noaug CSV：把 sym3x 的 CSV 过滤出 aug=="" 的行 -> exp-std/csv/train_top10_noaug.csv
     （shards 与 DINO cache 直接复用 sym3x 的现有产物，因其含原图行）
  B. 每个新增强策略 s：
       1) build_aug_variants.py   -> 图像 + train_top10_aug_<s>.csv + manifest
       2) expand_repa_cache.py    -> data/dino_cache/top10_aug_<s>_v1
       3) encode_aug_latents.py   -> exp-std/data/shards_img_aug_<s>(_calli)
  C. --reencode-baseline：把 sym3x（以及 noaug 复用同一个）的 calli shards 按修正约定重编。
     ⚠ 48 上现有的 exp-std/data/shards_img_aug_calli 是旧 `mode()` 约定（B2 bug）产出的，
       用修正约定训练前**必须**重编；--latent calli 时强烈建议打开。

用法
----
  python tools/ablation/prepare_data.py --latent calli --strategies thick,thin --reencode-baseline
  python tools/ablation/prepare_data.py --dry-run
"""
import argparse
import csv
import os
import subprocess
import sys

NEW_DATA_STRATEGIES = ["thick", "thin", "symwide", "sym4"]


def run(cmd, cwd):
    print("\n[run] " + " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=cwd).returncode


def make_noaug_csv(root, src_csv, out_csv):
    src = os.path.join(root, src_csv)
    out = os.path.join(root, out_csv)
    rows = list(csv.DictReader(open(src, encoding="utf-8")))
    keep = [r for r in rows if (r.get("aug") or "").strip() == ""]
    fields = list(rows[0].keys())
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(keep)
    print(f"[noaug] {src_csv} ({len(rows):,}) -> {out_csv} ({len(keep):,} 原图行)")
    return out_csv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--latent", default="calli", choices=["calli", "sd"])
    ap.add_argument("--strategies", default=",".join(NEW_DATA_STRATEGIES))
    ap.add_argument("--vae", default=None,
                    help="编码用 VAE 目录。calli 默认 data/pretrained/pretrained_models/calli_vae；"
                         "sd 默认 pretrained_models/sd-vae-ft-ema")
    ap.add_argument("--encode-mode", default=None, choices=["mode", "sample"],
                    help="默认 calli->sample, sd->mode")
    ap.add_argument("--image-size", type=int, default=256)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--shard-size", type=int, default=5120)
    ap.add_argument("--reencode-baseline", action="store_true",
                    help="把 sym3x(noaug 复用它) 的 shards 按当前约定重编")
    ap.add_argument("--skip-encode", action="store_true",
                    help="只做 CPU 部分（图像/CSV/REPA），跳过 latent 编码（编码占 GPU，可另跑）")
    ap.add_argument("--noaug-src", default="exp-std/csv/train_top10_aug_sym.csv")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    root = os.path.abspath(a.root)

    if a.encode_mode is None:
        a.encode_mode = "sample" if a.latent == "calli" else "mode"
    if a.vae is None:
        a.vae = ("data/pretrained/pretrained_models/calli_vae" if a.latent == "calli"
                 else "pretrained_models/sd-vae-ft-ema")
    suffix = "_calli" if a.latent == "calli" else ""
    strategies = [s.strip() for s in a.strategies.split(",") if s.strip()]

    print("=" * 90)
    print(f"[prep] root={root} latent={a.latent} vae={a.vae} encode_mode={a.encode_mode}")
    print(f"[prep] strategies={strategies} reencode_baseline={a.reencode_baseline} "
          f"dry_run={a.dry_run}")
    print("=" * 90)

    plan = []
    # A. noaug CSV
    plan.append(("noaug-csv", lambda: make_noaug_csv(
        root, a.noaug_src, "exp-std/csv/train_top10_noaug.csv")))

    # B. 每个新策略
    for s in strategies:
        out_csv = f"exp-std/csv/train_top10_aug_{s}.csv"
        imgs_dir = f"data/top10_style23/imgs_aug_{s}"
        repa = f"data/dino_cache/top10_aug_{s}_v1"
        shards = f"exp-std/data/shards_img_aug_{s}{suffix}"

        def _build(s=s, out_csv=out_csv, imgs_dir=imgs_dir):
            return run([sys.executable, "tools/ablation/build_aug_variants.py",
                        "--strategy", s, "--root", root,
                        "--out-csv", out_csv, "--imgs-dir", imgs_dir,
                        "--workers", str(min(48, (os.cpu_count() or 8)))], root)

        def _repa(out_csv=out_csv, repa=repa):
            return run([sys.executable, "tools/ablation/expand_repa_cache.py",
                        "--src", "data/dino_cache/top10_v1", "--csv", out_csv,
                        "--src-csv", "exp-std/csv/train.csv", "--out", repa], root)

        def _encode(out_csv=out_csv, shards=shards):
            return run([sys.executable, "tools/encode_aug_latents.py",
                        "--csv", out_csv, "--out", shards, "--vae", a.vae,
                        "--img-root", ".", "--batch", str(a.batch),
                        "--workers", str(a.workers), "--shard-size", str(a.shard_size),
                        "--image-size", str(a.image_size),
                        "--encode-mode", a.encode_mode], root)

        plan += [(f"build[{s}]", _build), (f"repa[{s}]", _repa)]
        if not a.skip_encode:
            plan.append((f"encode[{s}]", _encode))

    # C. baseline 重编
    if a.reencode_baseline:
        base_csv = a.noaug_src
        shards = f"exp-std/data/shards_img_aug{suffix}"
        def _reb(shards=shards, base_csv=base_csv):
            # 先移除旧 shards (约定不同, 不能断点续编)
            d = os.path.join(root, shards)
            if os.path.isdir(d):
                import glob
                old = glob.glob(os.path.join(d, "shard_*.npz"))
                for p in old:
                    os.remove(p)
                print(f"[baseline] 删除旧 shards {len(old)} 个 (约定变更, 重编)")
            return run([sys.executable, "tools/encode_aug_latents.py",
                        "--csv", base_csv, "--out", shards, "--vae", a.vae,
                        "--img-root", ".", "--batch", str(a.batch),
                        "--workers", str(a.workers), "--shard-size", str(a.shard_size),
                        "--image-size", str(a.image_size),
                        "--encode-mode", a.encode_mode], root)
        plan.append((f"reencode-baseline{suffix}", _reb))

    if a.dry_run:
        print("[prep] DRY-RUN 计划:")
        for name, _ in plan:
            print("   -", name)
        return 0

    for name, fn in plan:
        print(f"\n{'='*90}\n[prep] STEP {name}\n{'='*90}", flush=True)
        rc = fn()
        if rc != 0:
            print(f"[prep] !! STEP {name} 失败 rc={rc}，中止。", flush=True)
            return rc
    print("\n[prep] 全部完成。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
