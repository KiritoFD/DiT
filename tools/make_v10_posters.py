# -*- coding: utf-8 -*-
"""
make_v10_posters.py — 给 v10 系列拼 poster (seen + strict)。

v10 的 eval 图特点:
  * 分散在**多个重启实例目录** (每次重启新建 timestamp 目录) -> 先软链合并
  * 集名不统一: seen 用 `g/`, strict 用 `strict/` (n=237) 或 `strict50/` (n=50)
  * 没有 `{set}_input_g` (poster 第 1 行标准字) -> 本脚本用 CPU VAE decode 补
    (input_g 只与 eval 集有关, 与 run 无关 -> 全 run 复用同一次解码结果)

产出: <stage>/<run>/posters/{seen,strict,strict50}_{poster,struct}.png
      并 copy 回各 run 的主 results 目录 posters/ (留远程一份)。
纯 CPU, 不占 GPU。
"""
import glob
import os
import shutil
import sys

import torch as th
from PIL import Image

ROOT = "/root/Workspace/xy/DiT"
sys.path.insert(0, ROOT)
os.chdir(ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

RUNS = [
    "v10b_stdskel_fame3_c41x",
    "v10b_stdskel_fame3_c41x_cos",
    "v10b_stdskel_fame3_c41x_cos_e",
]
STAGE = "/tmp/v10_poster"
SHARDS = "data/skel/std_skel1_latents_fame3_v8"
CMAP = "assets/callig_id_map.json"
SF = 0.18215

# (poster 集名, step 下 sub 目录名, eval csv, n)
SET_SPECS = [
    ("seen",    "g",        "assets/eval_seen_v10.csv",                 10),
    ("strict",  "strict",   "assets/eval_fame3_strict_clean_v9.csv",    237),
    ("strict50", "strict50", "assets/eval_fame3_strict_clean_v9.csv",   50),
]

from src.eval.in_mem_eval import render_poster                      # noqa: E402
from src.eval.inference import make_eval_cache, load_eval_vae       # noqa: E402
from src.utils.callig_map import load_callig_id_map                 # noqa: E402

_INPUT_G_CACHE = {}          # (set,csv,n) -> {i: PIL.Image} 复用解码结果


def stage_run(run):
    srcs = sorted(glob.glob(f"assets/results/{run}/*/eval_samples_ctrl"))
    stage = os.path.join(STAGE, run, "eval_samples_ctrl")
    os.makedirs(stage, exist_ok=True)
    n_new = 0
    for sc in srcs:
        for sd in sorted(glob.glob(f"{sc}/step*")):
            dst = os.path.join(stage, os.path.basename(sd))
            if os.path.exists(dst) or os.path.islink(dst):
                continue
            os.symlink(os.path.abspath(sd), dst)
            n_new += 1
    return stage, n_new


def detect_subs(run):
    """列出该 run 实际存在且非空的 sub 目录名。"""
    d = os.path.join(STAGE, run, "eval_samples_ctrl")
    found = set()
    for sd in glob.glob(f"{d}/step*"):
        for x in os.listdir(sd):
            p = os.path.join(sd, x)
            if os.path.isdir(p) and not x.endswith("input_g"):
                if glob.glob(os.path.join(p, "g0.png")):
                    found.add(x)
    return found


def input_g_images(spec):
    """CPU decode 标准字形 latent -> {i: PIL.Image} (按 spec 缓存, 全 run 复用)。"""
    key = (spec[0], spec[2], spec[3])
    if key in _INPUT_G_CACHE:
        return _INPUT_G_CACHE[key]
    cmap = None
    if os.path.exists(CMAP):
        cmap, _ = load_callig_id_map(CMAP)
    n = spec[3]
    cache = make_eval_cache(spec[2], None, None, 256, n, 8, 4, SF,
                            skel_latent_shards_dir=SHARDS, callig_id_map=cmap)
    sk = cache.get("skels_latent")
    if sk is None:
        _INPUT_G_CACHE[key] = {}
        return {}
    vae = load_eval_vae(th.device("cpu"), "data/pretrained/sd-vae-ft-ema")
    with th.no_grad():
        dec = vae.decode(sk.float() / SF).sample
    arr = ((dec.clamp(-1, 1) + 1) / 2 * 255).byte().numpy()
    out = {i: Image.fromarray(arr[i].transpose(1, 2, 0)) for i in range(arr.shape[0])}
    _INPUT_G_CACHE[key] = out
    print(f"    input_g decoded for {spec[0]} (n={len(out)})", flush=True)
    return out


def write_input_g(run, spec):
    out_dir = os.path.join(STAGE, run, "eval_samples_ctrl", f"{spec[0]}_input_g")
    os.makedirs(out_dir, exist_ok=True)
    imgs = input_g_images(spec)
    n = 0
    for i, im in imgs.items():
        p = os.path.join(out_dir, f"g{i}.png")
        if not os.path.exists(p):
            im.save(p)
            n += 1
    if n:
        print(f"    input_g written {n}", flush=True)
    return len(imgs)


def main():
    for run in RUNS:
        print(f"== {run}", flush=True)
        stage, n_new = stage_run(run)
        subs = detect_subs(run)
        print(f"  staged +{n_new}; subs found: {sorted(subs)}", flush=True)
        if not subs:
            continue
        for spec in SET_SPECS:
            if spec[1] not in subs:
                continue
            write_input_g(run, spec)
            p = render_poster(os.path.join(STAGE, run), spec[0])
            print(f"  poster({spec[0]}): {p}", flush=True)
        # copy 回该 run 的最新实例目录 (远程留档)
        main_inst = sorted(glob.glob(f"assets/results/{run}/*/"))[-1]
        dst = os.path.join(main_inst, "posters")
        os.makedirs(dst, exist_ok=True)
        for f in glob.glob(os.path.join(STAGE, run, "posters", "*.png")):
            shutil.copy2(f, os.path.join(dst, os.path.basename(f)))
        print(f"  copied -> {dst}", flush=True)


if __name__ == "__main__":
    main()
