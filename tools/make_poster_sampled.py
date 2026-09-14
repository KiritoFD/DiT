# -*- coding: utf-8 -*-
"""
make_poster_sampled.py — 可读版 poster: 每行只放 N_COLS 张(均匀抽样), 行数也受控.

背景: eval_samples_ctrl 里 strict 有 237 张 / seen 10 张, 且 step 有几十个 ->
直接用 render_poster 会得到"几千像素宽 × 几十行"的巨图, 没法看。

本脚本:
  列 = 从该 set 的 N_total 张里**均匀抽 N_COLS 张** (所有 step 用同一组 index, 便于纵向比较)
  行 = [第1行: 标准字输入 std] + [中间: 均匀选取的 <=MAX_ROWS 个 step 的生成] + [最后1行: GT]
输出 <stage>/<run>/posters/{set}_poster10.png

纯 CPU, 不占 GPU。用法:
  python tools/make_poster_sampled.py --runs v10b_stdskel_fame3_c41x --sets seen,strict
"""
import argparse
import glob
import os
import sys

from PIL import Image, ImageDraw

ROOT = "/root/Workspace/xy/DiT"
sys.path.insert(0, ROOT)
os.chdir(ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

N_COLS = 10
MAX_ROWS = 8
CELL = 160
GAP = 4
LABEL_W = 150


def pick_steps(step_dirs, max_rows):
    """均匀选 <=max_rows 个 step (含最早的与最新的)。"""
    ds = sorted(step_dirs, key=lambda p: int(os.path.basename(p)[4:]))
    if len(ds) <= max_rows:
        return ds
    idx = [round(i * (len(ds) - 1) / (max_rows - 1)) for i in range(max_rows)]
    return [ds[i] for i in sorted(set(idx))]


def stage(run):
    """把 run 级 + 实例级的 eval_samples_ctrl 合并软链到 staging。"""
    stage_dir = f"/tmp/poster_sampled/{run}/eval_samples_ctrl"
    os.makedirs(stage_dir, exist_ok=True)
    srcs = [f"assets/results/{run}/eval_samples_ctrl"]
    srcs += sorted(glob.glob(f"assets/results/{run}/*/eval_samples_ctrl"))
    for sc in [s for s in srcs if os.path.isdir(s)]:
        for sd in sorted(glob.glob(f"{sc}/step*")):
            # 只软链"至少含一张可用图"的 step
            dst = os.path.join(stage_dir, os.path.basename(sd))
            if os.path.exists(dst) or os.path.islink(dst):
                continue
            os.symlink(os.path.abspath(sd), dst)
        for ig in sorted(glob.glob(f"{sc}/*_input_g")):
            dst = os.path.join(stage_dir, os.path.basename(ig))
            if os.path.exists(dst) or os.path.islink(dst):
                continue
            os.symlink(os.path.abspath(ig), dst)
    return stage_dir


def one_set(run, stage_dir, set_name, sub):
    # 收集所有含该 set 图的 step
    step_dirs = []
    for sd in sorted(glob.glob(f"{stage_dir}/step*")):
        if glob.glob(os.path.join(sd, sub, "g0.png")):
            step_dirs.append(sd)
    if not step_dirs:
        print(f"  [{set_name}] no data", flush=True)
        return None
    # 集大小 (以最后一步为准, 各 step 应一致)
    last = step_dirs[-1]
    n_total = 0
    while os.path.exists(os.path.join(last, sub, f"g{n_total}.png")):
        n_total += 1
    cols = ([round(i * (n_total - 1) / (N_COLS - 1)) for i in range(N_COLS)]
            if n_total > 1 else [0])
    cols = sorted(set(cols))
    rows = pick_steps(step_dirs, MAX_ROWS)

    in_dir = os.path.join(stage_dir, f"{set_name}_input_g")
    has_in = os.path.isdir(in_dir) and bool(glob.glob(os.path.join(in_dir, "g0.png")))

    n_row = len(rows) + 2          # std + steps + gt
    W = LABEL_W + len(cols) * (CELL + GAP)
    H = n_row * (CELL + GAP)
    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(canvas)

    def put(r, c, im):
        canvas.paste(im, (LABEL_W + c * (CELL + GAP), r * (CELL + GAP)))

    def cell(path, bg=(240, 240, 240)):
        if path and os.path.exists(path):
            return Image.open(path).convert("RGB").resize((CELL, CELL), Image.LANCZOS)
        return Image.new("RGB", (CELL, CELL), bg)

    r = 0
    if has_in:
        for c, i in enumerate(cols):
            put(r, c, cell(os.path.join(in_dir, f"g{i}.png")))
        d.text((4, r * (CELL + GAP) + CELL // 2), "STD (input)", fill=(0, 0, 0))
        r += 1
    for sd in rows:
        st = int(os.path.basename(sd)[4:])
        for c, i in enumerate(cols):
            put(r, c, cell(os.path.join(sd, sub, f"g{i}.png")))
        d.text((4, r * (CELL + GAP) + CELL // 2), f"step {st//1000}k", fill=(0, 0, 0))
        r += 1
    for c, i in enumerate(cols):
        put(r, c, cell(os.path.join(last, sub, f"gt{i}.png"), (255, 230, 230)))
    d.text((4, r * (CELL + GAP) + CELL // 2), "GT", fill=(180, 0, 0))
    r += 1

    out = f"/tmp/poster_sampled/{run}/posters/{set_name}_poster10.png"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    canvas.save(out)
    print(f"  [{set_name}] n_total={n_total} cols={len(cols)} steps={len(rows)} -> {out}",
          flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="逗号分隔的 run 名")
    ap.add_argument("--sets", default="seen,strict",
                    help="逗号分隔; seen 用 g/, strict 用 strict/, strict50 -> strict50:strict50")
    args = ap.parse_args()
    spec = []
    for s in args.sets.split(","):
        if ":" in s:
            name, sub = s.split(":", 1)
        else:
            name, sub = s, ("g" if s == "seen" else s)
        spec.append((name, sub))
    for run in args.runs.split(","):
        print(f"== {run}", flush=True)
        stage_dir = stage(run)
        for name, sub in spec:
            one_set(run, stage_dir, name, sub)


if __name__ == "__main__":
    main()
