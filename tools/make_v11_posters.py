# -*- coding: utf-8 -*-
"""
make_v11_posters.py — 用与 v10 相同的流程重拼 v11 系列的 poster (seen + strict)。

v11 的 eval 图同样散在多个重启实例目录, 但已有 `{set}_input_g`(之前生成过)
-> stage 时把 input_g 目录一并软链, 不重复解码。
产出 <stage>/<run>/posters/{seen,strict}_{poster,struct}.png, 并 copy 回各 run 主实例。
"""
import glob
import os
import shutil
import sys

ROOT = "/root/Workspace/xy/DiT"
sys.path.insert(0, ROOT)
os.chdir(ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

RUNS = [
    "v11_pretrain_S2_ref12ch",
    "v11_pretrain_S2_v8_aux02",
    "v11_pretrain_M432_v8_sk3",
    "v11_pretrain_M432_v8_auxc03s08",
    "v11_pretrain_M432_adaln4_sym",
    "v11_pretrain_M432_adaln4_sym_noise400k",
    "v11_pretrain_Sp2_base",
]
STAGE = "/tmp/v11_poster"
SETS = [("seen", "g"), ("strict", "strict")]

from src.eval.in_mem_eval import render_poster       # noqa: E402


def stage_run(run):
    stage = os.path.join(STAGE, run, "eval_samples_ctrl")
    os.makedirs(stage, exist_ok=True)
    n_new = 0
    # v11 有两层 eval_samples_ctrl: run 级 (常含 strict) + 实例级 (含 seen)
    srcs = [f"assets/results/{run}/eval_samples_ctrl"]
    srcs += sorted(glob.glob(f"assets/results/{run}/*/eval_samples_ctrl"))
    for sc in [s for s in srcs if os.path.isdir(s)]:
        # step 目录 + {set}_input_g 目录 都软链过去
        for sd in sorted(glob.glob(f"{sc}/step*")) + sorted(glob.glob(f"{sc}/*_input_g")):
            dst = os.path.join(stage, os.path.basename(sd))
            if os.path.exists(dst) or os.path.islink(dst):
                continue
            os.symlink(os.path.abspath(sd), dst)
            n_new += 1
    return stage, n_new


def main():
    for run in RUNS:
        print(f"== {run}", flush=True)
        stage, n_new = stage_run(run)
        dirs = sorted(d for d in os.listdir(stage) if d.startswith("step"))
        if not dirs:
            print("  no eval samples, skip", flush=True)
            continue
        print(f"  staged +{n_new}, steps={len(dirs)} ({dirs[0]}..{dirs[-1]})", flush=True)
        for name, sub in SETS:
            # 该 sub 可能在部分 step 才有 -> 扫全部 step, 不能只看最后一个
            has = any(glob.glob(os.path.join(stage, d, sub, "g0.png")) for d in dirs)
            if not has:
                print(f"  {name}: no images, skip", flush=True)
                continue
            p = render_poster(os.path.join(STAGE, run), name)
            print(f"  poster({name}): {p}", flush=True)
        # 只取 timestamp 实例目录 (排除已存在的 posters/ 等)
        insts = sorted(d for d in glob.glob(f"assets/results/{run}/*/")
                       if os.path.basename(os.path.normpath(d))[:2] == "20")
        main_inst = insts[-1] if insts else f"assets/results/{run}/"
        dst = os.path.join(main_inst, "posters")
        os.makedirs(dst, exist_ok=True)
        for f in glob.glob(os.path.join(STAGE, run, "posters", "*.png")):
            shutil.copy2(f, os.path.join(dst, os.path.basename(f)))
        print(f"  copied -> {dst}", flush=True)


if __name__ == "__main__":
    main()
