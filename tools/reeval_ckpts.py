"""用统一口径重评 ckpt(单个/一批), 结果汇总到 exp-std/eval_master.csv。

口径 = 训练内 in_mem_eval 完全一致: cfg 0.7 / 50 步 / heun / flow+logit_normal。
条件 (由 set 名自动路由, 见 in_mem_eval.py:849-866):
  eval200 -> eval_skel_latent_shards_dir = 修正后的 shards_std_w7_fixed_eval200
  seen    -> skel_latent_shards_dir      = 训练原 shards_std_w7
用法:
  python tools/reeval_ckpts.py --family                # 先列"可比"的 ckpt
  python tools/reeval_ckpts.py --ckpt <path> [--ckpt ...]
  python tools/reeval_ckpts.py --ckpt-list ckpts.txt
"""
import argparse
import csv
import glob
import json
import os
import sys
import time
import traceback

ROOT = "/root/Workspace/xy/DiT"
os.chdir(ROOT)
sys.path.insert(0, ROOT)

FIXED_SHARD = "exp-std/data/shards_std_w7_fixed_eval200"
TRAIN_SHARD = "exp-std/data/shards_std_w7"
EVAL_SET = ("eval200:exp-std/csv/eval200_fixed.csv:200,"
            "seen:exp-std/csv/seen20.csv:20")
MASTER = "exp-std/eval_master.csv"
MASTER_COLS = ["tag", "run", "ckpt", "step", "set", "n", "ssim_mean", "ssim_med",
               "lpips_mean", "ink_ssim_mean", "ink_iou_mean", "skel_iou_mean",
               "mse_mean", "frag_ratio", "sec"]


def list_family(roots=("assets/results", "exp", "_archive/20261003_twostage"),
                n_callig=23):
    """按 resolved_config.json 找"与 eval200 同桌子"的 run (23 槽位 + 骨架当字形条件)。"""
    out = []
    for r in roots:
        for cp in glob.glob(os.path.join(r, "**", "resolved_config.json"),
                            recursive=True):
            try:
                c = json.load(open(cp, encoding="utf-8"))
            except Exception:                                  # noqa: BLE001
                continue
            if int(c.get("num_calligraphers", 0) or 0) != n_callig:
                continue
            if not c.get("skel_as_glyph_cond"):
                continue
            run = os.path.dirname(cp)
            cks = sorted(glob.glob(os.path.join(run, "checkpoints", "*.pt")))
            for x in cks:
                out.append((run, x, c.get("glyph_inject_mode", "adalna"),
                            os.path.basename(str(c.get("data_csv"))),
                            os.path.basename(str(c.get("skel_latent_shards_dir")))))
    return out


def append_master(out_dir, ckpt, tag, run):
    sp = os.path.join(out_dir, "eval_stdskel_summary.csv")
    if not os.path.exists(sp):
        return 0
    new = not os.path.exists(MASTER)
    rows = list(csv.DictReader(open(sp, encoding="utf-8")))
    with open(MASTER, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=MASTER_COLS)
        if new:
            w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in MASTER_COLS} |
                       {"tag": tag, "run": run, "ckpt": ckpt})
    return len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", default=[])
    ap.add_argument("--ckpt-list", default="")
    ap.add_argument("--sets", default=EVAL_SET)
    ap.add_argument("--out-dir", default="exp-std/reeval")
    ap.add_argument("--tag", default="")
    ap.add_argument("--family", action="store_true", help="只列出家族内 ckpt, 不评")
    ap.add_argument("--all-family", action="store_true", help="评家族内全部 ckpt")
    ap.add_argument("--max", type=int, default=0, help="最多评几个")
    a = ap.parse_args()

    cks = list(a.ckpt)
    if a.ckpt_list and os.path.exists(a.ckpt_list):
        cks += [x.strip() for x in open(a.ckpt_list, encoding="utf-8") if x.strip()]

    if a.family or a.all_family:
        fam = list_family()
        print(f"[family] 23 槽位 + 骨架条件 的 ckpt 共 {len(fam)} 个")
        byrun = {}
        for run, ck, inj, data, sk in fam:
            byrun.setdefault(run, []).append((ck, inj, data, sk))
        for run, xs in sorted(byrun.items()):
            ck0 = xs[0]
            print(f"  {run}\n      n={len(xs)}  inj={ck0[1]}  data={ck0[2]}  skel={ck0[3]}"
                  f"  steps={os.path.basename(xs[0][0])[:-3]}..{os.path.basename(xs[-1][0])[:-3]}")
        if a.family and not a.all_family:
            return
        cks += [x[0] for x in fam]

    if not cks:
        print("[warn] 没有要评的 ckpt")
        return

    import torch as th
    from src.eval.model_io import load_model_from_ckpt
    from src.eval.in_mem_eval import run_in_mem_eval

    dev = th.device("cuda" if th.cuda.is_available() else "cpu")
    sets = [(s.split(":")[0], s.split(":")[1], int(s.split(":")[2]))
            for s in a.sets.split(",") if s.strip()]
    print(f"[sets] {sets}")
    print(f"[cond] eval200 -> {FIXED_SHARD}\n        seen    -> {TRAIN_SHARD}")

    if a.max:
        cks = cks[:a.max]
    ok = fail = 0
    for i, ck in enumerate(cks, 1):
        if not os.path.exists(ck):
            print(f"[{i}/{len(cks)}] 缺失 {ck}"); fail += 1; continue
        run = os.path.basename(os.path.dirname(os.path.dirname(ck.rstrip("/"))))
        try:
            step = int(os.path.basename(ck)[:-3])
        except ValueError:
            step = 0
        out_dir = os.path.join(a.out_dir, f"{run}__{step:07d}")
        os.makedirs(out_dir, exist_ok=True)
        print(f"\n[{i}/{len(cks)}] {run} step={step}")
        t0 = time.time()
        try:
            model, args = load_model_from_ckpt(ck, device=dev, use_ema=True,
                                               verbose=False)
            args.eval_skel_latent_shards_dir = FIXED_SHARD
            args.skel_latent_shards_dir = TRAIN_SHARD
            res = run_in_mem_eval(model, args, step, dev, out_dir,
                                  sets=sets, logger=print)
            n = append_master(out_dir, ck, a.tag or run, run)
            print(f"    -> {res}  ({time.time() - t0:.1f}s, 写入 {n} 行)")
            ok += 1
            del model
            th.cuda.empty_cache()
        except Exception as e:                                  # noqa: BLE001
            print(f"    [FAIL] {type(e).__name__}: {e}")
            traceback.print_exc(limit=2)
            fail += 1
            with open("exp-std/reeval_failures.txt", "a", encoding="utf-8") as f:
                f.write(f"{ck}\t{type(e).__name__}: {e}\n")

    print(f"\n[done] ok={ok} fail={fail}  汇总 -> {MASTER}")


if __name__ == "__main__":
    main()
