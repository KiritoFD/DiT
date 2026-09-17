"""推理参数扫描（cfg × steps），用**固定 ckpt**，零训练成本，不干扰训练。

为什么单独写而不是复用现有工具：
  - `gpu_batch_eval_v2.py` 是旧 ControlNet 时代的（用 MCCDDataset、不传 g）-> 不适用
  - `auto_eval_gpu.build_model` 已过期（硬编码 learn_sigma=True，缺 glyph_inject_mode /
    norm_type / rope / glyph_vec_cond 等）-> 会在 strict=False 下**静默加载错架构**
  所以这里**照 src/train/train.py 的构造原样复制**，并用 `load_state_dict(strict=True)`
  当护栏 —— 任何字段不一致会直接抛错，而不是静默用随机权重。

用法:
  python tools/cfg_sweep.py --ckpt <path.pt> --tag v12_S2_100k \
      --cfgs 0.0,0.3,0.5,0.7,1.0,1.3,1.7,2.2 --steps 50
"""
import argparse, csv, json, os, sys, time

import torch

ROOT = "/root/Workspace/xy/DiT"
os.chdir(ROOT)
sys.path.insert(0, ROOT)

DEFAULT_SETS = "seen:assets/eval_seen_v10.csv:10,strict:assets/eval_fame3_strict_clean_v9.csv:50"


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def build_model_from_args(a, device):
    """已统一到 src/eval/model_io.py（2026-09-17）。

    原来这里有一份**复刻 train.py 构造**的独立实现，是仓库里唯一 strict=True 的。
    现已提升为公共模块 src/eval/model_io.build_model_from_args，
    让 8 处 strict=False 的实现都能切过来 —— 见 docs/system/70 §1.1。
    本函数保留为薄封装，避免改动已有调用点。
    """
    from src.eval.model_io import build_model_from_args as _impl
    return _impl(a, device)



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--cfgs", default="0.0,0.3,0.5,0.7,1.0,1.3,1.7,2.2")
    ap.add_argument("--steps", default="50")
    ap.add_argument("--sets", default=DEFAULT_SETS,
                    help="'name:csv:n' 逗号分隔; 传 'skip' 则用 ckpt args 里的设置")
    ap.add_argument("--out-root", default="assets/results/_sweep")
    ap.add_argument("--ema", default="1", help="1=用 ema 权重, 0=用 model 权重")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    from src.eval import in_mem_eval as im

    from src.eval.model_io import load_model_from_ckpt

    model, a = load_model_from_ckpt(args.ckpt, device=args.device,
                                   use_ema=(str(args.ema) == "1"), verbose=False)
    log(f"model loaded strict=True OK (params={sum(p.numel() for p in model.parameters()):,})")
    if model.cfg_glyph_scale is not None:
        _gdp = float(getattr(a, "glyph_drop_prob", 0.0) or 0.0)
        log(f"双轴 CFG: cfg_glyph={model.cfg_glyph_scale}, w_inter={model.cfg_w_inter}, "
            f"glyph_drop_prob={_gdp}" + ("  ⚠ 内容轴未训练!" if _gdp <= 0 else ""))

    if args.sets != "skip":
        a.in_mem_eval_sets = args.sets
    a.in_mem_eval_save_samples = False
    a.eval_self_cond = False
    a.eval_blend_alpha = 0.0

    sets = []
    for spec in str(a.in_mem_eval_sets).split(","):
        if spec.strip():
            n, c, k = spec.strip().split(":")
            sets.append((n, c, int(k)))
    log(f"eval sets: {sets}")

    cfgs = [float(x) for x in args.cfgs.split(",") if x.strip()]
    steps_list = [int(x) for x in args.steps.split(",") if x.strip()]
    rows = []
    for st in steps_list:
        for cfg in cfgs:
            out = os.path.join(args.out_root, args.tag, f"cfg{cfg:g}_steps{st}")
            os.makedirs(out, exist_ok=True)
            a.eval_cfg = cfg
            a.eval_steps = st
            im._DIFF = None            # ★ 必须清: 模块级缓存按 steps 建, 不清会复用错的调度
            t0 = time.time()
            try:
                res = im.run_in_mem_eval(model, a, 0, args.device, results_dir=out,
                                         sets=sets, logger=lambda *x: None)
            except Exception as e:
                log(f"  cfg={cfg} steps={st}  FAILED: {type(e).__name__}: {e}")
                rows.append(dict(cfg=cfg, steps=st, failed=str(e)[:80]))
                continue
            r = dict(cfg=cfg, steps=st, dt=round(time.time() - t0, 1))
            for k, v in res.items():
                r[f"{k}_ssim"] = round(float(v), 4)
            # 取 LPIPS (in_mem_eval 已写 batch CSV)
            bp = os.path.join(out, "eval_stdskel_batch.csv")
            if os.path.exists(bp):
                for k in ("seen", "strict"):
                    vals = [float(x["lpips"]) for x in csv.DictReader(open(bp, encoding="utf-8"))
                            if x.get("set") == k and x.get("lpips")]
                    if vals:
                        r[f"{k}_lpips"] = round(sum(vals) / len(vals), 4)
            rows.append(r)
            log(f"  cfg={cfg:<4} steps={st:<4} " +
                "  ".join(f"{k}={r.get(k, '')}" for k in
                          ("seen_ssim", "strict_ssim", "strict_lpips", "dt")))

    out_csv = os.path.join(args.out_root, args.tag, "sweep.csv")
    keys = sorted({k for r in rows for k in r})
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    log(f"DONE -> {out_csv}")
    for r in rows:
        log("  " + json.dumps(r, ensure_ascii=False))


if __name__ == "__main__":
    main()
