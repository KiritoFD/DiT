# -*- coding: utf-8 -*-
"""collect_v89_series.py — 收集 v8/v9 系列 eval_auto json → assets/v89_eval_series.json.

在远程跑: /opt/conda/envs/cu121/bin/python tools/eval/collect_v89_series.py
每个系列取最新 run 目录, 兼容两种 json 格式:
  * ctrl 嵌套: {"ctrl": {ssim_mean...}, "base": {...}}  (train_controlnet daemon)
  * pretrain 平铺: {"ssim": ..., "lpips": ...}          (train.py daemon)
输出: [{"series": ..., "run_dir": ..., "steps": {step: {ssim, base_ssim, delta_ssim, skel_iou, mse, lpips, n}}}]
"""
import os, sys, json, glob

ROOT = "/root/Workspace/xy/DiT"
os.chdir(ROOT)
sys.stdout.reconfigure(encoding="utf-8")

SERIES = [
    ("v8b_ctrl_1px", "assets/results/v8_3stage/v8b"),
    ("v9a_repa_pretrain", "assets/results/v9a_repa_pretrain"),
    ("v9b_ctrl_repa_strong", "assets/results/v9b_ctrl_strong"),
    ("v9c_skel_joint", "assets/results/v9c_skel_joint"),
    ("v10a_skel_cond_pretrain", "assets/results/v10a_skel_cond_pretrain"),
    ("v10b_skel_only_pretrain", "assets/results/v10b_skel_only_pretrain"),
]


def latest_run_dir(base):
    ds = sorted(glob.glob(os.path.join(base, "*/checkpoints", )))
    return ds[-1].rstrip("/").replace("/checkpoints", "") if ds else None


def parse_step(fname):
    stem = os.path.basename(fname).replace("eval_auto_", "").replace(".json", "")
    digits = "".join(ch for ch in stem if ch.isdigit())
    return int(digits) if digits else None


def extract(d):
    def g(*keys):
        for k in keys:
            if k in d:
                return d[k]
        return None
    if isinstance(d.get("ctrl"), dict):
        c, b = d["ctrl"], (d.get("base") or {})
        def cg(*keys):
            for k in keys:
                if k in c:
                    return c[k]
            return None
        def bg(*keys):
            for k in keys:
                if k in b:
                    return b[k]
            return None
        ssim = cg("ssim_mean", "ssim")
        base_ssim = bg("ssim_mean", "ssim")
        row = {"ssim": ssim, "base_ssim": base_ssim,
               "skel_iou": cg("skel_iou_mean", "skel_iou"),
               "mse": cg("mse_mean", "mse"),
               "lpips": cg("lpips_mean", "lpips"),
               "n": cg("n") or g("n")}
        if ssim is not None and base_ssim is not None:
            row["delta_ssim"] = ssim - base_ssim
        return {k: (round(v, 6) if isinstance(v, float) else v) for k, v in row.items() if v is not None}
    row = {k: d.get(k) for k in ("ssim", "mse", "skel_iou", "lpips", "n")}
    return {k: (round(v, 6) if isinstance(v, float) else v) for k, v in row.items() if v is not None}


out = []
for name, base in SERIES:
    rd = latest_run_dir(base)
    if not rd:
        print(f"[skip] {name}: no run dir under {base}")
        continue
    steps = {}
    for f in glob.glob(os.path.join(rd, "checkpoints", "eval_auto_*.json")):
        st = parse_step(f)
        if st is None:
            continue
        try:
            steps[str(st)] = extract(json.load(open(f, encoding="utf-8")))
        except Exception as e:
            print(f"[warn] {f}: {e}")
    out.append({"series": name, "run_dir": rd, "steps": steps})
    best = max((v.get("ssim", -1) for v in steps.values() if v.get("ssim") is not None), default=None)
    print(f"[ok] {name}: {len(steps)} eval points, best ssim = {best}")

json.dump(out, open("assets/v89_eval_series.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print("wrote assets/v89_eval_series.json")
