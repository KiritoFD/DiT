# -*- coding: utf-8 -*-
"""build_master_results.py — 汇总全部实验数据 → 树状 results/ + 极详细 master_results.csv/json."""
import os, sys, csv, json, glob, shutil
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
sys.stdout.reconfigure(encoding="utf-8")

# ---- 数据代际定义 ----
ERAS = {
    "pixel_ddpm": {"models": ["DiT-2Cond-S/2", "DiT-2Cond-B/2", "DiT-2Cond-B/4", "DiT-2Cond-S/4", "DiT-3Cond-S/2"], "desc": "像素空间 DDPM (s2-s11, v3)"},
    "latent_ddpm": {"models": ["DiT-2Cond-S/2", "DiT-2Cond-XS/2"], "desc": "latent DDPM (s12-s14)"},
    "latent_flow": {"models": ["DiT-2Cond-S/2", "DiT-2Cond-WS/2"], "desc": "latent flow (s15-s21)"},
    "v2_arch": {"models": ["DiT-2Cond-S/2"], "desc": "v2 架构 rms/swiglu/qk_norm/RoPE (s21+)", "marker": "v8"},
}
DATASET_SIZES = {
    "top6": 10866, "top30": 45000, "3top30": 38000, "top30_clean": 38000,
    "train.csv": 62157, "kailishu": 20000, "mid_clean": 118776,
    "mid_common": 23597, "fame": 51322, "fame_clean_v8": 51321,
}

def era_of(model, series):
    if "v8" in series or "s3" in series: return "v2_arch"
    if "s1" in series and any(x in series for x in ("s15","s17","s18","s19")): return "latent_flow"
    if "s12" in series: return "latent_ddpm"
    return "pixel_ddpm"

# ---- 收集全部数据 ----
all_runs = []

# 来源 1: all_experiments_eval_20260903.csv (224 runs, 含配置)
src1 = "5script/all_experiments_eval_20260903.csv"
if os.path.exists(src1):
    seen = set()
    for r in csv.DictReader(open(src1, encoding="utf-8")):
        key = (r["series"], r["run"])
        if key in seen: continue
        seen.add(key)
        all_runs.append({
            "series": r["series"], "run": r["run"],
            "source": "all_experiments_20260903",
            "model": r.get("cfg_model",""), "data": r.get("cfg_data_csv",""),
            "diffusion": r.get("cfg_diffusion_type",""),
            "best_step": r.get("best_step",""), "best_ssim": r.get("best_ssim",""),
            "best_mse": r.get("best_mse",""),
            "config_json": json.dumps({k:v for k,v in r.items() if k.startswith("cfg_")}, ensure_ascii=False),
        })

# 来源 2: eval_unified_20260829.csv (统一口径)
src2 = "5script/eval_unified_20260829.csv"
if os.path.exists(src2):
    for r in csv.DictReader(open(src2, encoding="utf-8")):
        all_runs.append({
            "series": r.get("experiment",""), "run": r.get("run_dir",""),
            "source": "eval_unified_20260829",
            "protocol": "eval_strict_midclean n=501 cfg=r.get(cfg) steps=50",
            "n": r.get("n",""), "cfg": r.get("cfg",""),
            "ssim": r.get("ssim",""), "mse_x4": r.get("mse_x4",""),
            "lpips": r.get("lpips",""), "skel_iou": r.get("skel_iou",""),
            "ssim_楷": r.get("ssim_楷",""), "ssim_行": r.get("ssim_行",""), "ssim_隶": r.get("ssim_隶",""),
        })

# 来源 3: 1px daemon (v8b 之前的 1px 训练)
onepix = [
    {"step": 17500, "mse": 0.20491, "ssim": 0.7584, "skel_iou": 0.2708},
    {"step": 20000, "mse": 0.19728, "ssim": 0.7641, "skel_iou": 0.2810},
    {"step": 25000, "mse": 0.18117, "ssim": None, "skel_iou": 0.2920},
    {"step": 27500, "mse": 0.17632, "ssim": None, "skel_iou": 0.2979},
    {"step": 30000, "mse": 0.17123, "ssim": None, "skel_iou": 0.3014},
    {"step": 32500, "mse": 0.16610, "ssim": None, "skel_iou": 0.3069},
    {"step": 35000, "mse": 0.16145, "ssim": None, "skel_iou": 0.3126},
    {"step": 37500, "mse": 0.15740, "ssim": None, "skel_iou": 0.3193},
    {"step": 40000, "mse": 0.15450, "ssim": 0.7929, "skel_iou": 0.3216},
    {"step": 42500, "mse": 0.15261, "ssim": 0.7947, "skel_iou": 0.3235},
    {"step": 45000, "mse": 0.15123, "ssim": 0.7959, "skel_iou": 0.3253},
    {"step": 47500, "mse": 0.14986, "ssim": 0.7969, "skel_iou": 0.3286},
    {"step": 50000, "mse": 0.14924, "ssim": 0.7974, "skel_iou": 0.3318},
]
# 来源 4: v8b ctrl (当前训练)
v8b = [
    {"step": 2500, "delta_ssim": 0.2054}, {"step": 5000, "delta_ssim": 0.2124},
    {"step": 7500, "delta_ssim": 0.2211},
    {"step": 10000, "delta_ssim": 0.2276}, {"step": 12500, "delta_ssim": 0.2325},
    {"step": 15000, "delta_ssim": 0.2371}, {"step": 17500, "delta_ssim": 0.2390},
    {"step": 20000, "delta_ssim": 0.2418}, {"step": 22500, "delta_ssim": 0.2431},
    {"step": 25000, "delta_ssim": 0.2444}, {"step": 27500, "delta_ssim": 0.2452},
    {"step": 30000, "delta_ssim": 0.2457}, {"step": 32500, "delta_ssim": 0.2458},
    {"step": 35000, "delta_ssim": 0.2459}, {"step": 37500, "delta_ssim": 0.2458},
]
# 来源 5: fame final eval
final = "5script/fame_final_eval.json"
if os.path.exists(final):
    d = json.load(open(final, encoding="utf-8"))
    for arm, data in d.items():
        all_runs.append({
            "series": f"fame_final_{arm}", "run": "ctrl_fame_v2/094156",
            "source": "fame_final_eval", "n": data.get("n",""),
            "ssim_median": data.get("ssim_median",""), "ssim_mean": data.get("ssim_mean",""),
            "mse_median": data.get("mse_median",""), "fails": data.get("fails",""),
            "per_script": json.dumps(data.get("per_script_median",""), ensure_ascii=False),
        })
# 来源 6: zero-shot
zs = "5script/fame_zero_shot_eval.json"
if os.path.exists(zs):
    d = json.load(open(zs, encoding="utf-8"))
    for arm, data in d.items():
        all_runs.append({"series": f"zero_shot_{arm}", "source": "zero_shot_eval",
                        "n": data.get("n",""), "ssim_median": data.get("ssim_median",""),
                        "ssim_mean": data.get("ssim_mean",""), "mse_median": data.get("mse_median",""),
                        "fails": data.get("fails","")})

# ---- 输出 master CSV ----
all_keys = set()
for r in all_runs:
    all_keys.update(r.keys())
fieldnames = sorted(all_keys)
out_csv = os.path.join(ROOT, '5script', 'master_results.csv')
with open(out_csv, "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
    w.writeheader()
    w.writerows(all_runs)
print(f"master_results.csv: {len(all_runs)} rows, {len(fieldnames)} columns")

# ---- 输出 master JSON (带完整嵌套) ----
out_json = os.path.join(ROOT, '5script', 'master_results.json')
json.dump(all_runs, open(out_json, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
print(f"master_results.json: {len(all_runs)} entries")

# ---- 树状结构 ----
tree = {}
for r in all_runs:
    s = r.get("series","unknown")
    tree.setdefault(s, []).append(r)
print("\n=== 树状结构 ===")
for s in sorted(tree):
    runs = tree[s]
    best = max((float(r.get("best_ssim",0) or 0) for r in runs), default=0)
    print(f"  {s} ({len(runs)} runs, best_ssim={best:.4f})")
