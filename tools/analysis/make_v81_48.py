# -*- coding: utf-8 -*-
"""make_v81_48.py — 生成本机 v81 的「48 版」配置（路径按 48 的仓库布局对齐）"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

src = "src/train/configs/v81_stdsoft_vs_v68.json"
d = json.load(open(src, encoding="utf-8"))
print("  本机 v81 的路径类键:")
for k in ("data_csv", "latent_shards_dir", "vae_path", "eval_vae_path", "img_root",
          "latent_img_root", "vae_scaling_factor", "vae_shift_factor", "std_soft_root",
          "results_dir", "global_batch_size", "max_steps", "eval_cfg", "sde_gamma", "use_c2ot"):
    if k in d:
        print(f"    {k:22} = {d[k]!r}")

# 48 的布局与 ablation 配置一致（见 48 上 _ab_arm4.json）
d["experiment_name"] = "v81-stdsoft-vs-v68-48"
d["results_dir"] = "assets/results/v81_stdsoft_vs_v68_48"
d["img_root"] = "data/top10_style23/imgs"          # ← 48 用的相对根
for k in ("latent_img_root", "image_root", "data_root"):
    if k in d:
        d[k] = "data/top10_style23/imgs"
d["std_soft_root"] = "exp-std/data/std_soft_glyphs_v1"
d["_comment"] = (d.get("_comment", "") +
                 "\n\n【48 版】与 v68 单变量 A/B，跑在 48（GPU 空闲）。路径已对齐 48 布局；"
                 "std_soft store 由本机同步（8451 条、零缺字）。")
json.dump(d, open("src/train/configs/v81_stdsoft_vs_v68_48.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)
print("\n  -> src/train/configs/v81_stdsoft_vs_v68_48.json")
print("  关键项:", {k: d.get(k) for k in ("use_c2ot", "global_batch_size", "max_steps",
                                        "std_soft_cond", "std_soft_root", "data_csv")})
