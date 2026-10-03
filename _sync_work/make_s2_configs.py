#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""make_s2_configs.py — 生成 S2（三层语义分解 + 局部风格-骨架引导）消融阶梯配置。

设计依据: docs/922/20_style_encoding.md / 30_injection.md / 40_decision.md

## 为什么要"逐级消融"而不是一次全上

S2 一次引入四样东西（pair 残差 / 书体 FiLM / 局部 FiLM / 局部 cross-attn）。
如果一次全上且没涨，无法归因；如果涨了，也不知道是哪一样。
所以本脚本生成**单变量阶梯**，每一步只加一样。

## 与 v13_base 的可比性（关键）

所有 S2 配置以 ``src/train/configs/v13_base_50k.json`` 为模板**逐字段复制**，
只改下面列出的字段。所以：

  * 同一份数据（``train_50k_v2.csv``）—— 不换成 _fixed，否则与 v13_base 不可比
  * 同一个 schedule（max_steps 250000 / warmup 3000 / cosine / min_lr_ratio 0.1）
    → 这样在**任意 step** 上都能和 v13_base 同口径对比（跑到 ~160k 即可判读）
  * 同样冻结书家表 + 同一份 SupCon 预训练表
  * ``glyph_drop_prob`` 保持 0.0（不引入第二个变量）

用法（本地生成，再 scp 到远端）:
    python _sync_work/make_s2_configs.py            # 写 src/train/configs/v17_s2_*.json
    python _sync_work/make_s2_configs.py --fixed-data  # 用清洗后的 _fixed 数据（对照 v13 需重跑基线）
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE = os.path.join(ROOT, "src", "train", "configs", "v13_base_50k.json")

# variant -> (说明, patch)
VARIANTS = {
    # ── 阶梯 1：只加书体层（pair 残差冻结）─────────────────────────────────
    "s2a_scriptfilm": (
        "只加书体 FiLM on g_tok（pair 残差冻结，参数个数相同）",
        dict(hier_style=1, pair_residual=0, script_film=True),
    ),
    # ── 阶梯 2：只加 pair 交互残差（无书体 FiLM）────────────────────────────
    "s2b_pairres": (
        "只加 pair 交互残差（e_style = E_callig + α·E_pair）",
        dict(hier_style=1, pair_residual=1, script_film=False),
    ),
    # ── 阶梯 3：完整 S2（书体 FiLM + pair 残差）────────────────────────────
    "s2c_full": (
        "完整 S2 = 书体 FiLM + pair 残差",
        dict(hier_style=1, pair_residual=1, script_film=True),
    ),
    # ── 阶梯 4：+ 局部逐位置 FiLM（Phase 1，便宜的局部性验证）──────────────
    "s2d_spfilm": (
        "S2 + SpatialStyleFiLM(rank 64) —— 局部性验证（便宜）",
        dict(hier_style=1, pair_residual=1, script_film=True,
             spatial_film_rank=64),
    ),
    # ── 阶梯 5：+ 局部 cross-attn adapter（Phase 2，Q=g 更安全）─────────────
    "s2e_lca_g": (
        "S2 + LocalStyleGlyphAdapter ×2（block 2,6；Q=g_tok）",
        dict(hier_style=1, pair_residual=1, script_film=True,
             local_ca_layers=2, local_ca_at="2,6", local_ca_q="g",
             local_ca_heads=4, local_ca_rank=64),
    ),
    # ── 阶梯 6：同上但 Q=x + 窗口注意力 ────────────────────────────────────
    "s2f_lca_x": (
        "S2 + LocalStyleGlyphAdapter ×2（Q=x, window=5）",
        dict(hier_style=1, pair_residual=1, script_film=True,
             local_ca_layers=2, local_ca_at="2,6", local_ca_q="x",
             local_ca_window=5, local_ca_heads=4, local_ca_rank=64),
    ),
    # ── 阶梯 7：全套（局部 FiLM + 局部 CA 同时开）──────────────────────────
    "s2g_all": (
        "S2 全套：书体 FiLM + pair 残差 + SpatialStyleFiLM + LocalCA(Q=g)",
        dict(hier_style=1, pair_residual=1, script_film=True,
             spatial_film_rank=64, local_ca_layers=2, local_ca_at="2,6",
             local_ca_q="g", local_ca_heads=4, local_ca_rank=64),
    ),
}

# 所有 S2 变体共用的 patch（不是消融变量）
COMMON = dict(
    num_pairs=87,          # 87 个 (书家×书体) pair
    num_scripts=8,         # script_id ∈ {0,3,4}，取 8 留余量
    script_embed_dim=64,
    pair_init="zero",
    # ⚠ 必须设: 数据集靠它产出 y_pair（87 号）。hier 模式下主效应表用 y_callig_raw(45)，
    #   所以 y_callig 装的是 87 号也不影响（模型不读它）。
    callig_script_map="assets/callig_script_id_map.json",
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixed-data", action="store_true",
                    help="用 assets/train_50k_v2_fixed.csv + data/50k/shards_std_fixed "
                         "(229 异体 + 751 简繁修正)。⚠ 与 v13_base 不可比, 需另跑基线")
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "src", "train", "configs"))
    ap.add_argument("--only", default=None, help="只生成某个 variant")
    args = ap.parse_args()

    base = json.load(open(TEMPLATE, encoding="utf-8"))
    n = 0
    for name, (desc, patch) in VARIANTS.items():
        if args.only and args.only != name:
            continue
        d = dict(base)
        d.update(COMMON)
        d.update(patch)
        if args.fixed_data:
            d["data_csv"] = "assets/train_50k_v2_fixed.csv"
            d["skel_latent_shards_dir"] = "data/50k/shards_std_fixed"
            d["eval_skel_latent_shards_dir"] = "data/50k/shards_std_fixed"
            d["in_mem_eval_sets"] = (
                "seen:assets/eval_v13_seen_fixed.csv:20,"
                "strict:assets/eval_v13_strict_fixed.csv:249")
            _suffix = "_fixed"
        else:
            _suffix = ""
        d["experiment_name"] = f"v17-s2-{name}{_suffix}"
        d["results_dir"] = f"assets/results/v17_s2_{name}{_suffix}"
        d["_comment"] = f"[S2 消融] {desc}｜模板=v13_base_50k.json（逐字段复制，只改本文件列出的字段）"
        out = os.path.join(args.out_dir, f"v17_s2_{name}{_suffix}.json")
        json.dump(d, open(out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
        print(f"[s2] {os.path.basename(out)}")
        print(f"     {desc}")
        n += 1
    print(f"\n生成 {n} 个配置。基线对照: src/train/configs/v13_base_50k.json "
          f"(strict 0.5703 @155k / ink_ssim 0.3872)")
    if args.fixed_data:
        print("⚠ 用了 _fixed 数据 -> v13_base 不再是合法对照，需先跑一条 _fixed 基线")
    return 0


if __name__ == "__main__":
    sys.exit(main())
