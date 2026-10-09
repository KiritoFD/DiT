#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""make_ablation_configs.py — 以 v68 为 base 批量生成「OT × 数据增强」消融配置。

设计
----
* **单变量、可复现**：每份配置 = v68 逐字复制 + 明确的少量覆盖；`_comment` 写清差异。
* **等样本预算**：所有臂的 `global_batch_size × max_steps` 对齐 v68 的
  200,000 × 288 = 57,600,000。换 batch 只改步数，不改总样本量。
* **数据面联动**：aug 策略同时决定 `data_csv` / `repa_cache_dir` / `latent_shards_dir`。
  * noaug 可直接复用 sym3x 的 shards 与 DINO cache（它们含原图行），只需一张过滤 CSV
    -> 见 tools/ablation/prepare_data.py。
* **latent 轴独立**：`--latent calli|sd`。calli 走修正约定（shards 由 encode_aug_latents
  `--encode-mode sample` 产出），sd 走 v68 原始 sd-vae 约定（mode≈sample）。

内置臂（可用 --arms 子集）
------------------------
  noaug_c2ot     不增强 + C2OT            (v68 去掉数据增强 -> 孤立增强贡献)
  sym3x_noOT     3x 对称增强 + 不开 OT     (v68 去掉 OT -> 孤立 OT 贡献)
  sym3x_naiveOT  3x 对称增强 + 朴素 OT     (C2OT -> 朴素 OT 的对照)
  noaug_noOT     不增强 + 不开 OT          (最裸基线)
  thick_c2ot     只变粗 3x + C2OT          (单侧加粗; 测 ± 对偶是否必要)
  thin_c2ot      只变细 3x + C2OT          (单侧变细)
  symwide_c2ot   ±1/2/3 px 7x + C2OT      (更宽粗细跨度)
  sym4_c2ot      4-邻域 ±1 px 3x + C2OT   (更细/更锐的形态学)

用法
----
  python tools/ablation/make_ablation_configs.py --latent calli --batch 560
  python tools/ablation/make_ablation_configs.py --print          # 只打表
  python tools/ablation/make_ablation_configs.py --arms noaug_c2ot,sym3x_noOT
"""
import argparse
import copy
import json
import os
import sys
from datetime import datetime

# (name, aug_strategy|None, ot_mode)
ARMS = [
    ("noaug_c2ot",      None,      "c2ot"),
    ("sym3x_noOT",      "sym",     "none"),
    ("sym3x_naiveOT",   "sym",     "naive"),
    ("noaug_noOT",      None,      "none"),
    ("thick_c2ot",      "thick",   "c2ot"),
    ("thin_c2ot",       "thin",    "c2ot"),
    ("symwide_c2ot",    "symwide", "c2ot"),
    ("sym4_c2ot",       "sym4",    "c2ot"),
]
# 数据面依赖的 aug 策略（sym 依赖已存在的 v68 资料，无需重建）
NEW_DATA_STRATEGIES = ["thick", "thin", "symwide", "sym4"]
TOTAL_SAMPLES_DEFAULT = 200_000 * 288


def aug_paths(aug, latent, vae_downscale=8):
    """返回 (data_csv, repa_cache_dir, latent_shards_dir) —— 三处必须同源同 id 空间。"""
    if aug is None:
        csv = "exp-std/csv/train_top10_noaug.csv"
        repa = "data/dino_cache/top10_aug_v1"          # 含原图行, 可直接复用
        shards = ("exp-std/data/shards_img_aug_calli" if latent == "calli"
                  else "exp-std/data/shards_img_aug")
    elif aug == "sym":
        # 历史 v68/v69 的 3x 基线用无后缀目录名, 必须沿用否则会找不到现有 shards/cache
        csv = "exp-std/csv/train_top10_aug_sym.csv"
        repa = "data/dino_cache/top10_aug_v1"
        shards = ("exp-std/data/shards_img_aug_calli" if latent == "calli"
                  else "exp-std/data/shards_img_aug")
    else:
        csv = f"exp-std/csv/train_top10_aug_{aug}.csv"
        repa = f"data/dino_cache/top10_aug_{aug}_v1"
        shards = (f"exp-std/data/shards_img_aug_{aug}_calli" if latent == "calli"
                  else f"exp-std/data/shards_img_aug_{aug}")
    return csv, repa, shards


def build(base, name, aug, ot, latent, batch, vae_dir, sd_vae_dir, total):
    c = copy.deepcopy(base)
    csv, repa, shards = aug_paths(aug, latent)
    steps = max(1, round(total / batch))
    c["experiment_name"] = f"abl-{name}"
    c["results_dir"] = f"assets/results/ablation/{name}"
    c["data_csv"] = csv
    c["repa_cache_dir"] = repa
    c["latent_shards_dir"] = shards
    c["global_batch_size"] = batch
    c["max_steps"] = steps
    c["epochs"] = steps
    c["ckpt_every"] = 5000
    c["gpu_eval_every"] = 5000
    # 求解器 pin（关键）：v68 (2026-10-06) 训练/评测时代还没有 --sde-gamma 这个参数，
    # 其 in-mem eval 走的是**纯 Heun ODE**（FlowMatching 默认 sde_gamma=0.0）。
    # 当前 cli 默认 sde_gamma=0.5，会把 SDE 噪声项混进 eval 采样 -> 与 v68 不可比。
    # 消融统一 pin sde_gamma=0（纯 Heun），其余 (flow_sampler=heun, eval_steps=50,
    # eval_cfg=1.0, shift=1.0) 由 base 逐字继承。
    c["sde_gamma"] = 0.0
    # OT 轴
    c["use_c2ot"] = (ot == "c2ot")
    c["c2ot_mode"] = "slot"
    c["use_ot"] = (ot == "naive")
    # latent / VAE 轴
    if latent == "calli":
        c["vae_path"] = vae_dir
        c["eval_vae_path"] = vae_dir
        c["vae_scaling_factor"] = 0.18215
        c["calli_decode_noise"] = False          # 修正约定后无需补噪
    else:
        # 显式给出路径：48 上 sd-vae 在 pretrained_models/ 下, cli 默认的
        # data/pretrained/sd-vae-ft-ema 不存在 -> 不显式指定会 FileNotFound
        c["vae_path"] = sd_vae_dir
        c["eval_vae_path"] = sd_vae_dir
        c["calli_decode_noise"] = False
    desc = (f"消融臂 {name}：以 v68 为 base，只改「数据增强 + OT」。\n"
            f"  aug      = {aug if aug else '无 (仅原图)'}\n"
            f"  data_csv = {csv}\n"
            f"  repa     = {repa}\n"
            f"  shards   = {shards}   (latent={latent})\n"
            f"  OT       = {ot}   (use_c2ot={c['use_c2ot']}, use_ot={c['use_ot']}, mode=slot)\n"
            f"  batch={batch}  steps={steps}  total_samples={batch * steps:,} "
            f"(对齐 v68 的 200000x288={200_000*288:,})\n"
            f"  其余 (Sp/2 59.17M / lr5e-5 / warmup3000 / cosine / w_repa0.03 / "
            f"cond 表 / drop / eval200fix) 与 v68 逐字相同。\n"
            f"  generated {datetime.now().isoformat(timespec='seconds')}")
    c["_comment"] = desc
    return c, steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="src/train/configs/v68_aug_sp_c2ot.json")
    ap.add_argument("--out-dir", default="src/train/configs/ablation")
    ap.add_argument("--latent", default="calli", choices=["calli", "sd"])
    ap.add_argument("--batch", type=int, default=560)
    ap.add_argument("--total-samples", type=int, default=TOTAL_SAMPLES_DEFAULT)
    ap.add_argument("--vae-dir",
                    default="/home/ds/Workspace/DiT/data/pretrained/pretrained_models/calli_vae")
    ap.add_argument("--sd-vae-dir", default="pretrained_models/sd-vae-ft-ema",
                    help="48 上 sd-vae 的实际位置 (cli 默认 data/pretrained/sd-vae-ft-ema 不存在)")
    ap.add_argument("--arms", default=None, help="逗号分隔的臂名子集; 默认全部")
    ap.add_argument("--print", action="store_true", dest="dry", help="只打印计划，不落盘")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    base = json.load(open(a.base, encoding="utf-8"))
    arms = ARMS
    if a.arms:
        want = [x.strip() for x in a.arms.split(",") if x.strip()]
        by = {n: (n, g, o) for n, g, o in ARMS}
        arms = [by[w] for w in want]
    os.makedirs(a.out_dir, exist_ok=True)

    print("=" * 96)
    print(f"[cfg] base={a.base}  latent={a.latent}  batch={a.batch}  "
          f"total_samples={a.total_samples:,}")
    print(f"{'arm':<18} {'aug':<9} {'OT':<7} {'steps':>8} {'csv':<46} {'shards'}")
    print("-" * 96)
    for name, aug, ot in arms:
        c, steps = build(base, name, aug, ot, a.latent, a.batch, a.vae_dir, a.sd_vae_dir,
                         a.total_samples)
        need = aug in NEW_DATA_STRATEGIES
        print(f"{name:<18} {str(aug or 'none'):<9} {ot:<7} {steps:>8} "
              f"{c['data_csv']:<46} {c['latent_shards_dir']}"
              + ("   [需新建数据]" if need else ""))
        if not a.dry:
            p = os.path.join(a.out_dir, f"{name}.json")
            json.dump(c, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("-" * 96)
    if a.dry:
        print("[cfg] --print：未落盘")
    else:
        print(f"[cfg] 已写入 {len(arms)} 份 -> {a.out_dir}/")
    need_new = sorted({g for _, g, _ in arms if g in NEW_DATA_STRATEGIES})
    print(f"[cfg] 需 prepare_data.py 新建的增强策略: {need_new or '无 (全部复用现有数据)'}")
    print(f"[cfg] latent={a.latent} 时需先编码 shards: "
          f"{'是 (encode-mode sample)' if a.latent == 'calli' else '是 (encode-mode mode)'} "
          f"-> 见 README")
    print("=" * 96)
    return 0


if __name__ == "__main__":
    sys.exit(main())
