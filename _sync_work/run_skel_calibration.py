# -*- coding: utf-8 -*-
"""run_skel_calibration.py — 「骨架要多准」定准实验。

动机
----
v26(喂 GT 骨架) strict ssim 0.8163, 而接了 SkelNet 的 v27/v29 只有 0.585~0.605,
且 skel_iou 比"不用 SkelNet"的 v25(0.0238) 还低。差距全部来自骨架本身。
本实验定量回答: **骨架从 std 走向 gt 的过程中, 主干收益怎么兑现?**
-> 从而判断继续死磕 SkelNet 值不值、以及"预测骨架至少要有多准"。

做法
----
冻结 v26 主干(不动权重), 把 g 条件换成 latent 域的线性插值:
    g(α) = α * g_gt + (1-α) * g_std      α ∈ {0, .25, .5, .75, 1}
  α=0 -> 就是"直接用标准骨架"(≈ v25 口径)
  α=1 -> 就是"喂 GT 骨架"(v26 oracle)
再额外跑一个 **predskel 实测点**(当前 SkelNet 的真实输出) 以便把"现状"标到曲线上。

判据用 **训练期完全相同的 run_in_mem_eval** (同样的采样/同样的指标列),
避免自造评测造成口径漂移。

用法 (远程):
  python tools/run_skel_calibration.py \
      --ckpt assets/results/v26_gtskel/20260929-103927-v26-gtskel/checkpoints/0030000.pt
"""
import argparse
import csv
import glob
import os
import re
import shutil
import sys

import numpy as np
import torch as th

sys.path.insert(0, "/root/Workspace/xy/DiT")
os.chdir("/root/Workspace/xy/DiT")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

B = "data/top10_style23"
SEEN = ("seen", "assets/eval_top10_seen_20.csv", 20)
STRICT = ("strict", "assets/eval_top10_strict_subset84.csv", 84)
# 各集的 g_gt / g_std 来源 (see diag_calib_coverage.py 的覆盖实测)
SRC = {
    "seen": dict(gt=f"{B}/shards_aux_skel3", std=f"{B}/shards_std"),
    "strict": dict(gt=f"{B}/gt_skel_eval_strict84",
                   std="data/50k_v2_glyph15k/shards_std"),
}


def load_map(d):
    """img_id -> latent (4,32,32)"""
    mp = {}
    for sp in sorted(glob.glob(os.path.join(d, "shard_*.npz"))):
        with np.load(sp) as z:
            lat = z["latents"]
            for j, i in enumerate(z["img_ids"]):
                mp[int(i)] = np.asarray(lat[j], dtype=np.float32)
    return mp


def csv_ids(p):
    out = []
    for r in csv.DictReader(open(p, encoding="utf-8")):
        m = re.search(r"(\d+)\.png", r["image_path"])
        if m:
            out.append(int(m.group(1)))
    return out


def write_shards(out_dir, id2lat):
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    ids = sorted(id2lat)
    n, k = 0, 0
    for s in range(0, len(ids), 1000):
        chunk = ids[s:s + 1000]
        np.savez(os.path.join(out_dir, f"shard_{k:05d}.npz"),
                 latents=np.stack([id2lat[i] for i in chunk]),
                 img_ids=np.array(chunk, dtype=np.int64))
        n += len(chunk)
        k += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="assets/results/_calib_skel")
    ap.add_argument("--alphas", default="0,0.25,0.5,0.75,1.0")
    ap.add_argument("--no-pred", dest="pred", action="store_false", default=True,
                    help="跳过 predskel 实测点 (自检时用)")
    # ckpt 里存的 args 可能没有 *_pred 键 (后加的 config 键), 故允许显式指定
    ap.add_argument("--pred-seen", default=f"{B}/predskel_eval_seen20")
    ap.add_argument("--pred-strict", default=f"{B}/predskel_eval_strict84")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--cfg", type=float, default=None, help="默认沿用 ckpt 的 eval_cfg")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    from src.eval.model_io import load_model_from_ckpt
    import src.eval.in_mem_eval as IME

    print(f"[calib] ckpt = {a.ckpt}", flush=True)
    model, A = load_model_from_ckpt(a.ckpt, device=a.device, use_ema=True)
    model.eval()
    if a.cfg is not None:
        A.eval_cfg = a.cfg
    A.eval_steps = a.steps
    A.in_mem_eval_batch = a.batch
    A.in_mem_eval_vae_batch = a.batch
    print(f"[calib] eval_cfg={getattr(A,'eval_cfg',None)} steps={A.eval_steps} "
          f"model={getattr(A,'model',None)}", flush=True)
    print(f"[calib] seen skel(train)={getattr(A,'skel_latent_shards_dir','')}")
    print(f"[calib] strict skel(eval)={getattr(A,'eval_skel_latent_shards_dir','')}")
    print(f"[calib] pred dirs: {getattr(A,'skel_latent_shards_dir_pred','')} | "
          f"{getattr(A,'eval_skel_latent_shards_dir_pred','')}")

    # ── 准备 g_gt / g_std ──
    pools = {}
    for name, csvp, n in (SEEN, STRICT):
        ids = csv_ids(csvp)[:n]
        gt = load_map(SRC[name]["gt"])
        std = load_map(SRC[name]["std"])
        miss_gt = [i for i in ids if i not in gt]
        miss_std = [i for i in ids if i not in std]
        print(f"[calib] {name}: {len(ids)} ids, g_gt 缺 {len(miss_gt)}, g_std 缺 {len(miss_std)}")
        if miss_gt or miss_std:
            raise SystemExit(f"[FATAL] {name} 骨架缺失, 无法定准")
        pools[name] = (ids, gt, std)

    os.makedirs(a.out, exist_ok=True)
    rows_out = []

    def run_pass(tag, dirs, sets, step):
        """dirs: {set_name: shards_dir}; sets: [(name,csv,n)]"""
        A.skel_latent_shards_dir = dirs.get("seen", "")
        A.eval_skel_latent_shards_dir = dirs.get("strict", "")
        A.skel_latent_shards_dir_pred = dirs.get("seen_pred", "")
        A.eval_skel_latent_shards_dir_pred = dirs.get("strict_pred", "")
        A.in_mem_eval_sets = ",".join(f"{nm}:{cs}:{n}" for nm, cs, n in sets)
        IME._CACHES.clear()          # ⚠ 缓存键只有 (csv,n), 换 shards 必须清
        res = IME.run_in_mem_eval(model, A, step, a.device, a.out, sets=None)
        print(f"[calib] {tag}: {res}", flush=True)
        return res

    # ── α 扫描 ──
    _al = [float(x) for x in a.alphas.split(",") if x.strip()]
    for al in _al:
        id2lat = {}
        for name in ("seen", "strict"):
            ids, gt, std = pools[name]
            for i in ids:
                id2lat[i] = al * gt[i] + (1.0 - al) * std[i]
        d = os.path.join(a.out, f"_g_alpha{al:.2f}")
        cnt = write_shards(d, id2lat)
        print(f"\n=== α={al} ({cnt} ids) ===", flush=True)
        res = run_pass(f"alpha={al}", {"seen": d, "strict": d},
                       [SEEN, STRICT], step=int(al * 10000) + 1)
        rows_out.append((f"alpha={al}", res))

    # ── predskel 实测点 ──
    if a.pred:
        # ★ 显式 CLI 优先于 ckpt 存的属性 —— 否则 --alphas " "(不跑 α pass、属性
        #   不会被 run_pass 重置) 时会静默用 ckpt 里的旧目录 (2026-09-30 踩过:
        #   β 扫描三个 β 全测了同一个旧 predskel, 数字一模一样)。
        pdir_s = a.pred_seen or getattr(A, "skel_latent_shards_dir_pred", "")
        pdir_t = a.pred_strict or getattr(A, "eval_skel_latent_shards_dir_pred", "")
        if pdir_s and pdir_t and os.path.isdir(pdir_s) and os.path.isdir(pdir_t):
            print("\n=== predskel (当前 SkelNet 真实输出) ===", flush=True)
            # ★ 修泄露 (2026-10-01): ckpt 里存的 eval_blend_alpha (实测=0.5) 会把
            #   std 骨架混进 pred 条件 -> "predskel" 这一路评的其实是半个标准骨架
            #   (std 源自 GT 字形), 好分数是泄露出来的。pred 口径必须评**纯生成骨架**。
            _bak_blend = getattr(A, "eval_blend_alpha", None)
            A.eval_blend_alpha = 1.0
            res = run_pass("predskel", {"seen_pred": pdir_s, "strict_pred": pdir_t},
                           [("seen_pred", SEEN[1], SEEN[2]),
                            ("strict_pred", STRICT[1], STRICT[2])], step=90001)
            if _bak_blend is not None:
                A.eval_blend_alpha = _bak_blend
            rows_out.append(("predskel", res))
        else:
            print(f"[calib] 跳过 predskel (目录缺失: {pdir_s} | {pdir_t})")

    print("\n=== 汇总 ===")
    for tag, res in rows_out:
        print(f"  {tag:>12}: {res}")


if __name__ == "__main__":
    main()
