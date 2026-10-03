# -*- coding: utf-8 -*-
"""_diag_wz_scale.py — 诊断"生成全是墨团"的根因.

假设: 白底归零把目标 latent 从"近似标准正态"变成"稀疏 + 大负偏", 
      flow 的噪声端点 (eps~N(0,1)) 与目标分布严重不匹配 -> 模型学不到结构,
      采出来是"平均下来的黑团"。

对比: 未减白底 (final_latents_base_shards) vs 已减白底 (final_latents_base_wz)
      以及 aux 通道。
"""
import glob
import os
import sys

import numpy as np

ROOT = "/root/Workspace/xy/DiT"
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CAND = [
    ("img  未减白底", "data/latents/final_latents_base_shards"),
    ("img  已减白底(wz)", "data/latents/final_latents_base_wz"),
    ("canny 未减白底", "data/aux/aux_canny_latents_base"),
    ("canny 已减白底(wz)", "data/aux/aux_canny_latents_base_wz"),
    ("skel3 未减白底", "data/skel/aux_skel3_latents_base"),
    ("skel3 已减白底(wz)", "data/skel/aux_skel3_latents_base_wz"),
    ("std_g (条件,未减)", "data/skel/std_skel3_latents_base_full"),
]

print(f"{'通道':22s} {'mean':>8s} {'std':>8s} {'|x|>2 占比':>10s} {'p1':>8s} {'p99':>8s}  逐通道 std")
for tag, d in CAND:
    fs = sorted(glob.glob(os.path.join(d, "shard_*.npz")))
    if not fs:
        print(f"  {tag:22s}  (无 shard: {d})")
        continue
    z = np.load(fs[0])
    lat = np.asarray(z["latents"][:256], dtype=np.float32)
    cs = lat.std(axis=(0, 2, 3))
    print(f"  {tag:22s} {lat.mean():8.4f} {lat.std():8.4f} "
          f"{(np.abs(lat) > 2).mean():10.4f} "
          f"{np.percentile(lat,1):8.3f} {np.percentile(lat,99):8.3f}  "
          f"{np.round(cs, 3)}")

print("\n[判据] flow 的起点是 eps~N(0,1) (std=1)。若目标 std 远小于 1,")
print("       v=eps-x0 几乎等于 eps -> 模型学'预测噪声'很容易, loss 会降但学不到结构;")
print("       采样时积分从 eps 到 0, 得到接近 0 的 x0 -> decode 成灰/黑团。")
print("       若目标还有大负偏 (mean << 0), 平均策略会让输出整体偏黑 -> '墨团'。")
