"""验证"切到修正版 eval"这条接线是否真的通 —— **纯 CPU, 不碰 GPU**。

检查项:
 1. config 里三个路径存在 (csv / 条件 shards / 目标 shards)
 2. 修正版 csv 的行数、列名、img_id 与条件 shard 的 img_id **完全一致**
 3. 目标 shards(shards_img) 覆盖这些 img_id (eval 需要目标图 latent)
 4. 条件 shard 能正常读出 (4,32,32) f16, 无 NaN
 5. 用真实 CLI 解析器干跑该 config 的 argv (argparse 层无错, 且值原样落地)
"""
import csv
import glob
import json
import os
import subprocess
import sys

import numpy as np

ROOT = "/root/Workspace/xy/DiT"
os.chdir(ROOT)
CFG = "src/train/configs/v46_5050_80k.json"
c = json.load(open(CFG, encoding="utf-8"))
CSV, COND, TGT = (c["in_mem_eval_sets"].split(":")[1],
                  c["eval_skel_latent_shards_dir"], c["latent_shards_dir"])
print(f"[cfg] csv={CSV}\n      cond={COND}\n      tgt={TGT}")


def ids_of(d):
    s = set()
    for f in glob.glob(os.path.join(d, "shard_*.npz")):
        with np.load(f) as z:
            s |= {int(x) for x in z["img_ids"]}
    return s


ok = True
for p in (CSV, COND, TGT):
    e = os.path.exists(p)
    print(f"[1] 存在 {p} = {e}")
    ok &= e

rows = list(csv.DictReader(open(CSV, encoding="utf-8")))
csv_ids = {int(r["img_id"]) for r in rows}
cond_ids = ids_of(COND)
tgt_ids = ids_of(TGT)
n_spec = int(c["in_mem_eval_sets"].rsplit(":", 1)[1])
print(f"[2] csv 行数={len(rows)} (sets 里写 {n_spec}) | cond shard={len(cond_ids)}")
print(f"    csv.imgs == cond.imgs ? {csv_ids == cond_ids}"
      + (f"  差集 csv-cond={list(csv_ids - cond_ids)[:5]} cond-csv={list(cond_ids - csv_ids)[:5]}"
         if csv_ids != cond_ids else ""))
ok &= (len(rows) == n_spec) and (csv_ids == cond_ids)

miss = csv_ids - tgt_ids
print(f"[3] 目标 shards 覆盖: 缺 {len(miss)}/个 {list(miss)[:5]} | tgt 总数={len(tgt_ids)}")
ok &= not miss

bad = 0
for f in glob.glob(os.path.join(COND, "shard_*.npz")):
    with np.load(f) as z:
        a = z["latents"]
        bad += int(np.isnan(a.astype(np.float32)).any())
        print(f"[4] {os.path.basename(f)} latents={a.shape} {a.dtype} "
              f"均值={float(a.astype(np.float32).mean()):+.3f} "
              f"|max|={float(np.abs(a.astype(np.float32)).max()):.2f}")
        print(f"    每样本零值率 min={min(float((x == 0).mean()) for x in a):.3f} "
              f"(若=1.0 说明条件被解成空白)")
ok &= (bad == 0)

argv = ["-u", "src/train/train.py", "--config", CFG, "--results-dir", "/tmp/_v",
        "--max-steps", "1"]
r = subprocess.run([sys.executable, "tools/preflight_stage_args.py"] + argv,
                   capture_output=True, text=True)
print(f"[5] preflight rc={r.returncode}")
print("    " + (r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr.strip()[-200:]))
ok &= (r.returncode == 0)

print("\n===== 接线验证: " + ("全部通过 ✅" if ok else "有失败项 ❌") + " =====")
