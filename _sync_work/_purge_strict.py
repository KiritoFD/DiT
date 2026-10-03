# -*- coding: utf-8 -*-
"""_purge_strict.py — 清除 strict 集的错误指标, 以便用修复后的 g 重新评测.

背景: eval 侧的 std skel 曾误用训练目录 (img_id 不是同一套) -> strict 命中 0/237
      -> g 全零 -> decode(0) 解出灰黄棕 -> poster 首行发黄, 且 strict 指标失真
      (曾出现 strict 0.4837 > seen 0.4577 的反常)。
      run_in_mem_eval 会跳过 summary 里已存在的 (step,set), 所以必须先删掉
      strict 的错误行; seen 的行是正确的, 保留。
"""
import csv
import os
import shutil
import sys

ROOT = "/root/Workspace/xy/DiT"
os.chdir(ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

RD = "assets/results/v11_pretrain_Sp2_base_sym"


def purge(path, target_set="strict"):
    if not os.path.exists(path):
        print(f"  {path} 不存在")
        return
    shutil.copy(path, path + ".bak-gzero")
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    if not rows:
        print(f"  {path} 空")
        return
    keep = [r for r in rows if r.get("set") != target_set]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(keep)
    print(f"  {os.path.basename(path)}: 删除 {target_set} {len(rows)-len(keep)} 行, "
          f"保留 {len(keep)} 行 (备份 .bak-gzero)")


print("[purge] 清除 strict 错误指标")
purge(os.path.join(RD, "eval_stdskel_summary.csv"))
purge(os.path.join(RD, "eval_stdskel_batch.csv"))

# save_input_g 是幂等的 (已存在就跳过) -> 必须删掉旧的黄图才会重新生成
d = os.path.join(RD, "eval_samples_ctrl", "strict_input_g")
if os.path.isdir(d):
    shutil.rmtree(d)
    print(f"  已删除 {d} (强制用正确的 g 重新生成)")
else:
    print(f"  {d} 不存在")
print("[purge] done")
