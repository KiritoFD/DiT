import glob
import json
import os
import re

os.chdir("/root/Workspace/xy/DiT")
p = sorted(glob.glob("src/train/configs/v16_fs_沈周-行.json"))[0]
d = json.load(open(p, encoding="utf-8"))
print("  == config:", p)
for k in ("model", "num_calligraphers", "callig_embed_dim", "callig_n_style",
          "k_clusters", "data_csv", "in_mem_eval_sets", "max_steps",
          "global_batch_size", "lr", "lr_schedule", "use_ema",
          "freeze_callig_table", "callig_script_map", "callig_emb_pretrained",
          "cond_drop_all_prob", "glyph_inject_mode", "condition_fusion"):
    if k in d:
        print(f"    {k:<24} {d[k]!r}")

print()
print("  == diff 全程趋势 (沈周-行) ==")
L = "logs/v16_series/沈周-行_0921-130259.log"
rows = re.findall(r"step=(\d+)\) Diff: ([\d.]+)", open(L, encoding="utf-8",
                                                       errors="ignore").read())
for i in range(0, len(rows), max(1, len(rows) // 10)):
    print(f"    step={rows[i][0]:>7}  Diff={rows[i][1]}")
if rows:
    print(f"    首={rows[0][1]}  末={rows[-1][1]}  共 {len(rows)} 个采样点")

print()
print("  == 其它 v16 日志的 diff 末值 ==")
for f in sorted(glob.glob("logs/v16_series/*.log")):
    rr = re.findall(r"step=(\d+)\) Diff: ([\d.]+)", open(f, encoding="utf-8",
                                                         errors="ignore").read())
    if rr:
        print(f"    {os.path.basename(f)[:28]:<30} 首={rr[0][1]} 末={rr[-1][1]} "
              f"(step {rr[-1][0]})")
