import os
import sys
import glob

sys.stdout.reconfigure(encoding="utf-8")

p = "/root/Workspace/xy/DiT/assets/results/v54_minimal_tables_noskel/20261005-000439-v54-minimal-tables-S-noskel-freeze"
print(f"检查 v54 目录: {p}")
if os.path.exists(p):
    print("目录存在！查找 eval 子目录:")
    eval_dirs = glob.glob(f"{p}/**/eval*", recursive=True) + glob.glob(f"{p}/**/step*", recursive=True)
    for ed in eval_dirs:
        n_png = len(glob.glob(f"{ed}/*.png"))
        print(f"  {os.path.basename(ed)}: {n_png} pngs in {ed}")
else:
    print("目录不存在，搜索其它包含 v54 的目录:")
    matches = glob.glob("/root/Workspace/xy/DiT/assets/results/*v54*")
    print(matches)
