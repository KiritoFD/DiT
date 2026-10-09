import os
import shutil

src_m12 = "assets/server_48_evals/moyi_12ch_eval200fix"
dst_m12 = "assets/eval200_30_gathered/48_moyi_12ch"
os.makedirs(dst_m12, exist_ok=True)

target_30 = [
    55, 128, 129, 22, 115, 116, 27, 172, 60, 8,   # Top 10
    46, 30, 133, 77, 164, 59, 91, 38, 160, 120,   # Mid 10
    94, 17, 43, 2, 123, 173, 177, 13, 96, 49      # Worst 10
]

print("=== 复制 48_moyi_12ch 的 30 个目标字符 ===")
for idx in target_30:
    src_f = os.path.join(src_m12, f"g{idx}.png")
    dst_f = os.path.join(dst_m12, f"{idx}.png")
    if os.path.exists(src_f):
        shutil.copy2(src_f, dst_f)
    else:
        print(f"  ⚠ 缺失 moyi_12ch 样本 {idx}")

print("=== 检查 assets/eval200_30_gathered 下所有模型 ===")
base = "assets/eval200_30_gathered"
for m in sorted(os.listdir(base)):
    mp = os.path.join(base, m)
    if os.path.isdir(mp):
        files = [f for f in os.listdir(mp) if f.endswith(".png")]
        print(f"  [{m}]: 包含 {len(files)}/30 张目标 PNG")
