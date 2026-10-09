import os

base = "/root/Workspace/xy/DiT/exp_milestones/eval200_outputs"
stages = ["02_v13", "03_v21", "04_v23"]
target_30 = [55, 128, 129, 22, 115, 116, 27, 172, 60, 8, 46, 30, 133, 77, 164, 59, 91, 38, 160, 120, 94, 17, 43, 2, 123, 173, 177, 13, 96, 49]

for s in stages:
    sp = os.path.join(base, s)
    if os.path.exists(sp):
        files = os.listdir(sp)
        hit = sum(1 for idx in target_30 if f"{idx:02d}.png" in files)
        print(f"[{s}]: 命中 {hit}/{len(target_30)} 张目标图片, 总 png 数: {len([f for f in files if f.endswith('.png')])}")
    else:
        print(f"[{s}]: 目录不存在")
