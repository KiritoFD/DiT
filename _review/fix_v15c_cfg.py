import json
import os

os.chdir("/root/Workspace/xy/DiT")
p = "src/train/configs/v15c_fixed.json"
d = json.load(open(p, encoding="utf-8"))

d["ckpt_every"] = 5000          # 用户要求: 保持 5k
d["epoch_steps"] = 5000         # 必须与 ckpt_every 相等
d["global_batch_size"] = 240    # 260 -> 240（显存 97.8% 太紧）
d["in_mem_eval_batch"] = 16
d["in_mem_eval_vae_batch"] = 16

json.dump(d, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("  ✓ 配置已更新:")
print("    ckpt_every       =", d["ckpt_every"])
print("    epoch_steps      =", d["epoch_steps"])
print("    global_batch_size=", d["global_batch_size"])
print("    in_mem_eval_batch=", d["in_mem_eval_batch"])
print("    in_mem_eval_vae_batch=", d["in_mem_eval_vae_batch"])
