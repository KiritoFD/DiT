import json
import os

os.chdir("/root/Workspace/xy/DiT")
p = "src/train/configs/v15c_fixed.json"
d = json.load(open(p, encoding="utf-8"))

d["compile_mode"] = "reduce-overhead"   # default -> reduce-overhead（省显存+快）
d["global_batch_size"] = 320            # 240 -> 320（实测是否 OOM）
# 不做 expandable_segments（用户要求）

json.dump(d, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("  ✓ 配置更新:")
print("    compile_mode     =", d["compile_mode"])
print("    global_batch_size=", d["global_batch_size"])
print("    ckpt_every       =", d["ckpt_every"])
