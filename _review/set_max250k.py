import json
import os

os.chdir("/root/Workspace/xy/DiT")
p = "src/train/configs/v15c_fixed.json"
d = json.load(open(p, encoding="utf-8"))
old = d["max_steps"]
d["max_steps"] = 250000
d["compile_mode"] = "default"      # 保持 default（实测更快+显存低）
json.dump(d, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("  max_steps:", old, "->", d["max_steps"])
print("  compile_mode:", d["compile_mode"])
print("  global_batch_size:", d["global_batch_size"])
