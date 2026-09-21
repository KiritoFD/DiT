import json
import os

os.chdir("/root/Workspace/xy/DiT")
d = json.load(open("src/train/configs/v15c_fixed.json", encoding="utf-8"))
for k in ("compile_mode", "global_batch_size", "max_steps", "ckpt_every",
          "lr", "use_ema", "in_mem_eval_sets"):
    print(f"  {k:<22} = {d.get(k)!r}")
