import json
import os

os.chdir("/root/Workspace/xy/DiT")
p = "src/train/configs/v15c_fixed.json"
d = json.load(open(p, encoding="utf-8"))

ch = []
for k, v in (("global_batch_size", 240),      # 260 太紧(97.8%显存)
             ("ckpt_every", 500),             # 快速出 ckpt 以便测 eval
             ("epoch_steps", 500),            # 必须与 ckpt_every 相等
             ("in_mem_eval_batch", 16),       # eval 的 dit batch
             ("in_mem_eval_vae_batch", 16)):  # eval 的 vae batch
    old = d.get(k, "<未设>")
    d[k] = v
    ch.append((k, old, v))

d["_comment"] = ("v15c + 修复数据 + SupCon 词表 (2026-09-21)\n\n"
                 "## 2026-09-21 20:37 调整\n"
                 + "\n".join(f"  {k}: {o} -> {n}" for k, o, n in ch)
                 + "\n\n" + str(d.get("_comment", "")))
json.dump(d, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("  ✓ 更新 v15c_fixed.json")
for k, o, n in ch:
    print(f"    {k:<26} {o} -> {n}")
