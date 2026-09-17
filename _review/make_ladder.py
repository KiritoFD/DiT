import json, os

CFG_DIR = r"G:\GitHub\DiT\src\train\configs"
SRC = os.path.join(CFG_DIR, "v12_pretrain_S_cat_fame_kxl_tj_px60.json")
base = json.load(open(SRC, encoding="utf-8"))

# 探针阶梯: 全部与 v12 逐字段一致, **只改 model**。
# 目的 = 定位容量下界 (doc62)。数据/配方/drop/lr/batch/注入/REPA 全部不动。
LADDER = [
    ("v13_pretrain_XS_cat_fame_kxl_tj_px60",  "DiT-2Cond-XS/2",
     "P1 缩深度 12->8 (h 不变)",  "FLOPs 0.527x M/2, 预期 ~7.9 step/s, 50k 步 ~1.8h"),
    ("v14_pretrain_S320_cat_fame_kxl_tj_px60", "DiT-2Cond-S320/2",
     "P2 缩宽度 384->320 (d 不变)", "FLOPs 0.549x M/2, head_dim 仍 64"),
    ("v15_pretrain_XS6_cat_fame_kxl_tj_px60",  "DiT-2Cond-XS6/2",
     "P3 深度再降到 6",           "FLOPs 0.401x M/2, 定位深度下界"),
]

for name, model, what, note in LADDER:
    d = json.loads(json.dumps(base))
    d["experiment_name"] = name.replace("_", "-")
    d["results_dir"] = f"assets/results/{name}"
    d["model"] = model
    head = (f"\n==================== {name} (2026-09-16) ====================\n"
            f"[定位] 容量探针: {what}。相对 v12 (S/2 + factorized_cat + glyph_vec)\n"
            f"       **只改 model 一个字段**, 其余(数据/drop/lr/batch/glyph 注入/REPA)\n"
            f"       逐字段一致 -> 干净的容量单变量。\n"
            f"[规模] {note}\n"
            f"[依据] docs/system/62_param_budget_derivation.md —— 三条独立估计\n"
            f"       (gap 减半目标 ~23.5M / params-per-sample 启发式 14-23M /\n"
            f"        strict 对容量不敏感) 收敛到 N* ~ 15-25M。\n"
            f"[判据] 与 v12 在同 step 做**逐样本配对检验**(同 eval 样本, SE 仅 ~0.003),\n"
            f"       同时看 **LPIPS**(v12 起 in_mem_eval 已计算) 和 ssim。\n"
            f"       ⚠ 不要只看 ssim: 它会被大面积白底匹配骗过 (doc59), 15k 时\n"
            f"         v11 有字形 / v12 是墨团, ssim 却几乎相同。\n"
            f"[注] 该 model 无法从 v12 的 ckpt resume (hidden/depth 不同), 从零训练。\n")
    d["_comment"] = d.get("_comment", "") + head
    p = os.path.join(CFG_DIR, name + ".json")
    json.dump(d, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("written:", p)

print()
print("=== 校验: 三个探针相对 v12 的差异字段 ===")
diffkeys = set()
for name, model, _, _ in LADDER:
    d = json.load(open(os.path.join(CFG_DIR, name + ".json"), encoding="utf-8"))
    for k in d:
        if k == "_comment":
            continue
        if d[k] != base.get(k):
            diffkeys.add(k)
print("  差异字段集合:", sorted(diffkeys), " (应只有 experiment_name / results_dir / model)")
