import json, os

CFG_DIR = r"G:\GitHub\DiT\src\train\configs"
SRC = os.path.join(CFG_DIR, "v12_pretrain_S_cat_fame_kxl_tj_px60.json")
base = json.load(open(SRC, encoding="utf-8"))

d = json.loads(json.dumps(base))
NAME = "v16_pretrain_S_cat_sep_fame_kxl_tj_px60"
d["experiment_name"] = NAME.replace("_", "-")
d["results_dir"] = f"assets/results/{NAME}"
d["glyph_embedder_sep"] = True

d["_comment"] = base.get("_comment", "") + f"""
==================== {NAME} (2026-09-16) ====================
[定位] 单变量 A/B: 相对 v12 **只改 `glyph_embedder_sep: true`**。
       模型尺寸/数据/配方全部逐字段一致。
[对照] **v12 就是本实验的对照** —— v12 是无 sep、从零训的 (step 55000 时
       strict 0.5333 / seen 0.5250)，两者可做同 step 逐样本配对检验。
[动机] FLOP 实测 (`_review/flop_savings.py`): glyph_embedder_depth=2 的两层满秩
       3x3 conv 占全模型 **9.7% FLOPs**；depthwise-separable 同感受野只需
       1/8.8 的代价 -> 省 **8.6%**，只比"直接砍掉"少 1.1%。
[风险] 这是**架构替换**。已实测 (`_review/encoder_ablate.py`, v12 ckpt @52500):
       旁路那两层 3x3 conv 造成的输出变化 = 1.619，而完全去掉 g = 1.684，
       **比值 0.961** -> 那两层不是摆设，承担了 g 编码 96% 的作用。
       但那是"现有权重下在干活"，**不等于训练时必须有**；必须重训才能判定。
       理论上低风险: 输入是近二值细线 latent，信息量极低。
[不可 resume] sep 改变 glyph_embedder 结构 (Sequential 5 层 -> 9 层)，
       权重无法迁移 -> 从零训练。v12 也是从零，故对照有效。
[判据] 与 v12 同 step 的**逐样本配对** strict/seen + **LPIPS**(v12 起已计算)。
       ⚠ 不要只看 ssim (doc59: 会被大面积白底匹配骗过)。
       期望: FLOP -8.6% -> 步速 ~+9%；质量损失若不显著即可采用。
"""
p = os.path.join(CFG_DIR, NAME + ".json")
json.dump(d, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("written:", p)

# 校验差异字段
diff = sorted(k for k in d if k != "_comment" and d[k] != base.get(k))
print("相对 v12 的差异字段:", diff)
assert diff == ["experiment_name", "glyph_embedder_sep", "results_dir"], diff
print("OK: 单变量确认")
