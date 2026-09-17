import json, io, os

SRC = r"G:\GitHub\DiT\src\train\configs\v11_pretrain_Sp2_fame_kxl_tj_px60.json"
DST = r"G:\GitHub\DiT\src\train\configs\v12_pretrain_S_cat_fame_kxl_tj_px60.json"

d = json.load(open(SRC, encoding="utf-8"))

old_comment = d.get("_comment", "")
add = """
==================== v12 (2026-09-16) ====================
[定位] 单变量实验: 只换 **模型容量 M/2 -> S/2**, 其余配方与 v11 完全一致
       (同数据 / 同 drop / 同 lr / 同 batch / 同注入层数 / 同 REPA),
       用于检验 "seen-strict gap 0.20 且 gap 在扩大" 是否由容量过拟合导致。

[动机] v11(M/2, 42M) 在 step 152500 strict 见顶 0.5656, 之后 155k/157.5k 为
       0.5631/0.5634; seen 同期 0.7691->0.7713->0.7723 仍单调上升。
       配对检验 (n=50): peak->last d_mean=-0.0022, se=0.0019, t=-1.17 -> 属噪声;
       但从 110k 起 strict 回归斜率 +0.00018/1k (t=+10.9, R2=0.87) 仍在极缓上升,
       seen 斜率 +0.00075/1k 是 strict 的 4.2 倍 -> gap 在扩大 (过拟合征兆)。
       单次 50 样本 ssim 的标准误 ~0.0126, 单点"见顶"不可信, 看趋势线。

[变更 1] model: DiT-2Cond-M/2 (h=432, ~42M) -> **DiT-2Cond-S/2 (h=384, 36.46M)**
       doc54 insight#2: 容量决定 seen 上限, 对 strict 影响小
       (S/2 30M seen~0.52; M/2 49M 0.571; Sp/2 59M 0.67-0.76; strict 各容量均 0.51-0.57)。
       若本轮 strict 未显著高于 v11 而 seen 明显下降 -> 容量不是 gap 的主因,
       应改攻"书家多样性"(当前仅 36 位书家) 与 callig 正则, 而非缩模型。
       ⚠ S/2 与 M/2 hidden 不同, **无法 resume**, 本轮从零训练;
          因此与 v11 (resume 自 10k) 不是严格同起点对照, 解读时注意。

[变更 2] condition_fusion: factorized_add -> **factorized_cat**
       + **glyph_vec_cond=true** (concat 的第二个操作数)

       ref(Moyun) 的融合是 cat([callig_emb, font_emb, char_emb]) -> Linear(3h->h),
       即**在特征维拼接多个向量因子**。我们只有一个向量因子 (callig), 直接照搬会
       退化成单层 Linear —— 实测 cat 与 add 参数量完全相同 (36.4571M), 即恒等变换。

       所以我们**在哪里 concat**: 在**条件向量 c** 上, 操作数 = {e_callig, f(g)},
       其中 f(g) 把 g_tok 的 256 个 token 池化(mean) 后投影成 128 维向量:
           c = t_emb + Linear( concat([ e_callig(128) , e_glyph_vec(128) ]) )   # 256 -> h
       实测 cond_fusion = Linear(256, 384), concat 真正非退化。

       为什么是这里 (两个理由):
        (a) 忠实于 ref 的形式 —— ref 拼的是"向量因子", 我们第二个因子只能是把
            空间条件 g 向量化, 这是唯一既非退化又对应 ref 的位置。
        (b) 补一个真实的洞: 原先 c = t_emb + callig_proj(e_callig), **adaLN 调制
            分支从来看不到内容** —— 每个 block 的全局 scale/shift/gate 都不知道
            在写哪个字; g 只经输入层 token-add 与 4 层 ZeroAdaLNInjection 进入。
       池化源用**已做 drop 掩码后**的 g_tok, 保证 drop-g 样本该向量也为零。
       无 g 时走零向量但仍过 proj, 保证 DDP 参数参与前向 (已冒烟验证 0 个无梯度参数)。

       备选(未采用): 把 g 的 256 token 拼进**序列** ([x; g] 做 self-attention 再切片)。
       它是 xattn 的加强版, 而 doc54 已实测 xattn 在 strict 上 ≈0、成本 +33%,
       期望值更低, 故列为第二选。

       ⚠ 因此 v12 相对 v11 有 **两个** 变量 (容量 M->S, 融合 add->cat+glyph_vec),
         不是单变量。若要干净拆分融合项, 需另跑 v12b = S/2 + factorized_add +
         glyph_vec_cond=true (操作数集合完全相同, 只差融合方式)。

[已知静默丢弃] repa_layers: "8" —— 该键**未注册为 argparse 参数**
       (train.py 里是 getattr(args,"repa_layers","") 读取), 因此 config 里的值会被
       静默忽略, 实际走 train.py:651 的硬编码兜底 `or (8,)`。
       本轮与 v11 一样恰好都是 8, 无行为差异; 但若将来想改 REPA 层位,
       必须先注册 --repa-layers, 否则改了不生效 (doc56/59 同一类坑的又一实例)。

[不变] 数据 assets/train_fame-kxl-tj-px60.csv (28,569, 36 书家 / 4,429 字 / 3 书体)
       —— 按"先不动数据"要求, 不使用已备好的 83,909 对称增强集。
       REPA 缓存 data/dino_cache/base_sym_v1 已验证对 px60 id 100% 覆盖。
       drop: cond_drop_all=0.10 / cond_drop_one=0.0 (保持 v11 不变, 单变量)。
"""
d["_comment"] = old_comment + add

d["experiment_name"] = "v12-pretrain-S-cat-fame-kxl-tj-px60"
d["results_dir"] = "assets/results/v12_pretrain_S_cat_fame_kxl_tj_px60"
d["model"] = "DiT-2Cond-S/2"
d["condition_fusion"] = "factorized_cat"
d["glyph_vec_cond"] = True
d["glyph_vec_dim"] = 128
d["glyph_vec_pool"] = "mean"

# 从零训练: 去掉 resume 相关字段
for k in ("resume_full", "fresh_scheduler"):
    d.pop(k, None)

json.dump(d, open(DST, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

# ---- v12b: 融合方式的干净对照 ----
# 与 v12 只差 condition_fusion (cat -> add), 操作数集合完全相同
# ({e_callig, e_glyph_vec}), 因此 (v12 - v12b) 就是纯"融合方式"的效应。
DST2 = r"G:\GitHub\DiT\src\train\configs\v12b_pretrain_S_add_fame_kxl_tj_px60.json"
d2 = json.loads(json.dumps(d))
d2["_comment"] = (d["_comment"].replace("v12 (2026-09-16)", "v12b (2026-09-16)")
                  + "\n[本配置] v12b = v12 的**融合方式对照**: 只把 condition_fusion 从\n"
                    "         factorized_cat 换成 factorized_add, 操作数集合完全一致\n"
                    "         ({e_callig, e_glyph_vec})。因此 (v12 - v12b) 即纯融合方式效应。\n")
d2["experiment_name"] = "v12b-pretrain-S-add-fame-kxl-tj-px60"
d2["results_dir"] = "assets/results/v12b_pretrain_S_add_fame_kxl_tj_px60"
d2["condition_fusion"] = "factorized_add"
json.dump(d2, open(DST2, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

print("written:", DST)
print("written:", DST2)
print()
for k in ["experiment_name", "results_dir", "model", "condition_fusion",
          "glyph_vec_cond", "glyph_vec_dim", "glyph_vec_pool", "data_csv",
          "global_batch_size", "lr", "max_steps", "w_repa", "repa_layers",
          "repa_cache_dir", "glyph_inject_mode", "glyph_inject_layers",
          "cond_drop_all_prob", "cond_drop_one_prob", "cond_drop_which_glyph_prob",
          "glyph_drop_prob", "no_char_cond", "num_calligraphers", "callig_embed_dim",
          "in_mem_eval_sets", "skel_latent_shards_dir", "eval_skel_latent_shards_dir",
          "latent_shards_dir", "resume_full", "fresh_scheduler"]:
    print(f"  {k:34s} = {d.get(k, '<absent>')}")
