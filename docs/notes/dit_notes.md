# 注释汇编（**参考材料**，源码中并未迁出 —— 见文末说明）

> 由 `tools/` 的重构脚本自动迁出；源码原位保留一行指针 ✓

## L2477–L2486（10 行）

```text
            # V3-A 二因子可组合 mask（4-way）：
            #   - drop_all           -> unconditional（CFG 基准）
            #   - drop_one & which=0 -> glyph-only（drop callig，学 content score s_G）
            #   - drop_one & which=1 -> callig-only（drop glyph，学 style score s_A）
            #   - 其余               -> full（callig+glyph，学 joint score）
            # 默认 0.10/0.30 配比 => full 60% / callig-only 15% / glyph-only 15% / uncond 10%。
            # 书家维度样本充足而字符维度才是难点: cond_drop_which_glyph_prob 让 drop-one
            # 偏向 glyph-only（drop callig 保 char），把专门训练预算给 5461 个字符内容分。
            # 注意: drop mask 已在 forward 顶部计算 (y_callig_in 与骨架风格注入共享),
            # 这里不再重复 roll (重复 roll 会导致两分支 mask 不一致)。
```

## L2397–L2409（13 行）

```text
            # 两类 glyph_embedder 最终都输出 (N,D,16,16) -> flat (N,256,D)。
            # ★★ 2026-10-12 修（v81 白化根因，专家定位）: **显式减去零输入响应**
            #   实测证据(probe2_glyph.py, v81 同配方):
            #     ‖encoder(0)‖ : 19.93(20k) → 34.11(35k) → 50.69(55k) → 72.07(115k)  = 3.6×
            #     内容/背景比  : 2.48 → 1.76 → 1.41 → 1.13（腰斩）
            #     同期生成墨量 : 0.174(35k，最佳) → 0.098(55k) → 0.073(115k)
            #     稀疏字不画 / 重墨字过墨（體 0.4948 vs GT 0.3559）→ 条件→墨迹映射失控
            #   机制: 编码器前两层 Conv2d 带 bias，而 `glyph_drop_prob` 默认 **0**
            #     ⇒ 训练中从未出现"无条件"状态 ⇒ 背景/偏置响应无人约束、自由膨胀
            #     ⇒ 注入主干的"空白画布特征"逐渐压过内容 ⇒ latent 空间结构消失
            #     ⇒ 解码成 DC ⇒ **白纸**（实测: latent 取均值 → 灰度 254.7 / 硬墨 0.0000）
            #   修法: feat = enc(g) - enc(0)，**不 detach 第二项** ⇒ g=0 ⟹ feat=0
            #     （即使编码器内部仍有 bias）。注意: 仅去掉 Conv bias 不够——膨胀来自深层。
```

## L2378–L2385（8 行）

```text
            # ⚠ 条件被 drop 的样本（CFG uncond 分支）拿到的是 **null 风格向量** ->
            #   形变头会产出垃圾骨架, 等于往条件里注入噪声。
            #   实测: 10% 条件丢弃时 Deform loss 从离线的 0.167 涨到 0.37
            #   (0.9*0.167 + 0.1*2.0 ≈ 0.35, 吻合) —— 正是这个原因。
            #   -> 对丢弃样本**跳过形变**, 保持原骨架。
            # ⚠ 模型上**没有** num_calligraphers 属性（只是构造形参）——
            #   用 getattr(...,0) 会拿到 0, 让这段修复变成空操作。
            #   null 索引的真值在 y_callig_embedder.num_classes。
```

## L2278–L2285（8 行）

```text
        # ⚠ 这个元组必须与**所有会读 `y_callig_in` / `char_drop` 的融合分支**一一对应。
        #   2026-09-17 修: 原先漏了 "factorized_cat" —— 该分支在下方同样复用
        #   `y_callig_in` / `char_drop`, 但因不在元组里, 整个 drop 块被跳过 ->
        #   `callig_drop is None` -> `y_callig_in = y_callig` -> **训练全程零条件 dropout**,
        #   null token 从未出现, `null_embed` 停在随机初始化。
        #   后果: 推理时 `forward_with_cfg` 的 uncond 半用的是**未训练**的 null 向量,
        #   `cfg=0.7` 实际是在往一个随机方向插值, 而不是真正的 CFG。
        #   不报错、loss 正常下降 —— 又一例静默失效。**新增 fusion 分支时必须回来加。**
```

## L2252–L2265（14 行）

```text
        # ── 标准字形条件的随机丢弃（仅训练时）────────────────────────────
        #
        # 为什么要 drop：
        # 1. 防止「门控」。标准字形 latent 与 DINO CLS 都由字 ID 推出，存在冗余。
        #    std-skel 实验已证明：冗余条件会被 ControlNet 学成恒等门控掉
        #    （SSIM 0.494≈base，SkelIoU 0.015，12 个评测点全平）。
        #    随机丢弃使 g 不可预测，模型无法依赖「反正能从字 ID 推出来」而忽略它。
        # 2. 覆盖现实：fame 上 v2 字典只覆盖 53.1%，草/篆（46.9%）本来就没有
        #    标准字形，推理时 g 只能是零。训练时模拟这种缺失，让模型学会
        #    「g 有效时跟随，g 为零时依赖书家+字条件」。
        # 3. 与 CFG 兼容：丢弃时该样本等价于 uncond-g 分支。
        #
        # 注意：以**整张样本**为单位丢弃（而非逐像素），与「某字没有标准字形」
        # 的真实情况一致。
```

## L2099–L2106（8 行）

```text
        # ── 922/80 改动 2：风格专用 adaLN 支路的 W_up 必须**重新初始化** ──
        # ⚠ 极易漏：`_basic_init` 对**所有** nn.Linear 做 xavier_uniform_，
        #   会把 DiTBlock/FinalLayer.__init__ 里刚设好的初值**整片冲掉**。
        #   （同样的坑此前已在 script_film / spatial_film / local_ca 上踩过。）
        #
        # ⚠⚠ 这里**不能**用 zeros_：W_up=0 会让 ∂L/∂W_up ≡ 0（见 T5 实测），
        #    支路永远学不动。必须用 M.STYLE_ADA_INIT_STD 的小随机初值，
        #    让梯度从头就 ∝ ‖W_up‖ ≠ 0。
```

## L2045–L2053（9 行）

```text
        # 标准字形的逐层注入必须**保持 zero-init**。
        #
        # ZeroAdaLNInjection 在构造时已把 proj 的 weight/bias 置零，但上面
        # 的 ``self.apply(_basic_init)`` 会把**所有** nn.Linear（含这些 proj）
        # 重新初始化成 xavier_uniform —— 于是注入不再是恒等映射，
        # warm-start 语义被破坏：从已训练 ckpt 续跑时，新加的注入层会给
        # 残差流注入随机扰动，等价于在训练好的模型上叠加噪声。
        #
        # 这里显式重新置零，恢复 ``out = x*(1+s)+t`` 中 s=t=0 的恒等起点。
```

## L1751–L1767（17 行）

```text
            #
            # 为什么需要逐层注入（这是本项目的关键设计修正）
            # ---------------------------------------------------------------
            # 原实现只在 **输入层** 做一次 token-add：
            #     x = x + glyph_scale * g_tok
            # 之后 x 要穿过 12 层 Transformer。每层 block 的输出都是
            # `x + block(x)`，会把 g 的相对贡献逐层稀释 —— 等于把答案写在
            # 第一页，然后让人翻完整本书再回答。
            #
            # 而 ControlNet 用的是 **逐层 zero-init adaLN 调制**（12 次），
            # 实测 SSIM 0.80。两者条件信息量完全相同（都是 4×32×32 空间图），
            # 差别只在注入方式。所以瓶颈不是「信息量不够」，而是「信息进不去」。
            #
            # 这里复用 controlnet.ZeroAdaLNInjection，与 ControlNet 对齐：
            #   out = x * (1 + s) + t，s/t 由 zero-init Linear 产出
            #   → init 时恒等，不破坏已有训练；且比加法注入表达力更强
            #     （既能增强也能抑制残差流）。
```

## L1471–L1483（13 行）

```text
                # ---- concat 融合 (ref/Moyun LabelEmbedder 式, v12) ----
                # 所有向量因子的 embedding **拼接**后, 经**一个**联合 Linear 投影到 hidden。
                #
                # 与 factorized_add 的区别:
                #   add: 各因子独立投影到 hidden 后相加 -> 因子间交互只能靠后续 adaLN 间接产生
                #   cat: Linear 一次性看到全部因子 -> 能直接建模因子间交互,
                #        表达力严格更强 (ref 即此做法: concat(3张表) -> Linear(3h -> h))
                #
                # 操作数集合 (两者一致, 便于做纯融合方式的对照):
                #   e_callig (callig_embed_dim)  [+ e_char 若 use_char_cond]
                #   e_glyph_vec (glyph_vec_dim)  若 glyph_vec_cond
                # ⚠ 若两个操作数都没有(callig 之外无因子且 glyph_vec_cond=False),
                #   cat 会退化成单层 Linear, 与 factorized_add 等价 —— 参数量完全相同。
```

## L1427–L1434（8 行）

```text
            # ---- 922/80 改动 1: 风格分支的独立 LN + 独立增益 ----
            # style_gain 与 callig_scale 是**两个不同的标量**，语义分工：
            #   callig_scale : 旧通路，乘在 callig_proj 输出上（保持历史行为不动）
            #   style_gain   : 新通路，乘在"独立 LN 之后的风格向量"上
            # 两者都启用时 y_emb = ... + style_gain * LN(callig_proj(e_callig))，
            # 即原先的 callig_scale 项保留，风格分支额外加一份 —— 因为
            # use_char_cond=False 时 callig_scale 是 y_emb 的唯一来源，
            # 直接改写它会让新旧配置无法逐位对齐。
```

## L1408–L1423（16 行）

```text
            # ── 书家/字条件的可学习幅度平衡 ───────────────────────────────
            #
            # 实测（tools/probe_condition_injection.py，s21 best ckpt，64 字）：
            #     DINO 表输出 e_char 的区分度（不同字余弦相似度） = 0.0817  <- 很好
            #     char_proj 输出                                  = 0.1189  <- 很好
            #     与书家相加后的 y_emb                            = 0.6252  <- 被淹没
            # 即 **DINO 信号本身不坏，是被书家分支的幅度压过去了**：
            #     ||callig_proj 输出|| = 12.90     ||char_proj 输出|| = 7.32
            #     比值 1.76，相加后书家主导 -> 字符区分度从 0.12 劣化到 0.63。
            #
            # 边际贡献也一致（对 adaLN shift 的相对变化）：
            #     callig 0.226   char 0.068   （字条件只有书家的 1/3）
            #
            # 这里给两个分支各一个可学习标量，让模型自行收敛到合适比例，
            # 而不是把比例硬编码成 1:1（书家与字的最优权重未必相等）。
            # 初值 1.0 保持与原实现等价，不会破坏已有 ckpt 的语义。
```

## L1381–L1390（10 行）

```text
                # DINO 384 直通：char_embed_dim == hidden_size 时，char_proj 只做
                # LayerNorm 归一化，不再 Linear 投影（省 384*384≈147K 冗余参数）。
                #
                # ⚠ 实测问题（见 docs/system/12_dino_diagnosis_20260829.md）：
                # 冻结 DINO 表的**有效秩只有 34.1 / 384**（PC1 占 26.3% 能量），
                # 83% 的最近邻落在同一书体，跨书体字符检索 top-1 仅 1.9%。
                # 也就是说字符分支拿到的信号里"书体"远多于"字符身份"。
                # 此时 char_proj 只有 LayerNorm 的 2×384 个参数，
                # **没有任何可学习容量去放大/重组那 34 个有用维度**。
                # 新配置请优先用 "mlp" 或 "full"。
```

## L1312–L1319（8 行）

```text
            # 二因子可组合条件（V3-A）：calligrapher（风格）× glyph（内容=script×char 合并类）。
            # 每个因子独立低维 embedding。
            #
            # 两种融合方式（v12 起可选，见下方 cond_fusion 处说明）：
            #   factorized_add: 各因子**独立投影**后按可学习标量相加
            #   factorized_cat: 各因子 embedding **拼接**后经一个联合 Linear 投影（ref/Moyun 式）
            # 未见的 (callig, glyph) 组合
            # 由两个各自训练充分的边际 score 组合而成，而不是靠一整张联合表 memorization。
```

## L1171–L1182（12 行）

```text
        # ---- IDS 组件码本字嵌入 (替代 LabelEmbedder) ----
        use_ids_char_embedder=False,  # 是否用 IDS 组件码本
        ids_file=None,                # IDS 字典文件路径
        char_id_to_char=None,         # dict char_id -> char (None 时假设 char_id=Unicode)
        # ---- 标准字形 DINO 字嵌入 (冻结查表, 零可训练参数) ----
        use_std_dino_char_embedder=False,  # 是否用标准字形 DINO 冻结表
        std_dino_table_path=None,          # 标准字形 DINO 表路径 (默认 _sync_work/std_dino_char_table_768.npy)
        chars_per_script=7026,             # 每个书体字符数 (glyph_id = script*chars_per_script + char_id)
        # ---- 现代化开关（v2 arch）----
        # 默认全部开启。全部关闭时与旧实现数值等价（同 seed 可复现旧结果）。
        norm_type="rms",        # "rms" | "layer"
        mlp_type="swiglu",      # "swiglu" | "gelu"
```

## L1108–L1116（9 行）

```text
        # ---- S2 (2026-09-22): 三层语义分解 + 局部风格-骨架引导 ----
        # 全部默认关闭 = 与旧 ckpt **逐位等价**，不破坏任何历史 run。
        # 设计见 docs/922/20_style_encoding.md 与 30_injection.md。
        #
        # hier_style>0 时:
        #   e_style  = y_callig_embedder(书家主效应) + α · E_pair[pair]
        #   e_script = E_script[script]
        #   书家主效应**复用既有表**（SupCon 加载 / null 行 / 冻结 / few-shot 新增行
        #   四套机制全部沿用），StyleHierarchy 只新增 pair 残差与书体两层。
```

## L1097–L1104（8 行）

```text
        # ---- 多模态风格 (v15): 每类 K 个 style token + 三种注入方式矩阵 ----
        # >0 时 y_callig_embedder 换成 MultiStyleEmbedder（查表 (B,K,D)，DINO
        # K-Means 质心初始化），token dim = callig_embed_dim（None 则 = hidden）。
        # 注入方式（从头训练矩阵 v15a/b/c，单变量对照）:
        #   a) 仅 mean pooling -> adaLN（callig_style_ca=false, style_ctx=false）
        #   b) + CalligStyleCrossAttn 书家化骨架（callig_style_ca=true）
        #   c) + K token 拼进每层 xattn context（style_ctx_every_layer=true,
        #      需 glyph_inject_mode=xattn）
```

## L1072–L1095（24 行）

```text
        # ---- 922/80 改动 2: adaLN 的**风格专用 low-rank 支路** ----
        #
        # 改动 1 解决"风格幅度太弱"，但**没解决**"风格与时间步共用同一个
        # adaLN_modulation 矩阵"这件事。D1 实测：v14_s2 明明用了三层表、
        # e_callig 区分度也好，dmod 仍只有 v13 的 1/13 —— 因为
        #   ∂L/∂y_emb = ∂mod/∂c · ∂c/∂y_emb
        # 里 ∂mod/∂c 是**同一个** W（被 t 的梯度主导），风格只是加数上的一点扰动。
        #
        # 本改动给风格一条**完全独立**的低秩通路：
        #   mod = adaLN(c)          <- 主干（t 主导，含全部旧语义）
        #       + W_up · LN(e_callig)  <- 新增，独立参数、独立梯度
        # 两者相 ``+``（不是串联），所以：
        #   ✓ 主干梯度路径完全不变（不会像"替换 adaLN"那样破坏已收敛的时序建模）
        #   ✓ W_up zero-init -> step 0 恒等，可在旧 ckpt 上 resume 继续训
        #   ✓ 风格拿到**自己的** W_up，不再和 t 抢同一个矩阵
        #
        # 参数量（S/2, D=384, r=64, 12 层 + final）：每层
        #   LN(769) + down(384*64=24576, 无 bias) + up(64*6*384=147456 + 384)
        #   = 173185 -> 12 层 + final 的 2D 版 = 约 **2.1M（+6.4%）**
        # r=32 则约 1.1M（+3.2%）。rank=0 = 关闭 = 逐位兼容旧 ckpt。
        #
        # ⚠ 本改动与改动 1 正交，可单独开也可同开。若同开，建议先确认改动 1
        #   的 5k 步判据（dmod ≥ 0.3）是否达标 —— 达标则本改动可缓，
        #   未达标（说明共享矩阵确实是瓶颈）则本改动是主力。
```

## L1037–L1068（32 行）

```text
        callig_proj_mode="linear",   # 42 号实验: "mlp" 两层 MLP 补 callig 容量
        callig_scale_init=1.0,       # 42 号实验: callig_scale 初值 (1.5 增强风格权重)
        # ---- 922/80 改动 1: 风格分支独立 LN + 独立增益（治"adaLN 饿着"）----
        #
        # 实测动机（D1，7 个 ckpt）:
        #   c = t_emb + y_emb, 而 t_emb ≈ 29–49, y_emb ≈ 16–23
        #   -> y_emb/c 只有 0.34–0.77，且与 T2 强正相关（v13 0.77 最好 / v15c 0.34 最差）
        #   adaLN 的 dmod 从 0.65（v13）掉到 0.0499（v15c, 1/13）
        #
        # 两个病因:
        #   (a) **共享投影 + 加性混合** -> 梯度被 t_emb 主导，y_emb 只贡献小扰动
        #   (b) **zero-init 死锁** -> adaLN_modulation[-1] 零初始化时
        #       ∂L/∂y_emb ∝ ∂mod/∂c ∝ ‖c‖，y_emb 越小起步梯度越小 -> 越训越弱
        #
        # 修法:
        #   style_ln=True   : 风格分支过独立 LayerNorm -> 抹平 e_callig 的范数失衡
        #                     （D5 实测 v13 系表范数 0.21…13.07，失衡 63 倍）
        #   style_gain_init : 显式放大风格分支初值，让 y_emb 起步就与 t_emb 同量级，
        #                     从而 ∂mod/∂c 变大 -> 打破 (b) 的死锁
        #
        # ⚠ **不要手动猜 gain 的值**：LN 的输出范数 ≈ sqrt(hidden)=19.6，而
        #    t_embedder 输出只有 ~0.89（实测 S/2）。gain=1.0 就已经让 y/t=21.6，
        #    gain=2.5 更是 53 倍 -> 直接把 t 条件盖掉，训练崩。
        #    所以 gain 的实际初值由 ``style_y_over_t_init`` **反解**：
        #        style_gain = style_y_over_t_init * ‖t_emb‖ / ‖LN(e_callig)‖
        #    换 hidden / 换 embedder 都会自动适配，不需要重猜魔法数。
        #    传 style_gain_init=None 也走同一条自动标定路径（更推荐）。
        #
        # ⚠ 自动标定后 y/t ≈ 1.0，即两支**同量级**、谁都不淹没谁；
        #    上限仍建议 y/t ≤ 3（再高会让模型分不清加噪程度 -> 字形崩）。
        #    用 strict SSIM ≥ 0.56 守门（见 docs/922/80_injection_redesign.md §3）。
        # ⚠ 默认 (False, 1.0) = 与旧 ckpt **逐位等价**，不破坏任何历史 run。
```

## L1016–L1024（9 行）

```text
        # [v12+] glyph_embedder 的 3x3 卷积用 **depthwise-separable** 实现。
        #
        # 动机 (FLOP 实测, _review/flop_profile.py):
        #   glyph_embedder_depth=2 的两层满秩 3x3 conv 在 h=384 上各需
        #   16*16*384*384*9 = 339.7M MACs, 合计 680M = **全模型 9.7% 的 FLOPs**。
        #   而 depthwise-separable 同感受野只需 384*9 + 384*384 = 150.9K/位置
        #   -> 38.6M, 即 **8.8x 更便宜** (节省约 8.6% 总 FLOPs)。
        #   标准 MobileNet 式分解, 感受野/表达能力基本等价。
        # 默认 False = 保持原满秩实现, 不改变任何已有 ckpt 的行为。
```

## L932–L946（15 行）

```text
        # ---- g(标准字形) 的**向量**因子 (v12) ----
        # 让 g 以"全局内容向量"的身份进入条件向量 c, 而不是只走 token-add / 逐层注入。
        #
        # 动机 (两个):
        #  1) ref(Moyun) 的条件融合是 cat([callig, font, char]) -> Linear(3h->h),
        #     即**特征维拼接多个向量因子**。我们只有一个向量因子(callig), 直接照搬
        #     会退化成单层 Linear (与 factorized_add 等价, 参数量实测完全相同)。
        #     把 g 池化成向量作为第二个操作数, concat 才真正非退化。
        #  2) 补一个真实的洞: 现在 c = t_emb + callig_proj(e_callig), **adaLN 调制
        #     分支从来看不到内容** —— 每个 block 的全局 scale/shift/gate 都不知道
        #     在写哪个字。g 只经 token-add(输入层) 与 ZeroAdaLNInjection(4 层) 进入。
        #
        # 两种融合方式共用同一组操作数 {e_callig, e_glyph_vec}, 便于做纯融合方式的对照:
        #   factorized_add: 各自投影到 hidden 后按可学习标量相加
        #   factorized_cat: 拼接后经一个联合 Linear (ref 式)
```

## L476–L485（10 行）

```text
#################################################################################
#      S2 (2026-09-22): 三层语义分解 + 局部风格-骨架引导                          #
#################################################################################
#
# 设计文档: docs/922/20_style_encoding.md (表怎么分级) + docs/922/30_injection.md
#           (adaLN vs cross-attn) + docs/922/50_implementation.md (落地说明)
#
# 一句话: 全局笔法继续走 adaLN；局部书写引导由**新增的轻量通路**承担。
#         这里**不替换** adaLN，也不再用"假风格 token"当 K/V。
#################################################################################
```


---

> ⚠️ **本文档是尝试自动迁出注释时的副产物**。实测发现本仓库的注释**承重**
> （参数默认值/开关语义大量以注释形式存在），机械搬迁会把语义一起搬走 ⇒ **已回滚，源码保持原样** ✓
> 保留本文档仅作**阅读索引**（含原始行号，便于对照）✓
