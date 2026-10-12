# 注释汇编（**参考材料**，源码中并未迁出 —— 见文末说明）

> 由 `tools/` 的重构脚本自动迁出；源码原位保留一行指针 ✓

## L1322–L1334（13 行）

```text
                # === INFRA FIX: pred_xstart from training_losses carries the ENTIRE DiT
                # forward graph (every block's activations are kept alive because the
                # graph flows back through _predict_xstart_from_eps → model_output).
                # This is THE memory leak: the 256-token × 384-dim × 12-block activation
                # graph stays pinned until backward() runs, and struct decoders build
                # ADDITIONAL 256×256 activations on top of it. When t<=500 (~50% of steps)
                # the decoder runs → graph doubles; when t>500 the graph lingers but
                # decoders don't run → the cycling 20G→22G→14G pattern.
                #
                # FIX: extract pred_xstart WITH the graph only when we actually need it
                # for a differentiable struct loss (t<=max_t). For t>500 steps, detach
                # immediately so the full graph is freed at the next zero_grad. We also
                # break the reference in loss_dict so no stale graph survives the loop.
```

## L1293–L1300（8 行）

```text
                # Forward pass under bf16 autocast (same exponent range as fp32, no overflow).
                #
                # return_pred_xstart: flow 分支默认**不返回** pred_xstart（避免
                # autograd 图膨胀），但下面 w_std_mid 依赖它。若这里不请求，flow
                # 模式下该机制会因 `loss_dict.get("pred_xstart", None)` 恒为 None 而
                # **静默失效** —— 不报错、loss 正常下降、但从未生效。w_std_mid 正是
                # 「把去噪中段预测的 x0 拉向标准字形 latent」的预训练改进项。
                # gaussian_diffusion 不接受该参数，故用 try/except 兼容。
```

## L829–L837（9 行）

```text
        # Recover the step counter, 优先级:
        #   1) ckpt **顶层** "train_steps"  —— 唯一权威: 存盘时由真实步数写入(见下方
        #      checkpoint["train_steps"]), 与文件名无关, 任何文件名的 ckpt 都对。
        #      [2026-09-16] 此前**从未被读取**, 只靠文件名数字推测 -> 跨阶段备份用的
        #      无时间戳名字会被解析错, 例: "v12_S2_cat_last.pt" 的 digits=['12','2']
        #      -> resume_start_step=2 -> LR 退回 warmup 起点(≈6.7e-8, 看着像卡死),
        #      且 ckpt 从 0001000.pt 重新编号会**覆盖历史 ckpt**。
        #   2) 文件名末尾数字 (0055000.pt -> 55000), 兼容旧 ckpt (无顶层字段)。
        #   3) 保存的 args.train_steps —— 兜底 (历史 comment 认为该字段不存在)。
```

## L497–L507（11 行）

```text
    # 2) re-init the conditioning head (adaLN & final_layer) after loading pretrained body.
    # The pretrained adaLN was trained on ImageNet's single y_embedder; our 3-cond fused
    # condition vector c is out-of-distribution for it, which can blow up to NaN early on.
    # Re-initializing adaLN/final_layer to a small std (like the successful overfit run)
    # keeps the pretrained transformer body while letting the new condition head learn.
    # (Skipped on full resume: the delta already carries the learned adaLN.)
    # 条件调制层总是重置从头学（无论 legacy/factorized_add/xl_highdim）。
    # 关键认知：ImageNet 预训练 adaLN/final_layer/y_embedder 学的是"1000类自然物分类
    # →调制"，与书法(callig×glyph)条件完全正交。保留它= 强迫模型用 ImageNet 分类
    # 眼光生成，导致乱码/跑偏。我们只保留通用扩散引擎（t_embedder/x_embedder/attn/mlp），
    # 条件调制一律从头学，由训练目标(结构loss+diff)自行建立"条件→生成"耦合。
```

## L440–L447（8 行）

```text
                # ---- (2) 未命中行：用 DINO 均值填充，而不是留随机噪声 ----------
                # y_char_embedder 有 num_characters=35130 行，而 DINO 只覆盖 20468 个
                # glyph。未命中的行停留在 nn.Embedding 默认 N(0, 0.02) 且被冻结，
                # 对模型来说就是一个"随机字符"。
                # 更糟：char_proj='ln_only' 时 LayerNorm 逐样本归一化，把
                # "已知行范数=1.0" 和 "未知行范数≈0.39" 这个唯一可辨的线索也抹掉了
                # —— 模型在数值上无法区分。用 DINO 均值填充至少给出一个
                # "平均字形"的合理先验（eval_unseen 上有 6.6% 的 glyph 落在这里）。
```

## L369–L376（8 行）

```text
    # ── DINO glyph-embedding init for y_char_embedder ───────────────────────
    # glyph_id = script_id * 7026 + character_id (每 script 7026 个字符, 见
    # tools/remote_sync/_add_glyph_col.py). DINO vocab 是对"字*书体"(glyph) 取平均的:
    # 同一 glyph 的所有书写样本的 CLS token 平均后 L2 归一化, 维度必须 == char_embed_dim
    # (768), 之后 char_proj 直接 LayerNorm(768)->Linear(768,H) 投影, 不再经过中间 256 层。
    #
    # ⚠ 当 use_ids_char_embedder=True 时跳过 DINO 初始化:
    # IDSCharEmbedder 用部件嵌入池化, 不需要 DINO 初始化。
```


---

> ⚠️ **本文档是尝试自动迁出注释时的副产物**。实测发现本仓库的注释**承重**
> （参数默认值/开关语义大量以注释形式存在），机械搬迁会把语义一起搬走 ⇒ **已回滚，源码保持原样** ✓
> 保留本文档仅作**阅读索引**（含原始行号，便于对照）✓
