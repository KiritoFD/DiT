#!/bin/bash
cd /root/Workspace/xy/DiT
git add src/train/configs/v31_stage1_skel.json \
        src/train/configs/v32_stage2_img.json \
        src/train/train.py src/train/cli.py \
        docs/STATUS_2026-09-30.md \
        docs/experiments/2026-09-30-stage1-skel.md \
        tools/train_skelnet_dit.py tools/gen_predskel_dit.py \
        tools/poster_skelnet_now.py tools/build_std_w7.py \
        tools/run_skel_calibration.py tools/poster_skel_variants.py
git --no-pager diff --cached --stat | tail -12

cat > /tmp/commit_msg.txt <<'MSGEOF'
两阶段: stage1 用主干架构(S/2+adaln4)预测书家骨架; stage2 加整块抹白增强

stage 1 (v31_stage1_skel): std骨架+风格 -> GT骨架 latent, 替代 SkelNet
  = v25_stdskel 逐项照搬, 唯一改动是去噪对象 shards_img -> shards_gtskel_w3
    (后者 == shards_aux_skel3 == stage2(v26) 训练时的条件本身, 产物落在分布内)
  80k 步 + 早停(patience5 x check5000)

stage 2 (v32_stage2_img): GT骨架+风格 -> 真迹图像, 从 v26@30k 续训
  保留 v26 的条件加噪, 新增整块抹白 glyph_mask_*:
  随机把 3 个 3~5 格见方的连续区域填成纯白背景 (Z_BG_VEC)。
  尺度: 1 格=8px -> 默认 4 格 = 32x32 px。
  与既有 glyph_patch_drop(逐格散点) 的区别: 这里是连续一整块,
  才对得上推理时真实的失效模式 —— stage1 预测骨架时一整段笔画没画出来。

两份更正 (前两版都写错已回滚):
  1) 不要用 xattn: HISTORY_REVIEW 记 xattn strict ≈ 0 且成本 +33% (否决项)。
     第二版照 v10b 用了 Sp/2 + xattn12 -> 75.8M 参数偏大, 就是这个信号。
     改回主线 adaln 注入 4 层, 模型 S/2 -> 36.4M。
  2) 12ch concat 输入是负面的: HISTORY_REVIEW 记 12ch 从头训 strict -0.056 /
     seen -0.144; 12ch 只有先训好 4ch 再扩通道微调才有用。

另外:
  - STATUS 文档更正: 阶段1 不是零贡献。0.5399 是坏掉的 pred 重生成路径的产物,
    正确路径(gen_predskel_dit)测得 0.6427 -> 阶段1 实际值 +0.108 (缺口的 38%)
  - train.py/cli.py: 新增 glyph_mask_* 条件抹白增强
  - tools: 新增 poster_skelnet_now / build_std_w7; gen_predskel_dit 补
    --noise-start/--patch; train_skelnet_dit 补方向/模长损失+白化+配对守门
MSGEOF

git -c user.name="KiritoFD" -c user.email="kirito@users.noreply.github.com" \
    commit -q -F /tmp/commit_msg.txt
git --no-pager log --oneline -1
