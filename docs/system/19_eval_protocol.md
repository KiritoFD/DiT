# 19 · 评测层：协议、指标与基础设施

## 1. 评测协议（统一口径，所有新实验必须遵守）

```
数据集: eval_fame_strict_clean_v8.csv (500 行, 严格凸包)
  - (书体,字) 训练出现过
  - (书体,书家) 训练出现过
  - (书家,字) 组合训练从未出现
  - 图片本身未训练
采样: Euler/Heun 50 步, 固定噪声 seed=0
CFG: 0.7 (骨架条件); 1.0 (纯 ID 条件)
批量: eval_batch 50 (GPU) / dit_batch 8 (ctrl alongside training)
指标: SSIM (中位数为主), MSE(×4), LPIPS, SkelIoU, 失败率(ssim<0.4)
分书体: 楷/行/隶/草/篆/六体 分别统计
```

### 指标使用规范
- **SSIM 用中位数**：50 步 Euler 对数值扰动混沌敏感（同 ckpt 两次评测
  均值可差 0.05），中位数稳定。
- **skel_iou 不作早停/决策依据**：对模型差异不敏感（0.018-0.024）。
- **逐实例 SSIM 是有实例条件时的指标**：纯 ID 条件下天花板 ≈
  同写本真迹互比 0.563；零样本场景应看字形正确率 + FID。

## 2. 指标天花板（实测）

| 参照 | SSIM | 说明 |
|---|---|---|
| VAE 重建 | 0.962 | 编码器底噪 |
| 同 (字,书家) 真迹互比 | **0.563** | 逐实例无条件下不可超越 |
| 1-NN 检索同字 | 0.583 | 只用 char_id 的信息上限 |
| fame-ctrl (GT 骨架) | **0.8045** | 实例条件突破天花板 |
| fame 1px ctrl @50k | **0.7974** | 持续上行 |

## 3. eval 基础设施

### universal_metrics_daemon（自动指标）
```
递归扫描 5script/results + results 全部 checkpoints/
  ├── eval_pending_ctrl_*.json → base/ctrl 对比指标
  └── eval_pending_*.json      → 预训练单图指标
产出: eval_auto_*.json (next to ckpts)
保障: eval_supervisor.sh (tmux, 崩溃自动重启)
      flock 单实例 / 失败 marker 改名防热循环 / 绝对路径
```

### 手动评测工具
| 工具 | 用途 |
|---|---|
| `tools/eval/eval_unified.py` | 多实验统一口径对比 |
| `tools/eval/make_ctrl_posters.py` | base/ctrl 海报 |
| `_diag/local_zero_shot.py` | 零样本 4 臂评测 |
| `_diag/skel_follow_test.py` | 骨架跟随 IoU |

### poster 生成
`tools/eval/make_ctrl_posters.py` → 24 样本 (6 最差 + 18 均匀)
[生成 | GT] 并排，崩溃红圈标注，统计量头部。

## 4. 评测集演进

| 版本 | 行数 | 准则 | 问题 |
|---|---|---|---|
| eval100_top30/top6 | 100 | 图级 holdout | 宽松，跨实验不可比 |
| eval500_3top30 | 500 | 同上 | 同上 |
| eval_strict_top6 | 271 | 组合泛化(宽松) | 35% 字在 top6 训练出现过 |
| eval_strict_midclean | 501 | 组合泛化 + zero-shot 字混入 | 35% 字为真 zero-shot（混合口径） |
| **eval_fame_strict_clean_v8** | 500 | **严格凸包（4 条件全满足）** | **现行** |

## 5. 历史评测数字（不可跨口径对比）

各代 eval 集难度递增、指标口径不同。跨口径对比只认可：
`5script/eval_unified_20260829.csv`（s15/s17/s18/s19 统一评测）
和 v8 系列（eval_fame_strict_clean_v8 同口径）。
