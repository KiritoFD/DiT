# DiT 书法三条件生成（MCCD）

用 **DiT（Diffusion Transformer）** 做「书家 × 字体 × 汉字」三条件书法字生成（256×256）。

```
输入: (书家 calligrapher_id, 字体 script_id, 汉字 character_id)
输出: 256×256 书法字图像
```

基于官方 DiT（[facebookresearch/DiT](https://github.com/facebookresearch/DiT)）改造为 3Cond 架构，支持：
从零预训练 / 官方 XL-2 预训练微调 / LoRA 容量可调 / 结构 loss（Canny+Skeleton）可选。

> 远程训练服务器：`root@10.176.54.17:36430`，项目目录 `/root/Workspace/xy/DiT`。
> 本文档仓库为本地工作副本，与远程代码保持同步（`train.py/lora.py/eval_auto.py` 等）。

---

## 文档索引（按模块）

| 文档 | 模块 |
|---|---|
| [dataset.md](dataset.md) | **数据集**：MCCD_Character 规模/切分/清洗管道/CSV 与 latent 组织 |
| [TRAINING.md](TRAINING.md) | **训练**：`train.py`/`models.py`/`lora.py`、三种冻结模式、配置字段、ckpt 调度 |
| [INFERENCE.md](INFERENCE.md) | **推理/可视化/OOD**：单步重建 vs DDIM、OOD 消融、骨架分析、显存诊断 |
| [MONITORING.md](MONITORING.md) | **监控/自动化**：`tools/pull_log.py` + dashboard、`_watchdog.sh` 自动跑实验、10k 终评 |
| [MEMORY_GUIDE.md](MEMORY_GUIDE.md) | **显存**：构成分解、实测数据、预测公式、参数设置清单 |
| [plan.md](plan.md) | **实验计划与执行总结**：容量/冻结策略扫掠结果、OOD 发现、任务清单 |
| [HANDOVER.md](HANDOVER.md) / [TRAINING_NOTES.md](TRAINING_NOTES.md) / [DESIGN_REVIEW.md](DESIGN_REVIEW.md) | 历史交接、NaN 排查、设计评审 |
| [DOCUMENTATION.md](DOCUMENTATION.md) | 综合项目文档（数据/模型/训练链路，旧版汇总） |

---

## 快速开始（本地/远程）

### 环境
- 远程 Python：`/opt/conda/bin/python`（conda，torch 2.6 + cv2 + diffusers）。
- 本地推理：`python`（torch 2.13+cu132，RTX 4070 Laptop GPU），VAE 在 `pretrained_models/sd-vae-ft-ema`。

### 训练
```bash
# S 从零预训练（关结构 loss，大 batch）
/opt/conda/bin/python train.py --config exp_s_scratch.json

# XL-2 预训练 + LoRA 微调
/opt/conda/bin/python train.py --config exp_xl_head_r32.json
```
详见 [TRAINING.md](TRAINING.md)。

### 监控
```bash
cd tools && python -m http.server 8731        # 本地起 dashboard
python tools\pull_log.py --loop --interval 60 # 每 60s 拉远程日志
```
详见 [MONITORING.md](MONITORING.md)。

### 推理 / OOD 分析
本地加载 `ckpt_s_scratch.pt` 用单步重建或 DDIM 采样生成，方法见 [INFERENCE.md](INFERENCE.md)。

---

## 关键结论速览（2026-08-14）

- **官方 DiT 预训练**：S/B/L 均无（403），仅 `DiT-XL-2-256x256.pt` 可用。
- **容量扫掠**：XL 预训练 body 冻结 + 条件头/adaLN（+可选 LoRA）远优于 S 从零；
  10k test：B(无LoRA) 0.02544 / C(LoRA r8) **0.02530**，均优于 S 的 0.02681。
- **条件覆盖**：理论组合 1.74 亿，训练实际仅覆盖 17.7 万（0.01%）——OOD 泛化是核心课题。
- **推理口径**：eval 指标是单步重建；自由 DDIM 采样难度高，OOD 书家生成骨架不稳定（详见 INFERENCE.md）。

---

*License: 项目改造自官方 DiT（MIT），MCCD 数据集版权归原发布方。*
