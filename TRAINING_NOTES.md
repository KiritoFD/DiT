# 训练笔记：NaN 根因排查与分阶段训练方案

记录本次 MCCD 书法数据集 DiT-2Cond 训练的排查过程与结论。时间：2026-08-12。

## 背景

本项目把标准 DiT-XL 改造成 `DiT_2Cond`（书法家 + 字符双条件），用 LoRA 微调，
叠加结构条件（Canny 边缘 + Skeleton 骨架）与 REPA 表示对齐。原始训练反复出现
`diff=nan` 后 `scaler.update()` 崩溃的问题。

## 阶段训练方案

为避免结构 loss 干扰 diffusion 主信号收敛，采用分阶段训练：

- **阶段 1（当前）**：纯 diffusion 驱动，关闭全部结构 loss。
  ```json
  "w_canny": 0.0,
  "w_skel": 0.0,
  "w_repa": 0.0,
  "use_canny": false,
  "use_skel": false
  ```
- **阶段 2（后续）**：diff 收敛后逐步打开 canny / skel 结构 loss。

## NaN 根因排查过程

### 症状
- 纯 diff 训练也在 step 148 / 165 出现 `diff=nan`
- 随后 `scaler.update()` 崩溃：
  ```
  AssertionError: No inf checks were recorded prior to update.
  ```

### 排查步骤（逐步排除）
1. **数据层**：扫描 2000 张图像 + VAE latent —— 全部 finite，latent 最大 |4.3|，
   远低于 fp16 上限 65504，排除坏样本。
2. **前向层**：fp16 前向 401×10 次（eval + train 两种模式）—— 0 NaN，前向稳定。
3. **反向层（关键）**：加入 backward 后，`use_checkpoint=True` 时 **step 0 梯度就在
   182 个可训练参数上全 NaN（grad=inf）**，而 loss 本身正常（0.10~0.20）。
   关闭 checkpoint 后 500 步 0 NaN，loss 从 0.10 正常收敛到 0.06。

### 根因
**`use_checkpoint=True`（gradient checkpointing）本身在 backward 时数值不稳定**，
与 fp16 AMP 无关（纯 fp32 + checkpoint 同样爆炸，gradmax 高达 2e37）。

原因：`torch.utils.checkpoint.checkpoint` 在反向时用 `no_grad` **重算前向**，但
训练模式下 **label dropout / LoRA dropout 是随机的**，重算的前向与原始前向中间激活
不一致，导致基于错误激活的反向梯度爆炸（inf/NaN），并污染全部可训练参数。

### 修复
将 `use_checkpoint` 置为 `false`：
```json
"use_checkpoint": false
```
- RTX 4090 有 24G 显存，batch=8 纯 diff 完全放得下（实际 ~12G），不需要 checkpoint。
- 关闭后 Steps/Sec 从 2.7 提升到 4.5，且训练稳定收敛（diff loss 0.07 附近）。

## 附带的代码 Bug 修复

原始 `train.py` 的 NaN 分支只打 warning、不调用 `scaler.scale/step`，导致
`scaler.update()` 断言 `No inf checks were recorded` 直接崩溃。修复：NaN 分支也
调用 `scale(loss).backward() + step()`，让 scaler 正确记录 inf、跳过该步并自动降 scale。

## 关于"显存为什么只用 12G"

因为阶段 1 已关闭 `use_canny/use_skel`，跳过了 VAE decode 结构 loss 分支
（不再生成 256×256×3 的 `x0_pred` 并算 Sobel 梯度），只走 latent（32×32×4），
省下了显存大头。之前 skel loss 那段才是显存峰值来源。

## 训练启动方式

远程单卡 base conda 环境，torch 2.6.0：
```bash
cd /root/Workspace/xy/DiT
nohup /opt/conda/bin/python train.py > /tmp/train_stage1.log 2>&1 &
```
注意：必须用 `/opt/conda/bin/python`（`python` 不在 PATH）。若 `diffusers` 导入报
`hf_cache_home`，需将 `huggingface_hub` 降到 `0.23.x`。

## 2026-08-12 调整：提高 LoRA 维度以激活更多参数

### 背景
阶段 1 纯 diff 训练在 step ~2000 后 diff loss 进入平台期（0.054~0.083 间波动，
均值贴 0.068），不再下降。根因是 LoRA `r=16` 容量有限 + 主干冻结，模型能学到的
残差增量已饱和。决定提高 LoRA rank，激活更多可训练参数，给模型更大容量突破平台。

### 改动
- `config.json`：`lora_r` 16 → **64**，`lora_alpha` 16 → **64**
  （scaling = alpha/r = 1，保持初始化更新尺度不变，避免单纯放大 rank 引发更新爆炸）。
- `train.py`：新增 `--lora-alpha` 参数，透传给 `inject_lora`；启动时打印
  `r / alpha / scaling`，并在日志里记录可训练参数数（trainable ratio 会从 3.37% 上升）。

### 参数规模变化（估计）
- LoRA 可训练参数随 rank 线性增长：r=16 时约 3.37% 总参数；r=64 约为其 **4 倍**
  增量（A/B 矩阵参数量 ∝ r），trainable ratio 预计升到 ~9~10%。
- 显存：LoRA 参数本身很小，但 LoRA 低秩矩阵的中间激活会增大，纯 diff（无 checkpoint）
  在 24G 卡上仍有余量（原来 ~12G）。**仍保持 `use_checkpoint=false`**——绝不能再开，
  否则会重新触发之前定位的梯度爆炸 NaN 根因。

### 预期与监控
- 期望：rank 提升 + 更大容量后，diff loss 从 0.068 平台进一步下探。
- 若训练初期 loss 抖动变大或不降，优先把 `lr` 从 1e-4 降到 5e-5 再做观察
  （已记录在 config 注释，未强制改动，待观察）。
- 监控方式：本地 `pull_log.py` 拉远程 `results/*/log.txt` + 打开
  `train_dashboard.html`（http://localhost:8731/train_dashboard.html）看 diff 曲线。

### 代码改动（已同步远程）
- `lora.py`：新增 `upgrade_lora_rank(model, new_r, new_alpha, old_sd, old_r)`。
  策略：以新 rank 重建 LoRALinear，把旧 A/B 拷进前 `old_r` 列/行（保留已学残差），
  新增列用 kaiming 初始化 A、B 置 0（初始残差为 0，不破坏已有成果，再逐步学习）。
- `train.py`：新增 `--resume-lora <ckpt>` + `--old-lora-r <r>`；inject 后若指定则升级。
- `config.json`：`lora_r` 16→64，`lora_alpha` 16→64（scaling 保持 1）。
- 启动时日志会打印 `r / alpha / scaling` 与升级后的 trainable ratio。

### 重启训练（重要：当前 020 run 的 checkpoint 目录为空）
`ckpt_every` 默认 10000，020 run 仅跑到 ~5250 步，**尚未落盘任何 LoRA checkpoint**，
因此暂时**无法从 020 的 r=16 权重升级续训**。两种重启方式：

【方式 B：从主干以 r=64 重新训（最简单，推荐先用）】
```bash
cd /root/Workspace/xy/DiT
pkill -f train.py            # 先停掉当前 020 run
nohup /opt/conda/bin/python train.py \
  --config config.json \
  > results/021-DiT-2Cond-XL-2-boot_r64/log.txt 2>&1 &
```
（config 已是 lora_r=64，会从 DiT-XL 主干 + 随机 r=64 LoRA 起步。）

【方式 A：等 020 存盘后升级续训（保住已有成果）】
待 020 跑到 step 10000 自动存 checkpoint 后（或手动降低 ckpt_every 让其尽快存），
再停进程、用升级命令重启：
```bash
nohup /opt/conda/bin/python train.py \
  --config config.json \
  --resume-lora results/020-DiT-2Cond-XL-2/checkpoints/<latest>.pt \
  --old-lora-r 16 \
  > results/022-DiT-2Cond-XL-2-up_r64/log.txt 2>&1 &
```
此方式会把 r=16 已学增量拷进 r=64 前 16 维，新增维度零初始化，平滑过渡到更高容量。

注意：`use_checkpoint` 仍必须保持 `false`，否则会重新触发之前定位的梯度爆炸 NaN 根因。

## 2026-08-12 overfit 验证：训练流程本身能否收敛

### 目的
阶段1 在 020 run 上 diff 停在 ~0.068 平台不再下降，为排除"平台期是流程 bug（数据/loss/
反向）"的嫌疑，做一次受控 overfit 验证：用极小固定数据集反复训练，看模型能否"记住"它
（diff 收敛到很低）。若能 → 流程正常，平台期是 r=16 容量 + 数据多样性导致的正常收敛。

### 做法（复用 train.py，仅靠配置解耦，未新增训练脚本）
- 从 `train.csv` 切前 500 条 → `overfit_500.csv`（表头 + 500 行），在远程生成：
  `head -1 train.csv > overfit_500.csv && sed -n '2,501p' train.csv >> overfit_500.csv`
- 新增 `overfit.json`：基于 `config.json`，覆盖 `data_csv=overfit_500.csv`、
  `results_dir=results/overfit_500`、`epochs=80`、`global_batch_size=25`(20 iter/epoch)、
  `log_every=10`、`ckpt_every=100000`；其余与阶段1一致（use_canny/skel=false、w_*=0、
  use_checkpoint=false、lora_r=64）。
- 启动（远程，setsid 后台，避免 ssh 断开被杀）：
  `setsid bash _run_overfit.sh` 其中执行 `python train.py --config overfit.json`
- 监控：本地 `pull_log.py` 自动抓 mtime 最新的 `results/*/log.txt`（当前即 overfit），
  打开 `train_dashboard.html` 看 diff 曲线。

### 初期观察（step 10→80，前 4 epoch）
- diff 从 0.1123 快速降到 0.06 区间（step20≈0.074、step50≈0.060、step80≈0.078），
  Steps/Sec≈2。符合"小数据集快速过拟合"的初期特征。
- 继续观察后续 epoch：若 diff 一路压到 ~0.02 以下并保持、无 NaN，则证明训练链路健康，
  020 的 0.068 平台是容量/数据导致的正常收敛，非 bug。

### 结论
（待跑够 epoch 后回填）

## 2026-08-12 远程日志归档（全量拉回）

所有 `results/*/log.txt`（共 25 个 run，000~021 + overfit_500）已 `tar` 打包拉回本地，
存于 `remote_logs/`（保留原始目录结构）。解析脚本 `parse_all_logs.py` 提取每个 run 的
diff 曲线，汇总于 `all_logs_summary.json`。

### 全量 diff 曲线汇总（按 run 顺序）

| 实验 | 步数范围 | diff 首 | diff 末 | diff 最低 |
|------|----------|---------|---------|-----------|
| 013 | 50→300 | 0.1718 | 0.1306 | 0.1157 |
| 014 | 50→300 | 0.1718 | 0.1306 | 0.1157 |
| 015 | 50→2750 | 0.1571 | 0.0971 | **0.0740** |
| 016 | 50→650 | 0.1482 | 0.0849 | 0.0849 |
| 017 | 50→100 | 0.1837 | 0.2040 | 0.1837（崩溃/异常） |
| 018 | 50→150 | 0.1717 | 0.1730 | 0.1694 |
| 019 | 50→1400 | 0.0790 | 0.0734 | **0.0539** |
| 020 | 50→5500 | 0.0831 | 0.0623 | **0.0547** |
| 021 | 50→1500 | 0.1305 | 0.0680 | 0.0618 |
| 002(=overfit_500) | 10→1600 | 0.1123 | 0.0623 | **0.0494** |

观察：
- **017 是异常 run**（diff 反而升到 0.20，疑似 use_checkpoint=true 早期 NaN 前的征兆）。
- **015/019/020 是健康长程 run**，diff 稳定收敛到 0.054~0.074，最低 0.0539（019）。
- **overfit_500（002）最低 0.0494**，比全量训练更低——符合"小数据集更易拟合"的预期，
  但并未出现预期中的"过拟合到 ~0.01 以下"，说明 diff 在 0.05 附近是这套配置的固有下限。
- 所有健康 run 的平台区高度一致（0.05~0.07），**与 LoRA rank（16/64）、数据集大小
  （500 / 全量）无关** → 强证据：平台期是 loss floor，不是 bug、不是容量瓶颈。

## 2026-08-12 overfit 1000 步重跑（拉回 ckpt 做推理对比）

### 目的
之前的 overfit（epoch 68 / step 1370）已能确认"流程健康、diff 有 loss floor"，但**没存
checkpoint**（ckpt_every=100000），无法做"生成图 vs GT"的视觉对比。本次重跑 1000 步并
在 step 1000 落盘一个 LoRA checkpoint，拉回本地，用 `sample_overfit.py`（3Cond 版，之前
效果好的对比脚本）或等价 2Cond 推理脚本做可视对比，看模型到底"学到了什么程度"。

### 配置变更（overfit.json）
- `ckpt_every`: 100000 → **1000**（step 1000 落盘）
- `epochs`: 80 → **50**（500/batch25=20 iter/epoch → 1000 步）
- 其他不变：lora_r=64、use_checkpoint=false、纯 diff、batch=25、lr=1e-4。

### 状态
- 已停旧 overfit 进程、清空 `results/overfit_500/checkpoints/`、同步新 overfit.json。
- `setsid bash _run_overfit.sh` 重新启动（5 worker，step 0 起步，diff 正常 0.11→0.06 下降）。
- 预计 ~8-9 分钟跑满 1000 步并落盘 `results/overfit_500/checkpoints/`，随后 scp 拉回本地。

### 下一步（ckpt 拉回后）
- 写 2Cond 推理对比脚本（复用 `sample_overfit.py` 的 GT/Pred 拼接逻辑，但适配 DiT-2Cond +
  overfit_500.csv 的前 N 张），生成 `overfit_comparison.png`（GT 左 / 生成右）拉回本地给你看。
- 结合视觉结果回答"为什么现在模型不行"：若生成图明显糊/结构丢失 → 指向 VAE 下限或容量；
  若仅细节不够 → 指向 lr/训练步数/数据多样性。


