# DiT 书法三条件生成 · 项目计划与执行总结

> 更新：2026-08-14 ｜ 目标：MCCD 书法数据集上，探索 DiT 三条件（书家×字体×字）生成的最佳「参数激活（冻结）策略」，
> 并在 10k test 上评估「是否过拟合」与「OOD 组合泛化能力」。

---

## 1. 数据与任务

- **数据**：官方 MCCD_Character 329,715 张；切分 train 318,715 / test 10,000 / eval 1,000。
- **条件**：书家 1873 类、字体 12 类（0篆 1甲骨 2印 3楷 4金 5简 6隶 7行 8六体 9草 10石 11其他）、字 7765 类。
- **条件注入**：3 个 LabelEmbedder → `cond_fusion = concat(3D) → Linear → SiLU → Linear → D` → 每层 adaLN-Zero 调制。
- **评估口径**：1k = final_eval.csv（每 ckpt auto-eval）；10k = final_test.csv（`eval_test.py` 单步重建 MSE/SSIM）。

## 2. 关键排查结论（先验事实）

| 事项 | 结论 |
|---|---|
| 官方预训练权重 | S/B/L 全 HTTP 403（不存在），**仅 XL-2-256x256.pt 可下载**（远程已有 2.6G） |
| U-DiT（ModelScope） | **否决**：U 形+卷积架构（`encoder_level_*.downsampler/dwconv`），与标准 DiT 不兼容 |
| 显存瓶颈 | 结构 loss 的像素空间 VAE decode = **15.56G**（`_mem_diag.py`），与模型尺寸无关 → **关闭结构 loss** |
| eval OOM 隐患 | `prepare_eval_cache` encode batch=64 → 16.65G；修复 encode≤16 / decode≤8 → 峰值 **6.14G** |
| 条件覆盖 | 理论组合 1873×12×7765 ≈ **1.74 亿**；训练实际去重三元组仅 **176,805** → 覆盖率 **0.01%** |

## 3. 代码改造（已完成，见 git 提交）

- **`lora.py`**：`inject_lora(..., target="all"/"attn"/"mlp")` 可选注入范围（qkv+proj / fc1+fc2）。
- **`train.py`**：冻结策略与 `use_lora` 解耦 → 三种训练模式：
  1. 从零全参（`use_lora=false` + `pretrained=null`）
  2. 预训练 body 冻结 + 只训条件头+adaLN（`use_lora=false` + `pretrained=XL`）
  3. + LoRA（`use_lora=true` + `lora_r` + `lora_target`）
  - 新增 `--lora-target`；显存日志改 `Mem: <cur>G/<peak>G`（每 20 步采样）；`ckpt-keep` 轮转防磁盘爆。
  - 修复 `_state_to_cpu` 递归（`opt.state_dict()` 嵌套 dict 的保存崩溃）。
- **`eval_auto.py`**：encode/decode batch 参数收紧（防 OOM）；eval 落盘 = 1000 张留档目录 + `eval_latest.png` 拼图。
- **`eval_test.py`**：10k test 终评（任意 ckpt → MSE/SSIM + 前 20 张对比图）。
- **运行工具**：`_launch_exp.sh`（通用启动）、`_watchdog.sh`（自动跑完实验队列：训练→Done→终评→下一个，setsid+nohup 脱离 SSH）。

## 4. 实验矩阵与结果

共用：3 条件、`use_checkpoint=false`、lr=1e-4、latent preload、ckpt schedule（前5000每1000/后每4000）、关结构 loss。
XL 组 1 epoch（39839 步）；A（S）10 epochs。

| # | 实验 | 初始化 | 激活域 | trainable | 10k test MSE | 10k test SSIM |
|---|---|---|---|---|---|---|
| A | S-scratch | S 随机 | 全参 | 36.8M | 0.02681 | 0.9042 |
| B | XL-head | XL-2 | 条件头+adaLN（无 LoRA） | 242M | 0.02544 | 0.9075 |
| C | XL-head-r8 | XL-2 | + LoRA r8 all | 246M | **0.02530** | **0.9089** |
| D | XL-head-r32 | XL-2 | + LoRA r32 all | 259M | （watchdog 自动跑） | — |
| E | XL-head-r32-attn | XL-2 | + LoRA r32 仅 attn | 253M | （待跑） | — |
| F | XL-head-r64 | XL-2 | + LoRA r64 all | 275M | （待跑） | — |

**当前结论（A/B/C 已完成）**：
- XL 预训练 body 冻结 + 条件头/adaLN（B/C）显著优于 S 从零（A），且只需 1 epoch。
- LoRA r8 比无 LoRA 略好（C > B）：低秩适配 body 有正收益，且几乎不过拟合。

## 5. 推理与 OOD 泛化发现（重要）

详见 `INFERENCE.md`。核心：

1. **单步重建 vs DDIM 自由采样**是两类难度：eval 的 MSE/SSIM 是单步重建（给 GT latent + t=150），
   自由采样（纯噪声）难得多；对从零 S，DDIM 自由采样会崩字，**不要用 eval 指标预期自由生成质量**。
2. **书家 OOD 消融**（固定字体+字，只换书家）：re-gen（见过的书家）骨架与 GT 接近；OOD（没写过该字的书家）
   生成**不稳定**——部分书家能写对（骨架笔画齐全），部分明显缺笔画/崩结构（如字㐁 的 OOD 书家 1189/1412 笔画数 30/33 vs GT 71~101）。
3. **骨架是字形结构的最严指标**：连通域数（≈笔画数）+ 墨覆盖率可量化"OOD 是否掉笔画"。

## 6. 文档索引

| 文档 | 内容 |
|---|---|
| `README.md` | 项目入口（历史） |
| `DOCUMENTATION.md` | 综合项目文档（数据/模型/训练链路，旧） |
| `plan.md` | 本文件：计划 + 执行总结 |
| `INFERENCE.md` | **推理 / OOD 验证 / 骨架分析 / 显存诊断方法论**（本次新增） |
| `MEMORY_GUIDE.md` | 显存预测公式 + 参数设置清单 |
| `HANDOVER.md` / `TRAINING_NOTES.md` / `DESIGN_REVIEW.md` | 历史交接与排查记录 |
| `dataset.md` | 数据集说明 |

## 7. 任务清单

- [x] 核实官方权重（S/B/L 403，仅 XL-2）+ U-DiT 否决。
- [x] 显存诊断 + 关结构 loss + eval 峰值修复。
- [x] 代码重构（lora target + 冻结策略）+ 配置 + eval_test.py + watchdog。
- [x] 实验 A/B/C 跑完 + 10k 终评。
- [ ] 实验 D/E/F（watchdog 自动跑，不碰远程）。
- [ ] 汇总最终对比表 + 过拟合判定。
- [ ] （后续）latent 空间结构 loss 评估（解决自由生成崩字）。
