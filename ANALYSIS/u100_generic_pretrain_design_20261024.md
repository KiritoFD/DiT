# U100 骨干 + 风格条件化预训练 + Top10 精准后训练：设计提案（2026-10-24）

结论先行：回到 U100-Deep-Sym4 114M 骨干是对的，但直接把 generic48 的"预训练→微调"配方搬到 114M 上不会成功。
该配式的三个结构性缺陷（writer 盲预训练 / 关 STD 增强 / 微调冷启动风格通路）都有配置与指标证据，
本文档给出针对性改造：预训练阶段就带风格条件、两阶段接口零变更、2万后训练防过拟合套件。仅用 4090（24G）执行，不占用 48。

## 1. U100 / DiC 那代模型的实现核查

代码事实（`src/unet/model.py`、`src/unet/blocks.py`、`src/unet/configs/unet_deep_sym4_114m.json`、`DiC_U100_b768.json`）：

- 骨干：`LatentUNet`，SD-VAE 4×32×32 latent，FM flow-matching 目标。Deep-Sym4 为 channels 192/384/576、
  每尺度 Encoder/Decoder 各 4 ResBlock + Middle 2 ResBlock + 1 Attention（26 块，114.00M），silu，无 gate/无 stage adapter。
  DiC-768 为 num_res_blocks=2（75.83M）+ residual_cond_gate + stage adapter + gelu。
- 条件注入：**每个 ResBlock 一层 AdaGN 式 FiLM**（`film: Linear(512→2C)`，逐通道 scale/shift）。
  全局条件 `c = MLP(time128 + writer32 + script16 → 512)`，三尺度共享（Sym4 关闭 stage adapter）。
- 条件输入：STD 软骨架 64×64（三尺度投影相加进主干）+ writer-ID 共享 32D embedding（端到端训练，drop 8%）+ 书体 16D。
- 训练数据：`exp-std/csv/train_top10_aug_sym.csv` = **77,823 行 Top10 数据 + strong STD 形变增强**，从零训练。
  评测 eval200fix（187 行）。B512、LR6e-5 余弦、warmup1500、EMA.999、60M 曝光预算。
- 结果：Deep-Sym4 75k SSIM 0.6132 / 85k 0.6129 / LPIPS 0.3207（项目历史最高）；DiC-768@50k 0.6047。

**关键事实：0.613 是"writer 条件 + 强 STD 增强 + Top10 数据"三者全程共同训练的结果，不是纯架构属性。**
4090 上 DiC-768 已停于 80k（0080000.pt 保留）；Deep-Sym4 权重在 48（本轮不使用）。

## 2. 为什么 generic 预训练→Top10 后训练只有 ~0.56（证据链）

同性质留出集（Top10 书家未见字）横向对比：

| 路线 | 关键配置 | 结果 |
| --- | --- | --- |
| U100-Deep-Sym4 从零（Top10-aug 78k） | writer 端到端 + strong STD aug + 114M + 60M 曝光 | **0.613**（eval200fix 187） |
| generic48 预训练→Top10 FT | 见下 | val100：10k 0.525 / 延长至15k 0.5625 / 全解冻10k 0.5612 |
| reference 条件线（114M 同骨干） | DINOv2 参考特征单向量加法注入 | val100：0步 0.5586 → 15k 0.5661（平台） |

generic48 配置实锤（`generic48_kxl_top10_20261022/{pretrain_50k_config,top10_50k_config}.json`）：

1. **预训练 writer 盲**：`writer_mode: "none"`。100k 通用池的 636 个书家分组标签完全没用上，
   风格条件生成这个"技能"在预训练阶段不存在。
2. **微调风格通路冷启动**：FT 才切 `writer_mode: "film"`（middle+decoder 新增 rank16 FiLM，零初始化），
   仅 1000 步 adapter-only。20k 数据既要学风格通路又要适配分布。
3. **预训练关 STD 增强**：`std_shape_aug: "off"`；而 U100 线最强正则恰是 strong STD 形变增强，两阶段都没有。
4. **底子薄**：48M（96/192/256）预训练自身仅 ~0.428（1046 val），且 100k 池无审美过滤。
5. **过拟合形态**：FT 延长只涨 seen（seen/test gap 0.21+，48M 补测确认"拟合收益无法转移到留出"），
   说明瓶颈是容量分配+正则，不是步数。

reference 线额外教训：参考特征聚合成**单个 512D 向量直接加进全局条件**（`model.py:187`），
与 time/script 混在一起逐块共享，表达力最弱；SupCon 权重 0.05、636 组标签噪声大，
15k 步 SSIM 仅 +0.0075 即平台。**连续参考特征这条路当前注入方式已被证伪，不必再续 30k 之后。**

## 3. 改造方案（不照搬的部分）

### 改造 A：预训练就带风格条件（"学会写字 + 学会不同风格"）

- writer-ID 用 kxl 池现成的 source-scoped 分组（`reference_kxl_20261023/train.csv` 的 `style_writer_id`）：
  636 组中 248 组 ≥32 图、覆盖 96,659/100,095 张；其余并入 OTHER 桶 → `num_writers≈249`。
- 机制与 U100 完全一致：32D 共享 embedding 端到端训练；`drop_writer` 提到 10%（为推理期 writer CFG 留口，
  DiC 线 CFG 审计显示 writer CFG 有真实增益）。
- script 条件保留楷/行/隶三类，与 Top10 完全一致，两阶段无映射变化。
- Top10 的 10 位书家在池内就是 10 个组 ID：预训练见过、FT 直接沿用，**无冷启动、无新增模块**。

### 改造 B：两阶段接口零变更

- STD bbox sigma0 缓存 + **strong 形变增强在预训练与 FT 都开**（恢复 U100 的关键正则；
  当年关增强是为 clean-cohort 对照，不是因为增强有害）。
- VAE/latents（`generic48_kxl_top10_20261022/latents`，982M）与 STD bbox64（61M）缓存现成复用。
- writer 表：预训练端到端学出 → **FT 全表冻结**（不违反"独立表不在 FM 中训练"契约：FT 阶段表不更新）。

### 改造 C：2万精准后训练防过拟合套件

1. 全部权重自预训练终态 EMA 恢复；无新增模块、无 adapter-only 冷启动阶段。
2. backbone LR 1–2e-5 余弦短程 5–10k 步，warmup 200–500。
3. 通用回放 15–20%（generic48 为 10%，过拟合史下上调；回放含 OTHER 桶维持内容多样性）。
4. strong STD 增强持续；EMA.999；每 2.5k/5k 保存+评测。
5. 选点与停机标准写死：val100 主指标 + `train_probes`（seen 探针）gap 监控 + test400 确认；
   **seen 涨而 val 平即停**——延长步数已被证明只膨胀 seen。
6. 可选对照臂（低成本）：冻结骨干只训各 ResBlock 的 FiLM/affine + 输出层，比较过拟合曲线。

### 改造 D：闭环评测

FT 后同评四口径，第一次能直接回答"通用预训练是否优于从零"：

- val100 / test400 / train_probe：与 0.56 系（generic48、reference、cohort）同口径可比；
- eval200fix 187：与 U100 从零线 0.6132 直接可比（同集同协议同采样）。

## 4. 4090 单卡执行计划

- 现状：reference Top10 30k 在跑（18.4k/30k，1.92 step/s，约 1.7h 后自然完赛释放 GPU；不干预）。
- 规格依据：114M 同规模骨干在 24G 卡 B256 实测 22.49GiB / 1.92 step/s（reference 线），稳定可行。
- **第一轮（kxl 10万，先隔离变量）**：
  - 数据准备（CPU，可立即做）：writer 分组映射 CSV（248+OTHER+Top10 专属 ID）、复用 latents/STD 缓存、
    新建 114M Deep-Sym4 预训练 config（writer_mode=shared、drop_writer.10、strong STD aug、B256、LR6e-5）。
  - 预训练：40M 曝光 = 156k 步 ≈ **22.5h**；每 5k 评 kxl 1046（writer-given + writer-null 双口径）。
  - 后训练：10k 步 ≈ 1.5h；eval val100/test400/train_probe/eval200fix。
- **第二轮（40万全量，视第一轮结果）**：train_clean.csv 393,486 行（mccd 355k 主导、12 书体、2,255 组）；
  需扩展 STD 字库与去重/防泄漏，num_scripts 扩到 12；60M 曝光 ≈ 34h。
- 判停线：第一轮 FT 后 eval200fix **≥0.6132** 才算预训练路线胜出从零线；否则结论同样明确（容量/数据上限问题，
  转向 40万 池或承认 Top10-aug 从零是当前最优基线）。

## 5. 与历史实验的关系

- 与 generic48 curriculum 的区别：预训练带 writer 条件、开 STD 增强、114M、FT 无新模块、回放更高、表冻结。
- 与 reference 线的区别：离散分组 embedding 走 U100 已验证的注入通路（端到端学出），而非 DINO 聚合单向量加法；
  reference 线的连续特征可作后续消融（把 32D embedding 输入替换为连续风格向量），不在第一轮。
- 与 U100 线的区别：预训练数据从 Top10-aug 78k 换成通用 kxl 100k（40万），检验通用先验能否超过域内从零。
