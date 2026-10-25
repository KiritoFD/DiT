# Qwen LoRA 线全记录：Top10 书法风格（2026-10-24/25）

主海报：[lora_line_poster.jpg](../assets/qwen_lora_top10_20261024/lora_line_poster.jpg)（五节视觉证据）、
[model_matrix_poster.jpg](../assets/qwen_lora_top10_20261024/model_matrix_poster.jpg)（20 字×5 模型矩阵）。
技术详档：[qwen_lora_top10_20261024.md](qwen_lora_top10_20261024.md)。机器：48（4090 48G）。

## 0. 一页结论

| # | 实验 | 结果 | 判定 |
|---|---|---|---|
| 1 | Top10 纯 Prompt LoRA 20k | test200 LPIPS −13%、Ink +45% vs 底模，SSIM −.021 | **有效，主力权重** |
| 2 | 非 Turbo 同配方 40 步 | 与 Turbo 打平 | Turbo 用 1/5 成本，弃非 Turbo |
| 3 | LoRA 强度扫描 0–1.25 | 0.75–1.0 最优，1.25 全面变差 | 部署 scale 1.0 |
| 4 | 书家笔法描述 Prompt | 46 对微降（.5708→.5678） | 文字注入笔法无效 |
| 5 | Writer-swap（30 字 + 真实组合 23 槽） | 识别 8/30 vs 机会率 5.3；seed 噪声 .57 > 书家差异 .74 | 书家控制真实但弱 |
| 6 | 推理注入 2 参考图 | 全指标降 + 目标字被污染（深→九） | 路线证伪，需带参考训练 |
| 7 | 分辨率 256 vs 512 | 打平 + VLM 复核无崩坏 | 训练切 256，快 5.2 倍 |
| 8 | D1/D2/D3 条件诊断 | Qwen 条件信息>TTF；seen/test 差距 ≤.019；书家控制弱 | 支持"作条件"的增量假设 |
| 9 | U-Net 双通道 A/C 臂（5k 步） | C−A ≈ +.001 SSIM / ±0 LPIPS / −.007 Ink | 增量在噪声内，不扩全量 |
| 10 | P1 书家残差 rank8（2 书家×3k 步） | 颜真卿 +.005、褚遂良 −.002（n=10 噪声内） | 书家专精无显著收益 |
| 11 | P2-A STD-conditioned LoRA | **进行中**（step0：SSIM .5954 已超基线） | 见 §4.3 |

一句话：Qwen LoRA 是有效的书法域生成器（结构稳、墨迹真），但**书家特异控制是它的短板**，
且这个短板无法靠更多步数、更大 rank、文字描述或推理侧参考图补齐；
作为 114M U-Net 的额外条件通道也近乎冗余。当前唯一未关闭的方向是 P2-A（字形锚定 + 原生图像条件）。

## 1. 主线：Top10 纯 Prompt LoRA

- **方法**：Qwen-Image-2.1-Turbo（7.1B）冻结，仅训 Attention q/k/v/out rank16 LoRA（16.78M，0.24%）；
  条件纯文字 prompt（`书法家：{名}。书体：{书}书。……书写汉字「{字}」`），冻结文本编码器；
  文本缓存键 =（字, 书家, 书体）17,453 个，避免旧缓存污染。
- **数据**：Top10 全量 train 23,485（楷 9,363/行 12,285/隶 1,837）/val 100/test 200，
  10 书家真实组合，无阈值清洗、不补造标签；GT 原生 256。
- **训练**：256、B12、无梯度检查点、LR 1e-4 余弦至 1e-5、20k 步 = 240k 样本 ≈ 10.2 epoch，
  1.216 s/步（峰值 44.07 GiB），总 6.7h 零重启；每 4k 保存+评测。
- **test200**：SSIM **.5821** / LPIPS **.4365** / Ink-IoU **.2827**（底模 .6032/.5027/.1948）。
  分书体：隶 Ink 最高（.3305）；分书家 SSIM .546–.629 无掉队。
- **收敛**：SSIM/Ink 8k 后平台（±.006），LPIPS 前 8k 贡献 61%、末 4k −.005；无过拟合回退；
  20k 为合理停点（加练预期 ≤.01 LPIPS）。
- **权重**：5 检查点各 64MB fp32 safetensors，已拉回本地
  `assets/qwen_lora_top10_20261024/weights/`；加载需与训练一致的 fp32 转换 + 逐张量校验。

## 2. 推理侧与条件诊断（不训练）

- **强度扫描**（val100×5 档）：SSIM .6039/.5857/.5857/.5853/.5805，LPIPS .5072/.4508/.4453/.4458/.4530
  ——0.75–1.0 最优；**1.0 保持**。
- **笔法描述 Prompt**：46 对同字同 seed 配对，加入十书家专属笔法描述后 SSIM .5708→.5678、
  Ink .279→.273——**文字描述注入笔法无效**，文本通道潜力到顶。
- **Writer-swap**：30 字×10 书家（LPIPS 质心分类，候选楷 10/行 10/隶 3）底模 5/30、LoRA 8/30、
  GT 上限 10/30；真实组合版（23 槽）换 seed .5696 < 换书家 .7441——**seed 噪声主导**。
- **参考图注入**（2 张同书家异字训练图，经 Qwen3-VL 原生图像条件）：LoRA 配对全降
  （SSIM −.012/LPIPS +.009/Ink −.025），底模同样净负（SSIM −.044）；
  **污染实锤**：目标「深」被写成「九」（见海报 D 节）——模型未分离参考的内容与风格。
- **分辨率裁决**：底模九字 256/512 对照指标打平 + VLM 复核 256 无结构崩坏 → 训练切 256，
  0.101 s/样本（比 512/B8/检查点快 5.2 倍）。
- **D1/D2/D3**（为下游条件实验立项）：
  D1 Qwen 条件 vs GT Ink .283/LPIPS .436 ≫ TTF .197/.681（SSIM 均 .582）→ 携带 TTF 没有的墨迹信息；
  D2 seed 噪声 > 书家差异（见上）；D3 train200 vs test200 差距 ≤.019 → 无条件质量悬崖。

## 3. 非 Turbo 对照

同数据同配方 20k 步（唯一差异：采样 40 步 vs 8 步）：test200 SSIM .5758 / LPIPS .4299 / Ink .2791
vs Turbo .5821/.4365/.2827——基本打平各占半项。**蒸馏无微调劣势，Turbo 以 1/5 推理成本胜出**，
后续全线以 Turbo 20k 为起点。验证趋势两轮同形（LPIPS 单调改善 16k 后平台）。

## 4. 下游条件实验（Qwen → U-Net / 自身条件增强）

### 4.1 U-Net 双通道 A/B/C 臂（Deep-Sym4 114M 85k 底座，5k 步同预算）

C 臂 = std_stem 1→2 通道（第二通道零初始化）喂 Qwen 字形（4,493 键存储，与 TTF std_soft 同协议）+
15% dropout。eval200fix（σ.65、EMA、n=187）：

| 臂 | SSIM | LPIPS | Ink-IoU |
|---|---|---|---|
| BASE 85k | .5812 | .35903 | .24548 |
| A TTF-only 90k | .5850 | **.35805** | **.24628** |
| C TTF+Qwen 90k | **.5864** | .35809 | .23966 |

C−A = +.0014/+.0000/−.0066——**噪声量级内，Ink 反降** → Qwen 字形通道对 114M U-Net 近冗余，
不扩全量（与 D2 自洽）。B 臂未训（A/C 已回答主问题）。

### 4.2 P1 书家残差 LoRA（2026-10-25）

20k 冻结（双适配器）+ rank8 残差，单书家数据 3k 步（颜真卿 2,503 张/褚遂良 2,424 张，LR 5e-5，B8）。
未见字（validation，同 seed 同协议）残差 vs 共享：颜真卿 +.0051/−.0043/−.0004，
褚遂良 −.0021/+.0020/+.0056——方向相反、均在 n=10 噪声内。
**书家联合训练不存在可测的梯度冲突红利**；瓶颈在条件信息本身。

### 4.3 P2-A STD-conditioned LoRA（进行中）

按用户路线 A：`STD 字形 + 书家 Prompt → GT`。Turbo + 20k 冻结（双适配器）+ 新 rank16，
条件 = 目标字 TTF 字形（字体级联现渲染 256 RGBA σ.65）经 Qwen3-VL 原生图像条件；
协议逐位镜像管线（spy 实测：hidden = 条件图 VAE latents 256 tok ⊕ target 256 tok、
img_mask = VL 槽 + target//4 槽、img_shapes 双段、loss 只取 target 段）。
全量 23,485 张、B4×accum4、梯度检查点、4,000 步、文本 embedding 预计算
（17,331 键 20.2GB，文本编码器训练期驻 CPU）。
**step0 参照**（STD 条件 + 未训适配器）：val100 SSIM **.5954** / LPIPS .4656 / Ink .2179
——结构锚定立竿见影（SSIM 超共享基线 .5853），墨迹待训练补回。
判定口径：2k/4k 评测 vs 共享基线 .5853/.4458/.2855 与 step0 自照；
**LPIPS/Ink 回到基线水平且 SSIM 保持 .59+ 即路线成立**。

## 5. 工程结论（对后续所有 LoRA/条件实验有效）

1. LoRA 冻结"权重更新"不冻结"计算"：每步全量过 7.1B 底模（≈0.46 PFLOP/步@B32），
   4090 在 B8 即算力打满——加大 batch 不改每样本耗时（0.524 vs 0.518 s 实测）。
2. 无检查点省 1/3 计算、激活 ×3：512/B8 与 256/B16 双双 OOM@46.5+ GiB；文本 token 不随分辨率缩
   （144 vs 336/样本）；最终 256/B12/无 ckpt/expandable_segments 43.9 GiB 稳定。
3. 双适配器（PEFT 多适配器 + set_adapter 列表）是"冻结共享 + 残差"的正解——
   `merge_and_unload` 在 transformers 集成下不可用；跨适配器加载需键名归一 + fp32 先行。
4. Qwen3-VL 原生图像条件协议（spy 实测）：hidden = 条件图 VAE latents（16×16=256 tok）⊕ target；
   img_mask = VL 槽（encode pad）+ target//4 槽；img_shapes = [条件图, target] 双段；
   条件图必须 RGBA 4 通道；文本编码器 ~16GB，批训练前应预计算 embedding 驻 CPU。
5. Turbo 8 步蒸馏调度是固定的（官方 sigmas）；改 num_inference_steps 不等于线性加采样。

## 6. 产物索引（本地 assets/qwen_lora_top10_20261024/，gitignore 保持仓库精瘦）

- 海报：lora_line_poster.jpg（五节证据）、model_matrix_poster.jpg（20 字矩阵）、
  test200_poster_20k.jpg、test200_contact_sheet_20k.jpg、res_probe_montage.png、
  d1_poster.jpg、d2_grid.jpg（T/Q/GT 与诊断图，Turbo 侧；非 Turbo 海报在 assets/qwen_lora_top10_base_20261024/）
- 指标：test200_summary_20k.json、p0_summary.json、d1/d2/d3_summary.json、
  test200_summary_20k.json（base 目录同）
- 权重：weights/turbo20k_{0004000..0020000}.safetensors（64MB×5）
- 代码：tools/qwen_lora/（训练/评测/诊断/海报全脚本）、tools/qwen_condition/（字形生成、
  48 仓库补丁器、warm-start 验证、离线评测）；远端 run 目录
  `/home/ds/Workspace/xy/qwen-lora-top10-20261024`（Turbo）、`qwen-lora-top10-base-20261024`（非 Turbo）、
  `qwen-lora-ref-20261025`（P2-A）、`qwen-writer-res-{yan,chu}-20261025`（P1）
