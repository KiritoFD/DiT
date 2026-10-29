# PROTOCOL: S5 机制与监督可行性验证（48，24小时）

## 目标

在投入 100k+25k 正式训练前，用同起点短程实验回答四个问题：
1. E-G 的梯度能否迁移到独立 Writer 评价器（E-G2）？
2. 图像域 E-G 监督与中间特征对齐，哪个更有效？
3. S5 的局部注意力/空间调制是否存在明确的 Writer 条件改进空间？
4. 下一次正式训练应保留哪些监督和结构？

统一处理：**起点 = S5@85k EMA**（`assets/style_focus_85k_20261029/48_v3/runs/S5/checkpoints/0085000.pt`，评测已通过）；**batch 512；backbone LR 1e-5；style/adapter LR 3e-5；warmup 200；5k（D臂 2k）cosine；Decoder 冻结；EMA 0.999**。不混用起点。新权重只存 `assets/eg_pilot48_tmp/`，评测后可删。

## 阶段与门控

| 阶段 | 模块 | 产出 | 门控 |
|---|---|---|---|
| 0 链路验收 | exp0_env | environment.json | 失败即中止 |
| 1.1 E-G2 | exp1_eg2 | eg2_real_validation.json | AUC≥90 才可作独立裁判；未过则 E-G 监督结论只能算探索 |
| 1.2 梯度因果 | exp1_grad | eg_gradient_audit.csv + verdict | eg_rank 多数 t 满足 d_eg>0 且 d_eg2>0 → 才跑 B 臂 |
| 2 Writer路径 | exp2_path | writer_path_ablation.csv | 无门控（纯诊断） |
| 3 三臂 | train_arm A/B/C | train_arm_curves.csv + tmp ckpt | B 受 1.2 门控 |
| 4 Adapter | train_arm D0/D1 | 同上 | 超时则跳过，不压缩前臂 |
| 统一评测 | gen_val100/gen_swap/eval_arms | matched_eval + writer_swap.csv | — |
| 分析 | analyze | analysis_summary + FINAL_REPORT | — |

## 损失定义

- B（图像域）：L = L_FM + λ·L_rank，L_rank = CE(E-G 相似度分数：目标书家异字 top5 均值 vs 其他书家 top5 均值，τ=0.05)。每 4 步取 8 样本，ẑ0=z_t−t·v（携带梯度经 VAE Decoder → E-G）。λ 自适应：aux 梯度范数 ≈ 2%×FM 梯度范数。
- C（表征对齐）：L = L_FM + λ·(1−cos(P(pool(h_bottleneck)), EG_slot_mean(writer,script)))。λ 在 step1 用梯度范数比标定后固定。预缓存真实 bank 的 E-G 特征，训练循环内不跑 E-G/Decoder。
- 两条臂的辅助梯度均以 GradNorm 比记录（train_arm_curves.csv 的 aux 字段）。

## 验收规则（预冻结）

E-G 监督方法成立须同时满足：独立 E-G2 正向改善（≥+2pp AUC 或 Writer-Swap 方向明确改善）；内容非劣效（ΔSSIM≥−0.002、ΔLPIPS≤+0.003，报告配对 CI）；Writer-Swap 幅度更接近真实而非更极端；变化不来自近白/异常增粗。**仅 E-G 自己涨分 = 评价器定向优化，不是新方法。**

止损：梯度 pilot 不通过→跳过 B；E-G2 不通过→B/C 结果只作探索；C 对齐降而生成不变→放弃对齐；Adapter 无条件增益→不启动 S5++；两监督都失败→预算转向分层采样对照。

## 指标清单

必留：E-G AUC/EER、E-G mAP、E-G2 AUC/EER（最关键新增）、Writer-Swap Direction/Amplitude、SSIM、LPIPS、Ink-IoU、Near-Blank/Width/SoftInk；样本足够时 Style-FID/KID。Hinge/Quill 只作几何诊断，不恢复裁判资格。

## 追溯

environment.json 记录 S5/E-G 权重 SHA、数据 CSV SHA、各臂种子与配置。链日志：chain.log。执行机：48，串行脚本 run_serial.sh。
