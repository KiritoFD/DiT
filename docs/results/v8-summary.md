# v8 基模重构与实验线总结（2026-08-29 → 09-02）

> 语言极简 · 数据全部来自远程 eval_auto_*.json 实测 · 图见文末 grid

## 1. 训练进度（09-02 20:30）

- **v8a 基模（新，清洗数据+batch432+OT-chunks4+compile）**：step **97,400**，eval ssim **0.5045 @ 95k**（mse 1.037 / lpips 0.418），**仍在爬升**，早停 92.5k/95k 连续 NEW BEST。
- **对比旧 S30 基模（best 0.4841 @ 130k）**：v8a 用 **73% 步数**拿到 **+0.020 ssim**，领先一个身位。
- 预计 100k+ 触发早停 → 自动进 B 段 skel-ctrl。

| step | v8a ssim | v8a mse | v8a lpips | 旧S30 ssim(同step) |
|---|---|---|---|---|
| 80k | 0.5015 | 1.044 | 0.420 | 0.472 |
| 90k | 0.5037 | 1.039 | 0.419 | 0.474 |
| **95k** | **0.5045** | 1.037 | **0.418** | 0.474 |

## 2. 时间线：模型 / infra / 数据改动与有效性

### A. 基模（模型）
| 时间 | 改动 | 结果 | 有效? |
|---|---|---|---|
| 08-29 | S21 fame 预训练 | base ssim 0.4670@40k | 基线 |
| 09-01 | **S30**：DINO真迹字表+char_proj=mlp+freeze+char强化，batch384+compile | base ssim **0.4841@130k**（+0.017） | ✅ |
| 09-02 | **v8a**：清洗数据重编码 latent + batch432 + OT-chunks4 | ssim **0.5045@95k**（+0.020 vs S30，且省27%步数） | ✅✅ |

### B. 数据管线（数据）
| 时间 | 改动 | 结果 | 有效? |
|---|---|---|---|
| 09-01 前 | latent 未统一：`final_latents_fame_clean`(训练用) 与 `final_latents_fame`(v7写回目标) 内容不一致(diff 6.27) | 训练吃旧 latent，v7 写回打空 | ❌ 发现即弃 |
| 09-02 | **清洗 v7**：51,822 张 fame 图检测 → **21,050 张(40.6%)缺陷**（bars 11,287 张=21.8%、removed 18,506 张、翻转39） | 消除约 4 成坏样本（bars/残缺/反相） | ✅ |
| 09-02 | **v8 资产集（不覆盖旧文件）**：GPU 重编码 21,050 变更图 img+skel3+skel1 → `fame_clean_v8.npz` + `final_latents_fame_v8` + `final_skel_latents_fame_1px_v8` + `final_imgs_fame_v8`(GT) + v8 csv | 训练/GT/latent 三域严格一致，无脏图死角 | ✅✅ |
| 09-02 | eval 集：同 500 id，但 GT 指向清洗后 v8（其中 27 张 v7 修复） | eval 口径更干净，对比公平 | ✅ |

### C. infra / 训练工程
| 时间 | 改动 | 结果 | 有效? |
|---|---|---|---|
| 09-01 | s31 eval skel latent bug：`final_skel_latents_eval_1px`(501个) 覆盖 0/500 → eval 全零 | ctrl.ssim 假低 0.54，假回撤 | ❌ 修复 |
| 09-01 | 修 bug → `final_skel_latents_fame_1px`(全量) | s31 真实 **0.8081@42.5k** | ✅ 关键修复 |
| 09-02 | REPA 事后：s32 w0.1→0.8181；s32b w0.3 L2→0.8177 | ctrl 提升 +0.010，视觉 8/11 改善、噪点全降 | ✅ 但小 |
| 09-02 | REPA 长收敛 s32c 90k：峰值 0.8204@40k 后回落 | **base.ssim 崩 0.46→0.23** | ❌ 长训伤 base |
| 09-02 | REPA 超强 s32d w2.0：早期 0.72 | 重建通道被挤崩 | ❌ 杀掉 |
| 09-02 | **OT 优化**：exact 匈牙利 CPU 13ms@384 → **ot_chunks 分块 4ms**（每块内仍精确）；Sinkhorn GPU 实测更慢(40ms)且质量-4.8%；curegot 需 py3.11+torch2.10 不兼容 | 大 batch 下 OT 开销降 70%，无质量损失 | ✅✅ |
| 09-02 | **batch 384→432** + compile（实测 peak 19.75G@24G 安全，480 OOM） | 吞吐 1769 img/s（vs 旧 1632），OT 匹配更准 | ✅✅ |
| 09-02 | **compile 缓存持久化**：`TORCHINDUCTOR_CACHE_DIR` 固定（默认 /tmp 进程即丢） | 避免反复编译，B/C 段同 shape 秒加载 | ✅ |

### D. 结论（有效排序）
1. **清洗 + 数据域统一（v8 资产集）**：最大单项收益，基模 +0.020 且省 27% 步数。
2. **eval bug 修复**：让所有后续对比口径可信（假 0.54 → 真 0.8081）。
3. **OT 分块 + batch 432 + compile 缓存**：吞吐↑、显存可控、无质量损失。
4. **REPA 正确用法**：w≈0.3 + 短训（≤20k），长训/超强均伤 base。
5. **无效/有害**：Sinkhorn（慢且差）、curegot（环境不兼容）、ONNX CPU encode（无增益）、REPA 90k 长收敛、w2.0。

## 3. 样例 grid（基模对比：GT | S30-old@130k | v8a@95k）

![grid_all](assets/v8_grid/grid_all.png)

### 逐行
| 行 | 图 |
|---|---|
| id 0 | ![row0](assets/v8_grid/row_0.png) |
| id 1 | ![row1](assets/v8_grid/row_1.png) |
| id 10 | ![row10](assets/v8_grid/row_10.png) |
| id 100 | ![row100](assets/v8_grid/row_100.png) |

> 观察：v8a 笔画更实、背景更净、边缘利落（对应 lpips 0.418 < S30 的 0.434、tv/saltpepper 类噪点指标全面更低）。

## 4. 产物位置
- eval 汇总 CSV：`_ot_scratch/v8_dash/evals_summary.csv`（182 行，7 组实验）
- HTML dashboard（steps×eval，chartjs 自包含）：`_ot_scratch/v8_dash/v8_dashboard.html`（需与 chart.umd.min.js 同目录）
- grid 原图：`_ot_scratch/v8_dash/montages/`
