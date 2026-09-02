# 数据清洗管线：去噪 / 极性 / 1px 骨架（2026-08-30 ~ 09-02）

> 范围：fame 数据集（51,822 张）的噪点清洗、极性归一、1px 骨架 ControlNet 变体、
> CPU encode 级联。清洗算法经 6 轮以上「实现→实际看图→修正」迭代收敛到 v7。

## 1. 噪点分类（全量扫描实证）

对 51,822 张 fame 图逐张连通域分析（`/tmp/fame_full_scan.jsonl`）：

| 类别 | 判据 | 规模 |
|---|---|---|
| A. 赃物（笔画外小连通域 <80px） | small_frac | >1%: 4,686 张；>3%: 1,594 |
| B. 贴边黑条/黑框 | 非主体组件触边且边带占比高 | >10%: 2,677；>20%: 786 |
| C. 反色拓片（黑底白字） | 边带中值 <128（后述判据演进） | 896+32 |
| D. 石花纹理（拓本背景斑驳） | 高 small_frac 的拓片 | 与 B/C 重叠 |

**严重图合计 ~3,946 张（7.6%）**。目检确认的关键案例：園/窟/至（反色拓片）、
向/隅（上下黑条）、下（笔画触边+装订线）、江/旗/从（石花纹理网）。

## 2. 迭代史（每轮实际看图验证，教训即规格）

| 轮 | 方案 | 结果 |
|---|---|---|
| v1 | 删小件 + 删触边大连通域 | **反色图被清空；触边字被整字删除**（笔画与装订线连成一体） |
| v2 | 极性归一（cleaner 内特调）+ 抹 14px 边带 + 保主体 | 反色/边条修好；但「隅」的阝旁被贴边大块规则误删 |
| R1 | 原生分辨率 + 白底 letterbox 探测 | **发现旧管线先按字裁剪**——fame 256 图是裁剪后拉伸，letterbox 仅适用于未来重渲染 |
| R4 | opening(5×5) 分离纹理 | **倒退**：细笔画被 opening 破坏，字被删光 |
| R5 | Otsu + 滞后二值化（强核+连通弱部） | 纹理拓片大幅改善（甾/闬 提取成功）；全弱笔画被丢弃 |
| **v7（现行）** | 极性(黑组件>55%→反色) → **切边条**（边行黑>50% 连续白化，上限25%）→ 外框3px白化 → 删小件。**无缩放/无裁笔画** | 全部案例通过；全量 51,822 张执行完毕，0 错误 |

### 关键教训
1. **连通域分析对「笔画触边」失效**：装订线与笔画粘连成一个组件，
   任何「删触边组件」规则都会毁字 → 用「抹边带/切边条」代替「删组件」。
2. **opening 会破坏细笔画** → 弃用；纹理用「滞后二值化」在灰度层解决。
3. **极性判据不可用 ink 占比**（白边稀释、浓墨白底字误判），
   最终以「数据层翻转 + 人工标注兜底」解决（见 §3）。
4. **每个批量修改必须先出 before/after 图实际看**（本轮两次事故均为未先看图）。

## 3. 极性处理规范（最终）

1. 数据层翻转：`tools/` 内各 pass（ink>0.5 → 97 张；band-median → 896 张
   中大部分为「白底+黑边条」**误翻**，已恢复；人工标注 39 张）。
   累计翻转备份：`flip_backup_fame{2,3,4,5}/`、`flip_backup_manual/`。
2. 残留反色（宽白边拓片，ink≈0.5 边界）：**人工看图标注**，逐张翻转。
3. 更稳健的自动判据（推荐后续采用）：
   `黑底 ⇔ 最大黑组件面积 >55% 且该组件触 ≥3 边`；或白域封闭性测试
   （白色区域完全不触边 ⇒ 反色）。两者取或，边界情况人工复核。

## 4. 目录约定（重要）

| 目录 | 内容 | 用途 |
|---|---|---|
| `final_imgs_256/` | **共享 GT**（fame 之外的数据集未动） | REPA 等 GT 图训练 |
| **`final_imgs_fame_v8/`** | **fame 图像现行位置（51,822 张，v8 数据集 img_root）** | v8 训练 |
| **`final_latents_fame_v8/`** | **img latents（20 shards，v8 构建）** | v8 训练 |
| **`final_skel1_fame_v8/` + `final_skel_latents_fame_1px_v8/`** | **1px 骨架 PNG + latents（v8 构建）** | v8b ctrl 条件 |
| `final_imgs_256_v7backup/` | v7 清洗前原图备份（51,822） | 回滚/对照 |
| `final_skel_latents_fame_std/` | 标准字库骨架 latents | std 变体（未训通，归档） |
| ~~`final_latents_fame/`、`fame.npz`、`final_skel{1,3}_fame/`、`final_skel_latents_fame[_std]/`、`final_imgs_fame_clean/`~~ | **已删除**（09-03，被 v8 目录取代；清单见 15_progress §五） | — |

注意：
1. 清洗后图像已变化 ⇒ **img latents / 骨架 / 骨架 latents 必须级联重编码**（见 §5）。
2. `fame-ctrl`（0.8045）训练于 09-02 上午，含约 896 张误翻图 ≈1.7% 污染——
   指标仍有效（占比小），V8-B 训练已使用修正后数据。
3. `skel_bank_{train,std}.npz` 中被清洗/翻转 id 的条目基于清洗前图，**未同步**，
   推理用途影响极小（骨架结构不变），如需精确可重算。

## 5. CPU encode 级联（`/tmp/cpu_encode_v7.py` → tmux `cpu_encode`）

- 触发：v7 后变更图 21,050 张（切条/去赃/翻转）+ 人工翻转 39 张。
- 设计：CPU VAE encode（RAM 251G，batch 256，DataLoader 24 workers 预解码），
  **不占 GPU**（与 1px 训练并行）；GPU 版此前已处理人工 39 张。
- 级联内容：① 骨架 PNG 重算（1px+3px）② img latents → `fame.npz` +
  `final_latents_fame/` ③ 3px skel latents → `final_skel_latents_fame/`
  ④ 1px skel latents → `final_skel_latents_fame_1px/`。
- 坑：DataLoader collate 将 numpy 转 tensor（astype 前需 `.numpy()`）；
  CPU 上 `.to(torch.float16).cpu().numpy()`。

## 6. 1px ControlNet 变体（`ctrl_fame_1pix_v1`）

- 动机：3px 骨架太粗（用户判断），1px 保留笔画中心线。
- 数据：`final_skel_latents_fame_1px/`（1px 骨架 latent，训练/评测同域）。
- 训练：warm-start s21@30000，resume 自 22500，**max_steps 扩至 100k**（用户），
  batch 72，cfg0.7，tmux `fame_1pix_ctrl`。
- 中期指标（20k 步，resume 前）：SSIM 中位 0.7641 / MSE 0.1973 /
  SkelIoU 0.2810 —— **快于同期的 3px 版**（3px 25k 时 0.7288）。
- ⚠ 该训练已于 50k 步完成（数据为清洗前版本，含约 896 张误翻图 ≈1.7% 污染）；
  09-03 起被 V8-B（v8a 基模 + v8 净数据）取代，其 ckpt 仅作对照。

## 7. 验收指标（纳入 fame 数据管线）

| 指标 | 定义 | 阈值 |
|---|---|---|
| dirt_ratio | <80px 连通域墨迹 / 总墨迹 | >1% 告警，>5% 拒收 |
| border_junk_ratio | 边带内黑占比（切条前的边行黑占比） | >50% 连续 → 切条 |
| polarity | 边带中值 <128 或黑组件>55% → 反色；边界人工复核 | 人工兜底 |
| main_kept | 清洗后主体面积 ≥ 清洗前最大组件 95% | 必须（防过度清洗） |
| post_residual | 清洗后 sub-80px 残留 | <0.1% |

## 8. 手动审查工作流

低置信度图（极性/构图边界，1,670 张）生成带 id 标注的 contact sheet
（`/tmp/lowconf_sheet.png` 4列×24行），人工标注需反转的 (行,列) 位置清单 →
`manual_flip_denoise_encode.py` 执行翻转 + 全量去噪 + encode 级联。
本轮人工标注 39 张。自动化判据（黑组件>55%、白域封闭、边带中值）
各有盲区，**边界情况以人工为准**。

## 9. 待办

1. cpu_encode 完成后：重启 1px 训练载入净数据； Fame ctrl（GT skel）若续训同样受益。
2. 3px 管线已证劣于 1px，弃用（std 变体未训通，归档）。
3. fame_clean 目录若重渲染（letterbox），需同步重算全部 latents/骨架。
