# 53. 阶段总结与问题审查（2026-09-13）

> 范围：v10b/v11 线的模型-数据-评测工程；含**代码实现审查**与**模型层面问题清单**。

## 1. 当前状态

| 项 | 状态 |
|---|---|
| 活跃训练 | `v11_pretrain_M432_adaln4_sym`（恢复中，step ~18k，3.7 sps，17.1G）|
| 历史最佳（可部署） | `v10b_stdskel_fame3_c41x_cos_e@390k`，strict n=237 / cfg0.7 = **0.5680**，seen 0.763 |
| 次佳 | `v11_struct-loss@490k` strict **0.5675**（resume + 结构 loss）|
| 当前研究线 | M/2 (h432) + 3px skel + 12ch(canny0.3/skel0.8) + 对称增强 + REPA0.03 + adaLN4 |
| 数据备份 | `/root/Workspace/xy/dit_data_backup_20260913`（硬链接，含 5script→assets 前全部资产+results）|

### 已确立的关键结论（同口径 strict n=237 / cfg0.7）
- **容量**: S/2 30M seen 上限 ~0.52；M/2 49M seen 0.571@152k；Sp/2 59M 0.67–0.76。
- **xattn**: 短中程 strict ≈0（sp adaLN4 0.5174@80k ≈ c41x xattn12 0.518@80k），seen 仅 +0.01~0.02，代价 +33% 计算。
- **12ch 等权**会把 ~51% final-layer 梯度给结构通道（patch_embed aux 2.6×img）→ 必须 per-group 加权。
- **3px skel**: VAE round-trip IoU 0.861→0.960；字间 raw cos 0.900→0.841（centered cos 均≈0，即字特异性残差可分）。
- **数据增强 v3.3 有偏**（`target=K*(mid-w)` 把宽度分布拉向 mid）；v4 改为 **对称 ±（同幅 dilate/erode）**，单图均值不变。
- **REPA 是像素/DINO 的**（不是 latent），需 GT 图像；在 latent-only 运行里被关掉过，现在恢复。

## 2. 目录与资产（新布局）

```
DiT/
├── assets/                 # (原 5script) csv / 增强图 / results / 清单
│   ├── results/            # 保留 9 个 run (活跃+最佳+当前研究); _archive/ 归档 29 个
│   ├── train_fame3_*.csv   # 训练/评测元数据 (sym/v8/e/plus_eval)
│   └── fame3-e/ fame3-sym/ # 增强图
├── src/
│   ├── model/dit.py        # 主模型
│   ├── train/train.py      # 训练 (configs 在 src/train/configs/)
│   ├── data/vae_io.py      # 【新】统一 VAE encode/decode 轮子
│   ├── eval/               # inference.py / batch_eval.py【新】/ loop.py【新】/ posters.py
│   └── utils/latent_dataset.py
├── tools/                  # eval/data/aug/... (eval 下为 thin wrapper)
├── scripts/{ops,debug,build}
├── legacy/                 # 旧 DiT 核心/配置/脚本/数据 (只读归档)
├── REPO_MAP.md / ASSETS.md
```

## 3. 轮子（本次重构）

### `src/data/vae_io.py`（VAE 轮子）
- DataLoader 多进程并行解码（旧 builder 串行 PIL 是瓶颈）；channels_last + TF32；bf16 encode/decode；`--device cpu` CPU 模式。
- **实测 batch（独占 24.5G GPU）**：

| 操作 | batch | dtype | 峰值显存 | 吞吐 |
|---|---|---|---|---|
| encode | 128 | bf16 | **19.8G** | 160 img/s |
| encode | 144+ | bf16 | OOM | — |
| decode | 64 | fp32 | 20.0G | 50.6 img/s |
| **decode** | **64** | **bf16** | **15.7G** | **74.7 img/s** |
| decode | 96+ | bf16 | OOM | — |

  默认: encode bs=128 / decode bf16 bs=64；编解码吞吐在此配置下已不随 batch 增长（bf16 encode 平台 ~160 img/s）。
- CLI: `python -m src.data.vae_io --csv <csv> --out <dir> --transform gray|skel|canny [--skel-dilate 1] [--device cuda|cpu]`。

### `src/eval/`（评测轮子收敛）
- `batch_eval.py`：批量评测核心（g 覆盖校验 + seen/strict + 幂等 CSV + eval_auto JSON + 样本落盘）。
- `loop.py`：轮询循环；**评测时 SIGSTOP 训练 → in-mem 评测 → SIGCONT**（SIGSTOP 不释放显存，故训练 batch 192 留 ~6G）。
- `tools/eval/*` 保留同名 thin wrapper（兼容现有 tmux/脚本）。

## 4. 软件工程问题清单（按严重度）

1. **静默失效模式（本仓库惯犯，共 5+ 处）**：缺文件/零张量/None 时静默降级。
   - flow 分支曾把 `w_latent_skel/canny` 加入禁用列表 → 结构 loss 从未生效（已修，`train.py`）。
   - `latent_struct_max_t=500` 在 flow（t∈[0,1]）下恒真 → 全噪声步监督（已修 + float 类型）。
   - `make_eval_cache` 查不到 skel latent 时给零 → strict 评测"无字条件"假性崩塌（已修 + WARNING，`inference.py`）。
   - v1 字形字典不存在 → `w_glyph_cond` 全程零条件（历史）。
   - 旧 `w[-1].requires_grad_(True)` 对索引张量是 no-op → null token 实际冻结（历史）。
   **对策**：所有"可选项"在启动时打印生效状态 + 覆盖率；凡"查表失败"必须显式告警或报错。
2. **id 解析 bug**：`re.search(r"(\d+)\.png", "_b1.png")` 会把变体 id 解成 `1`；`build_data/skel/std_skel1_latents.py` 曾用**行号**当 img_id（与 dataset 的 img_id 查找不一致，已修为 img_id）。
3. **配置系统过厚**：argparse + JSON 默认注入 + 大量兼容开关（`getattr(args, ...)`）使"配置未生效"难发现；建议：**显式 dataclass 配置 + 未识别键报错**。
4. **重复实现**：`MCCDDataset` vs `MCCDLatentDataset`；CPU eval daemon / in-process auto_eval / GPU loop 三套评测；`build_*` 各自实现 VAE encode（本次收敛到 `vae_io`）。
5. **死开关**：`callig_spatial`、`callig_style_attn/style_token`、`xl_highdim`、`IDS/DINO char embedder`、`skel_head`（均为 0 效果或旧线），增加阅读/维护成本；建议移入 `legacy` 或删除（旧 ckpt 用 legacy 分支推理）。
6. **硬编码**：路径 `/root/Workspace/xy/DiT`、`structure_size=256`、`dit_batch/…` 常量散落；建议统一常量模块。
7. **无测试/CI**：`tests/` 基本空白；关键不变量（参数名、latent 形状、g 覆盖、指标口径）应有 smoke 测试。
8. **日志/可观测**：启动只打参数量；aux 权重、g 覆盖、eval 的 g 覆盖率等关键量之前不可见（本次已补）。
9. **编码/换行**：Windows 侧 CRLF 与 GBK 历史文件导致多处脚本推送后 `\r` 报错（本 session 已多次踩）；建议 `.gitattributes` 统一 LF + UTF-8。

## 5. 模型层面问题/错误清单

1. **12ch 目标与图像目标耦合**：共享 trunk + 输出头，等权时结构通道吃 ~51% 梯度（实测）；必须 per-group 权重（现 canny0.3/skel0.8，建议再扫）。
2. **VAE 对细线不保真**：1px 骨架 round-trip IoU 0.861（3px 0.960）；作为**条件 g** 与 **监督 aux** 都受影响 → 已切 3px。
3. **g 路径权重漂移**：`glyph_scale`（token-add）随训练 0.6→0.10~0.19；g 主要经 adaLN 注入（`glyph_injections` |w| 14–22）。即"条件在换通路"，需持续监控而非误判为失效。
4. **单点身份**：`no_char_cond=true` 时字身份**只来自 g**；若 g 弱/缺失则输出不可控（strict 评测曾因此假崩）。可考虑 DINO 字表作为第二身份通路（历史 s28/s30 无同口径结果；v10adino=skel+dino 且非可部署）。
5. **xattn 冗余**：strict ≈0 且 +33% 计算；当前线已用 adaLN 注入、adaLN4（省 ~17% 算力、-3M 参数）。
6. **REPA 语义**：REPA 对齐 DINO（像素），与 latent-only/12ch 运行天然不同轨；开/关要作为**配方变量**记录（不要静默）。
7. **评测口径风险**：历史 strict 数值依赖 `skel_latent_shards_dir` 的 id 覆盖；换 shard/数据集必须重新确认覆盖率（已加告警）。
8. **结构 loss（probe/StructDecoder）**已从主分支删除（验证为错误路线：像素/二值 target 与 latent 目标不匹配）；如需结构监督走 **12ch latent 目标**。
9. **容量-质量曲线**：S/2 seen 上限 ~0.52；M/2 ~0.57@150k 仍在涨；Sp/2 0.76。strict 对容量不敏感（~0.53–0.57），对**数据（增强+长训）与 resume** 敏感。

## 6. 下一步（建议）

1. **跑完 M432_adaln4_sym（到 100k/200k）**：验证"对称增强 + REPA + adaLN4 + 12ch 加权"组合；对比 `sk3@150k`（strict 0.5108）与 S/2 线。
2. **严格 A/B**: `adaLN4 vs adaLN12`（单变量，验证简化是否无损）；`aux 权重 0.3/0.8 vs 等权 vs 0`。
3. **身份通路实验**: g(3px) vs DINO 字表 vs 两者（同口径 strict）；目标是突破 0.57 的 strict 平台。
4. **工程**: 把 `legacy/eval` 旧路径清出 `tools/eval`；配置改 dataclass + 未知键报错；加 3 个 smoke 测试（模型参数名/ latents 形状/ g 覆盖）。
5. **文档**: 把上述结论同步到 `docs/system/50`（演进复盘）与 `REPO_MAP.md`。
