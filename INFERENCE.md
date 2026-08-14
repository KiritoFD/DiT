# DiT 推理 / 可视化 / 分析指南

> 2026-08-14 ｜ 本文档沉淀本项目**推理、OOD 泛化验证、骨架分析、显存诊断**的标准做法与踩坑记录。
> 范围仅限**本地推理与离线分析**，不涉及远程训练（远程由 `_watchdog.sh` + tmux 自动跑，见 plan.md）。

---

## 0. 两个必须分清的概念（本项目最大的坑）

**模型在训练 / eval 中算的 `MSE/SSIM`，和"从条件自由生成"不是一回事。**

| 方式 | 输入 | 说明 | 难度 |
|---|---|---|---|
| **单步重建**（`eval_auto.eval_in_memory`） | **给定 GT 的 latent** `z` + 固定 `t=150` + 固定噪声 | 只去噪一步 `pred_xstart` | 易 → 指标很好看 |
| **DDIM 自由采样**（`sample_3cond` / `_infer_*`） | **纯随机噪声** `z=randn` | 从噪声滚 N 步直到出图 | 难 → 决定性生成能力 |

**结论**：eval 的 MSE 0.025 **不代表**模型自由生成就漂亮。对**从小训练、重条件耦合**的任务，DDIM 自由采样可能崩字、OOD 更不稳。所以：
- 想验证"条件注入是否生效 / 与 loss 一致" → 用**单步重建**；
- 想验证"从无到有生成 / OOD 泛化" → 用 **DDIM 自由采样**，别拿 eval 指标预期它的质量。

---

## 1. 本地推理环境

本地（`G:\GitHub\DiT`）具备 GPU 推理能力，**不占远程卡，可在远程训练进行时并行**：

```
RTX 4070 Laptop GPU, torch 2.13+cu132, VAE 在 pretrained_models/sd-vae-ft-ema/
```

- 模型权重 `ckpt_s_scratch.pt`（S-scratch 训练好的 ckpt，441MB，已 scp 回本地根目录，git 忽略）。
- **本地无 final_images 索引数据集**（可用的是几张已拉回的 GT 图，命名 `{id}.png` / `gtimg_{id}.png`）。
- 远程推理则用远程的 `final_images|final_canny|final_skeleton`。

**加载 ckpt 的口径**：本项目 `use_lora=false`（全参/条件头）时 ckpt 的 `delta` 就是**完整 state_dict**；
`use_lora=true` 时是 LoRA 增量。统一用：
```python
ck = torch.load(ckpt, map_location="cpu", weights_only=False)
delta = ck.get("delta", ck.get("model", ck))
model.load_state_dict(delta, strict=False)
```

---

## 2. 三种推理脚本（在 `archive/probes/`，含说明）

> 这些是非正式探针脚本（本地跑，未进 git main）。用法与原理如下：

### 2.1 `_infer_ood_local.py` — DDIM 自由采样，OOD 组合生成
对 OOD_PLAN 里每个 (书家,字体,字) 三元组做 `cfg=4` 的 50 步 DDIM 采样生成图，和 GT 参考拼对比。
```bash
python archive\probes\_infer_ood_local.py   # 读 OOD_PLAN.json，输出 ood_results/ood_{i}.png
```
- OOD 三元组代表"训练从没见过的组合"，测自由生成泛化。
- 输出每张 2 拼图：左 = 模型生成(OOD)，右 = GT 参考。

### 2.2 `_infer_char_multi.py` — 固定 (字体,字)，只换书家（**书家 OOD 消融**）
严格消融"书家"这一个变量：固定 `script` + `char`，生成换不同书家（含没写过该字·该字体的 OOD 书家），
和该 (字体,字) 的真实 GT 并排。
```bash
python archive\probes\_infer_char_multi.py   # 读 CHAR_PLAN.json，输出 ood_fontchar_fixed_callimulti.png
```
- 用于回答：固定字型不变，模型能否用不同书家风格调制出同一字。
- **重要**：默认用固定随机种子的 **DDIM**；若要与 eval 对口径，改 `sample_cond` 为单步重建（见 2.4）。

### 2.3 `_analyze_skel_2175.py` — 骨架分析（字形结构的最严指标）
对固定 (script,char)，比较：GT 真实骨架 vs re-gen(见过的书家生成)骨架 vs OOD 骨架。
```bash
python archive\probes\_analyze_skel_2175.py   # 输出 skel_2175_pairs.png / skel_2175_1step.png
```
- 骨架用 `cv2` 形态学细化（无 skimage 依赖），每样本 `[原图 | 骨架]` 并列、白底黑骨、加粗可见。
- **坑**：1px 骨架在缩略图看不清（黑成一片）；务必 `dilate` 加粗 + 浅色底。
- 会打印每张的**连通域数（≈笔画数）** 与墨覆盖率，用于量化"OOD 是否掉笔画/崩结构"。

### 2.4 `_analyze_skel_2175.py`（单步重建版）
把脚本的生成函数从 `ddim_sample_loop`(纯噪声) 换成 `diffusion.training_losses(...t=150...)[pred_xstart]` + VAE decode，
即为**单步重建**，与 eval MSE/SSIM 同口径 —— re-gen 应接近 GT。这是"纠偏"单步重建 vs 自由采样混乱的范例。

---

## 3. OOD 组合 / 计划的生成（数据侧）

| 脚本 | 作用 |
|---|---|
| `archive/probes/_prepare_ood.py` | 从 `final_train.csv` 统计三元组，构造 OOD (书家,字体,字) 计划 → `OOD_PLAN.json` |
| `archive/probes/_find_char_callis.py` | 固定 (script,char) 下，列"写过该字的书家(GT)" + "没写该字的书家(OOD)" → `CHAR_PLAN.json` |
| `archive/probes/_analyze_combos.py` | 三条件纠缠度 + 实测覆盖率统计 |

**覆盖率的宏观数字（重要背景）**：
- 理论组合 = 书家 1873 × 字体 12 × 字 7765 ≈ **1.74 亿**
- 训练实际去重三元组 = **176,805** → 覆盖率仅 **0.01%**
- 绝大多数 (书家,字体,字) 组合**训练时从未见过**，所以 OOD 泛化是核心能力问题，不是边缘 case。

---

## 4. 骨架自检 & 可视化规范

- 骨架提取：`morph_skeleton`（cv2 交叉核迭代细化），再加 3×3/5×5 椭圆 `dilate` 加粗才可见。
- 展示底色用浅色(240)或白，骨架黑线；原图按原样贴。
- 标注每个格的**条件 id + 连通域数**，方便量化。

---

## 5. 显存诊断（开实验前先估）

| 脚本 | 用途 |
|---|---|
| `archive/probes/_mem_diag.py` | 分解模型/优化器/VAE/前反向/结构loss decode 的显存 |
| `archive/probes/_probe_eval_mem.py` | 定位 auto-eval（encode/decode）的显存峰值 |

结论记录在 `MEMORY_GUIDE.md`：结构 loss 的像素空间 VAE decode = 15.56G 是显存天花板；eval 的 `prepare_eval_cache` encode batch 必须 ≤16、decode ≤8。
