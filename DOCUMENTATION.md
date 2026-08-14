# DiT-3Cond 项目文档

> 最后核对时间：2026-08-14。本文档基于代码与数据全文核查，与旧文档（HANDOVER / DESIGN_REVIEW / TRAINING_NOTES）不一致处以本文为准。

---

## 1. 项目概览

本项目用 **DiT（Diffusion Transformer）** 做「按照书家风格 + 字体 + 指定汉字」的三条件条件生成：

```
输入条件： (书家 calligrapher, 字体 script, 汉字 character)
输出：     256×256 书法字图像
```

- 模型：`DiT-3Cond-XL/2`（见 `models.py`，派生自官方 DiT-XL/2 骨架）
- 训练数据：官方 **MCCD_Character**（329,715 张书法字）
- 技术栈：PyTorch + bf16 混合精度 + LoRA 微调官方预训练主干
- 附加监督：训练时额外用 **Canny 边缘图 + Skeleton 骨架图** 做像素级结构 loss，缓解扩散模型糊字、结构飘的问题。

---

## 2. 当前状态（重要）

- **远程服务器正在重建训练数据**：`train.csv / test.csv / val.csv` 当前均为 0 行（新 CSV 尚未生成），因此旧配置暂时无法直接跑完整训练。
- **远程活跃训练任务**：tmux 会话 `skel0`，配置 `train_full_3cond_skel0.json`（由 `_mk_skel0_cfg.py` 生成），启用 `use_canny + use_skel`，从 0（无预训练权重）在新数据上训练，结果目录 `new_data_skel0/results_full_3cond/`。
- 本地仓库中 `final_*.csv`、`_id_maps.json`、`final_manifest_split.json` 为清洗管道产物（新数据方案），与远程重建方案一致。
- 预训练权重（`DiT-XL-2-256x256.pt`）需自行下载放入脚本规定的路径（见 `train.py`、`lora.py` 中加载点），不在仓库内。

---

## 3. 数据集

### 3.1 官方 MCCD 数据

数据源：tencentmusic/mccd 官方数据集，仓库内路径 `MCCD/MCCD/MCCD_Character/`。

```
MCCD_Character/
├── trainset_dataset/
│   ├── train/          # 234,255 张（每个汉字一个子目录）
│   ├── test/           #  95,460 张
│   ├── train_label.txt # 234,255 行
│   └── test_label.txt  #  95,460 行
└── (其他子集见 MCCD/ 顶层：MCCD-Style / MCCD-Dynasty / MCCD-Calligrapher)
```

- 总样本：**329,715 张**（train 234,255 + test 95,460）。
- 每张图固定 512×512 灰度（训练时缩放到 256）。

### 3.2 label 行格式

每行格式（`-` 分 5 段）：

```
字-字体-朝代-出处/书家-样本id
例：竝-篆-宋-集古文韵上声韵第三-66275
```

字段解析（用于 `build_id_maps.py`）：

| 位置 | 含义 | 类别数（新映射） |
|---|---|---|
| parts[0] | 汉字 | 7,765 |
| parts[1] | 字体（篆/楷/草/行/隶/甲骨/金/简/印/石/六体/其他） | 12 |
| parts[3] | 出处/书家 | 1,872 真实 + `others` 兜底 |
| parts[4] | 该字内的样本 id（不是全局 id） | — |

> 少数 label 有 6 段（含碑帖名），共 1,433 行，书家字段取对应段（`seg6` 分支），最终与 5 段行合并统计。

### 3.3 新旧映射文件差异（重要）

仓库里存在**两套不一致的书家/字体/汉字识别表**，调用采样/评估脚本前必须确认用的是哪套：

| 文件 | 书家 | 字体 | 汉字 | 说明 |
|---|---|---|---|---|
| `labels/calligrapher_to_id.json` 等（`labels/*.json`） | 2,243 | 12 | 7,765 | **旧派生**，`sample_3cond.py` 默认使用 |
| `_id_maps.json` | 1,873（1872 真 + `others`） | 12 | 7,765 | **清洗后新映射**，训练/评估管道使用 |

差异根源：
- 旧 `labels/*.json` 按「文件名 → parts[3]」直接收集，含噪声值（数字、空串、未清洗），得到 2,243。
- 新 `_id_maps.json`（`build_id_maps.py` 生成）将噪声归并入 `others`，得到 1,873。

脏数据统计（meta 字段）：

```
总行数            329,715
seg5（标准5段）   328,282
seg6（含碑帖名）   1,433
书名段缺失 → others   16,786
分割出纯数字(如"六体") 423
空段             10
```

> 注意：早期训练配置（如 `train_full_3cond.json`）中使用 `vocab` 2021 类书家，属于更早一版 VOCAB（`_official_vocab.json`），与上面两者都不一致。

### 3.4 final_manifest 切分

`final_manifest_split.json`：**329,715 行**（与 official 全量一一对应），每行含 `final_split` 字段：

```
train: 318,715   (96.7%)
test :  10,000
eval :   1,000
```

- `gen_split_csv.py` 由它生成 `final_train.csv / final_test.csv / final_eval.csv`（本地与远程均有此脚本）。
- **无重叠**：同一 `img_id` 只会出现在一个 split 中。

### 3.5 CSV 列结构（训练/评估读取）

训练与评估均基于 CSV（`dataset.py` / `latent_dataset.py` / `eval_auto.py`），必需列：

```
image_path, canny_path, skeleton_path,
calligrapher_id, script_id, character_id
```

- `image_path`：形如 `final_images/<img_id>.png` 或 `final_imgs_256/<img_id>.png`
- `canny_path` / `skeleton_path`：训练时若 `use_canny / use_skel` 则用于结构 loss；评估采样 `[GT | Canny | Skel | Pred]` 网格
- id 列：对应 `_id_maps.json` 的索引（书家=1873 类、字体=12 类、字=7765 类）

### 3.6 数据增强

训练（is_train=True）随机仿射增强，**同步作用于原图 / Canny / Skeleton**：

- 旋转 ±2°
- 平移 ±2%
- 缩放 0.95 ~ 1.05

---

## 4. 模型架构

### 4.1 DiT-3Cond-XL/2（models.py）

骨架与官方 **DiT-XL/2** 完全一致，改动仅在于条件注入方式：

```
DiT-XL/2 参数：depth=28, hidden=1152, heads=16, patch_size=2
可完整加载官方预训练《DiT-XL-2-256x256.pt》主干权重
```

条件处理链：

```
3 × LabelEmbedder(dropout=0.1)
   [书家 1873(或2021) , 字体 12 , 字 7765]
        ↓ (各自查表 embedding)
       concat
        ↓
   cond_fusion MLP (Linear → SiLU → Linear)
        ↓
   y_emb = cond_fusion(...)
   c     = t_emb + y_emb          ← 进入 DiT block 的 adaLN 调制
```

- **LabelEmbedder**：对给定 id < 类别数时查表，`cfg`（classifier-free）时置 0 下采样。
- **cond_fusion**：三种条件拼成一个向量输入 adaLN（traditional adaLN 只接受「单标签」；这里用 MLP 融合三条件等效于一个大标签）。
- **dropout 0.1**：训练时随机丢弃条件，等价于 CFG 训练。
- `forward_with_cfg`：采样专用，无 dropout。

### 4.2 LoRA 注入（lora.py）

为「体积小、只训增量、避免动预训练主干」而引入：

- 注入位置：每个 block 的 **qkv / proj / fc1 / fc2** 四条 Linear（linear 化后的注意力与 MLP 全接上）。
- 参数：`r=32, alpha=32`（`scaling = alpha / r = 1`）。
- 初始化：A 用 Kaiming，B 全 0 → 初始等于原主干。
- `upgrade_lora_rank`：可把旧低秩(如 r=4/8)权重无损迁移到 r=32。

### 4.3 预训练初始化策略（train.py 加载顺序）

```
1. 构造 DiT-3Cond-XL/2
2. 加载官方 DiT-XL-2-256x256.pt 主干（过滤掉 y_*/cond_fusion/自研头）
3. reset_cond_head()：把 adaLN / final_layer 重置为 std=0.02（防预训练偏置）
4. inject_lora()
5. resume_full()：从 delta checkpoint 恢复训练量
6. freeze 主干（body），仅训练：
      LoRA 权重 + LabelEmbedder×3 + cond_fusion + adaLN + final_layer
```

### 4.4 Checkpoint 格式（delta）

训练保存 `latest.pt`，只含**变更量**（体积小、合并方便）：

```
{
  "delta": { lora_*, y_*, cond_fusion*, adaLN*, final_layer * },
  "opt":   优化器状态,
  "args":  配置,
}
```

冻结 body 不从 ckpt 读，而是**从预训练文件加载**。重建完整推理模型用统一入口：

```
lora.build_model_from_ckpt(path)  # 构造 → 载预训练 body → inject_lora(r32) → 载 delta
```

---

## 5. 训练 / 评估 / 采样调用方式

### 5.1 训练（train.py）

```bash
python train.py --config train_full_3cond.json          # XL/2 3cond + canny + skel
python train.py --config train_full_3cond_skel0.json    # skel0（远程正在跑，从0训练）
```

`train_full_3cond.json` 关键配置：

```
model              = DiT-3Cond-XL/2
cond_mode          = 3cond
vocab              = 2021 / 12 / 7765      ★ 书家用旧 VOCAB(2021)，与 _id_maps(1873) 不一致
lr                 = 1e-4
batch_size         = 4
lora_r=32, lora_alpha=32
reset_cond_head    = true   # adaLN/final_layer 重置 std=0.02
train_cond_head    = true   # 条件头参与训练（修复 SSIM 平台 0.653 问题）
use_canny / use_skel = true, w=0.1
epochs             = 2
ckpt_every         = 1500
auto_eval          = true (eval_n=1000, eval_csv=test.csv)
```

技术要点：

- **bf16 autocast** 前向；NaN 步跳过不更新（不用 GradScaler，直接判 `math.isfinite`）。
- **use_checkpoint=false**（DiT 前向）：历史证 grad checkpoint + dropout 导致重算不一致 → 梯度爆炸。
- **loss = diffusion + 0.1·Canny + 0.1·Skeleton**：结构 loss 在 VAE 解码后的像素空间算，fp32 + grad checkpoint 单独处理。
- 每 `ckpt_every` 步触发 `eval_auto`：单步 t=150 重建 fixed-noise 评估集，报 MSE / SSIM。

### 5.2 内存评估（eval_auto.py）

不落盘的快速评估，挂在训练循环上：

```python
prepare_eval_cache(cfg)   # 训练启动时预编码 test 子集 latent
eval_in_memory(model)     # 每 ckpt 用当前权重单步重建 + 算 MSE/SSIM
```

### 5.3 离线评估（eval_full_3cond.py）

```bash
python eval_full_3cond.py --ckpt <path> [--outdir ...]
```

- 硬编码 XL/2 + r=32；从 `ckpt["ema"]`（回退 `ckpt["model"]`）加载权重。
- 固定 noise、单步 `training_losses()["pred_xstart"]`（t=150）重建。
- 输出 `[GT | Canny | Skel | Pred]` 四列拼图网格。

### 5.4 采样 / 推理（sample_3cond.py）

```bash
python sample_3cond.py \
  --ckpt <path> \
  --calligrapher <书家> --script <字体> --character <字> \
  [--mode ddim] [--steps 50] [--cfg 4.0] \
  [--model DiT-3Cond-S/2]   # 默认是小模型 + labels/*.json 映射
```

- 加载：若 ckpt 含 `lora_` 键则先注入 LoRA 再加载；统一走 `build_model_from_ckpt` 更稳。
- VAE：latent 解码后 **÷ 0.18215**。
- 采样：DDIM 50 步 + CFG=4（`forward_with_cfg`）；batch=1 使用。

### 5.5 远程日志拉取（pull_eval.ps1）

```powershell
./pull_eval.ps1          # scp 拉远程最新 ckpt → 本地跑 eval_full_3cond.py → 按 step 去重到 out/
```

### 5.6 数据/映射/切分脚本

| 脚本 | 作用 |
|---|---|
| `build_id_maps.py` | 解析官方 label → 生成 `_id_maps.json`（清洗 + others 归并） |
| `gen_split_csv.py` | `final_manifest_split.json` → `final_train/test/eval.csv` |
| `prepare_mccd_dataset_fast.py` | 旧管道（随机 80/10/10 + 文件名解析），**已废弃** |
| `_mk_skel0_cfg.py` | 由 skel 配置生成 `train_full_3cond_skel0.json`（从0训练） |
| `_run_full_3cond.sh` | 远程一键重启训练（pkill 旧进程 → nohup 新进程） |

---

## 6. 训练 / 推理一致性

三条链路在「模型 forward 语义」上一致：

| 链路 | 前向方式 | 条件输入 | 重建/采样 |
|---|---|---|---|
| 训练 `training_losses()` | 完整前向（含 dropout） | 输入条件 batch | 目标：预测 noise / pred_xstart |
| 评估 `eval_auto` / `eval_full_3cond` | 同一接口单步 | 相同条件 | t=150 单步 `pred_xstart` |
| 采样 `sample_3cond.py` | `forward_with_cfg`(CFG) | 书家/字体/字 | DDIM 50 步迭代 |

要点：

- 训练用的条件头 = 评估用 = 采样用的条件头（同一个 `y_embedder` 与 `cond_fusion`），不存在「训练一套、推理一套」。
- 评估「单步重建」与采样「多步 DDIM」共享 `pred_xstart`（或 `epsilon`）语义，只是步数不同。
- **注意**：`forward_with_cfg` 有一个 batch>1 bug（`half = x[:len(x)//2]` 丢弃非首个样本）；采样 batch=1 未触发，**尚未修复**，批量采样前需修。

---

## 7. 训好后能写哪些字？

模型**不新增、不裁剪**汉字与书家集合，只能写「映射表内存在」的组合：

- **汉字**：7,765 个 —— `labels/character_to_id.json` 或 `_id_maps.json["character"]` 的全部 key（含生僻字、扩展区字）。查表代码：

```python
import json
chars = json.load(open("labels/character_to_id.json", encoding="utf-8"))  # 或 _id_maps.json
print(len(chars), list(chars)[:20])
```

- **字体**：12 类 —— 篆、楷、草、行、隶、甲骨、金、简、印、石、六体、其他（`script_to_id.json`）。
- **书家**：取决选用映射 —— 旧表 2,243 人 / 新表 1,873（含 `others`）/ 训练配置 2021 类。

**生成条件**：给定 (书家, 字体, 字) 三元组，三者都必须在所用映射内；任一越界 → 模型无对应 embedding，行为未定义（应做映射校验）。

**关于「新」字**：模型无法「发明」映射外的汉字；但同一汉字的**风格迁移**（不同书家 × 字体）天然支持。

---

## 8. 已知问题与坑（踩过记录）

1. **fp16 NaN / GradScaler 崩溃** → 改 bf16 autocast + NaN 步跳过；历史 lr=3e-3 过大 → 降到 1e-4。
2. **use_checkpoint 反向爆炸**（label/LoRA dropout 随机 → 重算前向与反向不一致）→ DiT 前向不开 checkpoint；仅 VAE 解码 loss 用 grad_ckpt。
3. **adaLN / final_layer 曾被错误冻结** → 输出糊、SSIM 平台 0.653；`train_cond_head=true` 修复。
4. **ckpt 冗余 3 份**（model / ema / inference_delta）→ 统一 delta + `build_model_from_ckpt`（已完成，旧的 DESIGN_REVIEW「待重构」表述已过时）。
5. **forward_with_cfg batch>1 bug**（丢弃样本）→ 未修，采样单 batch 可用。
6. **新旧映射不一致**（2243 / 1873 / 2021 三套书家表）→ 训练与推理必须用同一套，当前 skel0 配置建议统一到 `_id_maps.json`（需同步生成 CSV）。
7. **远程数据重建中**：train/test/val.csv 为 0 行，skel0 训练使用重建管道的新 CSV 后才有完整数据支撑。

---

## 9. 快速参考 FAQ

**Q: 现在想重新跑一轮训练？**
A: 先确认远程 `train.csv/test.csv` 非空（重建完成），然后 `python train.py --config train_full_3cond_skel0.json`。

**Q: 想评估某个 ckpt 质量？**
A: 本地 `python eval_full_3cond.py --ckpt <path>` 看 4 列对比网格；或训练中看 `eval_in_memory` 的 MSE/SSIM。

**Q: 想生成某个字？**
A: `python sample_3cond.py --ckpt <path> --calligrapher 王羲之 --script 行 --character 永`（字与书家必须存在于所选映射）。