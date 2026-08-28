# 训练体系

> 对应源码：`src/train/train.py`（主模型）、`src/train/train_controlnet.py`（ControlNet）、`src/train/configs/`（配置）
> 启动方式不变：`python train.py --config <json>`；ControlNet：`python src/train/train_controlnet.py --config ctrl_skel_s18_flow.json`（旧调用方式兼容）。

## 1. 主模型训练（train.py）

### 1.1 入口与启动

- `main(args)`：真正的训练（约 line 122）。分布式初始化默认单卡；rank 0 创建**带时间戳的实验目录** `{results_dir}/{YYYYmmdd-HHMMSS}-{experiment_name}`，永不覆盖。
- `main_from_cli(argv=None)`（约 line 1314）：argparse（配置默认值 + CLI 覆盖）→ `main(args)`。根目录 `train.py` 只是两行 launcher。

### 1.2 每步做了什么

```python
# 1) 数据：latent shard 数据集（preload）+ factor_balanced 采样器
# 2) 统一时间步采样
t = diffusion.sample_t(x_latent.shape[0], device)          # flow∈[0,1) / ddpm∈{0..T-1}
model_kwargs = dict(y_callig=y_callig, y_char=y_char)      # (+g 若 w_glyph_cond, +REPA 若 w_repa)
# 3) bf16 前向
with torch.autocast("cuda", dtype=torch.bfloat16):
    loss_dict = diffusion.training_losses(model, x_latent, t, model_kwargs)
    loss = loss_dict["loss"].mean()
# 4) 渐进结构损失（$x0$ 图只在需要时保留，属 INFRA 防内存泄漏设计，见下方 §1.4）
# 5) backward → grad clip → optimizer.step → scheduler.step
# 6) EMA 更新（warmup 衰减: min(decay, (1+step)/(10+step)))）
# 7) 每 ckpt_every: 存 ckpt(+.done) → 轮换 ckpt_keep → in-process GPU eval
```

### 1.3 EMA 与推理评测的换权重机制

- `use_ema=true`（默认 0.9999 + warmup）：`ema_model = deepcopy(model).eval()`，每步 `update_ema` 更新参数与浮点 buffer。
- eval 时**原地换上 EMA 权重**跑采样，结束后恢复训练权重（`_orig_sd` 快照 → `load_state_dict` 回写）——避免训练进程外另开 GPU 进程。

### 1.4 INFRA 防泄漏设计（历史教训）

- `training_losses` 返回的 `pred_xstart` 携带整条 DiT 计算图；若结构损失未启用（`_need_x0_grad=False`），立即 `.detach()` 释放，防止 20G→22G→14G 的训练显存震荡。
- flow 模式下 `_flow_disabled` 自动禁用 DDPM 专属辅助（canny/skel 像素损失、x0 结构损失、REPA 等），防止语义错配。
- ckpt 轮换同步删除 `.done` 与对应 `eval_*` 目录。

## 2. ControlNet 训练（train_controlnet.py）

- warm-start（`train_ctrl_only=true`）：`load_main_model(main_ckpt)` 冻结主模型 → `ControlNetDiT(main, train_ctrl_only=True)` → 只训 `ctrl_encoder` + `out_projs`（EMA 只更新 trainable 参数）。
- 数据：latent shards + `final_skeleton_d3` 骨架（(N,1,256,256) 0/1）。
- **统一 `sample_t`**；可选 skel 条件 dropout（`cond_drop_struct_prob`）。
- ckpt：只存 `ctrl_encoder` 权重（`"ctrl"` + `"ema"`），from-scratch 模式另存 `main.*`。
- 每 `gpu_eval_every` 步 in-process GPU eval（GT-skel 的 ctrl 组 + 无 skel 的 base 组）→ pending marker → CPU daemon 出指标（`07_eval.md`）。

## 3. 优化器 / 调度 / EMA 默认

| 项 | 主模型 | ControlNet |
|---|---|---|
| 优化器 | AdamW | AdamW |
| lr | 2e-4 | 1e-4 |
| 调度 | cosine + 3000 warmup, min_lr_ratio 0.1 | cosine + 1500 warmup, min_lr_ratio 0.1 |
| weight_decay | 0.02 | 0.01 |
| EMA | 0.9999 + warmup | 0.9999，前 2000 步 4 次方 ramp |
| grad clip | 有 | max_norm=1.0 |
| batch | 240（单卡） | 152 |
| 精度 | bf16 autocast（数值稳定；VAE 编码保持 fp32） | bf16 autocast |

## 4. 训练配置字段全解（主模型 json）

见 `s19_midclean_s_flow.json`（根目录，当前 s19 预训练配置）：

| 字段 | 值（s19） | 含义 |
|---|---|---|
| `experiment_name` | s19-midclean-s-flow | 实验 slug（拼在时间戳后，唯一目录） |
| `data_csv` | 5script/train_mid_clean.csv | 训练表（mid-clean 118,776 行） |
| `results_dir` | 5script/results/s19_midclean_s_flow | 实验根目录 |
| `model` | DiT-2Cond-S/2 | 模型规格 |
| `cond_mode` / `condition_fusion` | 2cond / factorized_add | 条件模式与融合 |
| `callig_embed_dim` / `char_embed_dim` | 128 / 384 | 条件嵌入维度 |
| `char_dino_embeddings` / `char_dino_index` | pretrained_models/dino_embeddings/glyph_dino_embeddings_384.npy (+index) | DINO 字形表（384 维，LN-only 直通） |
| `char_proj_mode` / `freeze_char_table` | ln_only / true | 字符投影只 LN、表冻结（条件=纯 DINO 向量） |
| `cond_drop_all_prob` / `cond_drop_one_prob` / `cond_drop_which_glyph_prob` | 0.05 / 0.25 / **0.75** | 4-way dropout 配比（`03_model.md` §2） |
| `image_size` | 256 | 图片尺寸（latent 32） |
| `num_calligraphers` / `num_characters` | 1011 / 35130 | 词汇表尺寸（对齐 MCCD 全集），数据只用 67/5461 |
| `max_steps` / `epochs` | 300000 / 100000 | 训练上限（max_steps 优先） |
| `lr` / `lr_schedule` / `warmup_steps` / `min_lr_ratio` | 2e-4 / cosine / 3000 / 0.1 | 优化 |
| `weight_decay` | 0.02 | L2 |
| `global_batch_size` / `global_seed` | 240 / 0 | 数据 |
| `sampler` / `balance_char_alpha` / `balance_callig_alpha` | factor_balanced / 0.35 / 0.15 | 平衡采样 |
| `use_ema` / `ema_decay` / `ema_warmup` | true / 0.9999 / true | EMA |
| `vae` / `vae_path` | ema / pretrained_models/sd-vae-ft-ema | 解码用 VAE |
| `autoeval*`（`eval_csv` / `eval_n` / `eval_steps` / `eval_cfg` / `eval_seed` / `eval_batch` / `eval_vae_batch`） | eval_strict_top6.csv / 271 / 50 / **1.7** / 0 / 100 / 32 | in-process GPU eval 参数（`07_eval.md`） |
| `early_stop*` | combo / patience 3 / min 30000 | 早停（依赖 CPU daemon 出的指标 json） |
| `latent_shards_dir` | final_latents_mid_clean | latent 缓存 |
| `diffusion_type` | **flow** | 扩散框架（`02_diffusion.md`） |
| `ckpt_every` / `ckpt_keep` / `log_every` / `num_workers` / `preload*` | 2500 / 0 / 20 / 8 / true+48 | 训练节奏 |
| `use_lora` / `reset_cond_head` / `train_cond_head` | false / false / false | 微调开关（当前预训练关闭） |
| `w_canny*` / `w_skel*` / `w_repa` / `use_canny` / `use_skel` | 0（关闭） | 结构/辅助损失（flow 下禁用） |

**ControlNet 专属字段**（`src/train/configs/ctrl_skel_s18_flow.json`）：

| 字段 | 含义 |
|---|---|
| `main_ckpt` | 冻结主模型 ckpt 路径（warm-start） |
| `skel_root` / `skel_cond_channels` | 骨架图根 / 条件通道数（1） |
| `cond_drop_struct_prob` | skel 条件 dropout 概率（0.1） |
| `train_ctrl_only` | warm-start / from-scratch |
| `batch_size` | 152 |
| `gpu_eval_*`（csv / img_root / skel_root / every / n / steps / **cfg 1.7** / dit_batch / vae_batch） | in-process GPU eval 组 |

## 5. 可追溯性（resolved_config 等）

每次启动 rank 0 会在实验目录写三个文件：

1. `resolved_config.json` —— **vars(args)** 全量落盘（含 json 默认 + CLI 覆盖后的最终值）。**所有配比决策都可通过它事后核查**。
2. `source_manifest.json` —— 关键源码 sha256 + 环境（python/torch/cuda/hostname）。
3. `_active_ckpt_dir.txt` —— 指向当前实验 ckpt 目录，供独立 CPU daemon 定位。

## 6. 重启 / 恢复

- 主模型：`--pretrained <ckpt.pt>`（载入 `ema` 优先）→ 继续训练（可选 `--fresh-scheduler`、`--reset-cond-head` / `--train-cond-head`）。
- ControlNet：`--resume` + ckpt 内 `ctrl_encoder` 权重（优先 ema）；resume 时重建 sampler/optimizer 状态。
- **数据只留远程**：重跑任何训练都基于远程现有 csv/shards，本地只改代码与配置。