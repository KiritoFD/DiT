# 训练模块说明（TRAINING.md）

> 2026-08-14 ｜ 覆盖 `train.py` / `models.py` / `lora.py` / 配置文件与启动方式。

---

## 1. 支持的模型（`models.py`）

| 模型名 | depth | hidden | heads | 参数 |
|---|---|---|---|---|
| `DiT-3Cond-S/2` | 12 | 384 | 6 | 36.9M |
| `DiT-3Cond-B/2` | 12 | 768 | 12 | ~130M |
| `DiT-3Cond-L/2` | 24 | 1024 | 16 | ~458M |
| `DiT-3Cond-XL/2` | 28 | 1152 | 16 | 706.9M |

三条件（书家/字体/字）→ 3 个 `LabelEmbedder` → `cond_fusion`（concat 3D → MLP → D）→ 每层 **adaLN-Zero** 调制。
`forward_with_cfg` 支持 CFG 采样；`use_checkpoint` 默认 false（历史 NaN 根因）。

## 2. 三种训练模式（冻结策略，`train.py`）

冻结策略与 `use_lora` **解耦**（`if use_lora or pretrained is not None` 才冻结 body）：

| 模式 | 配置 | 可训练 | 适用 |
|---|---|---|---|
| 从零全参 | `use_lora=false, pretrained=null` | 全部 | 小模型预训练（S） |
| 预训练+条件头 | `use_lora=false, pretrained=<XL.pt>` | 条件头+adaLN+final_layer（body 冻结） | XL 微调最轻档 |
| +LoRA | `use_lora=true, pretrained=<XL.pt>, lora_r=N, lora_target=all/attn/mlp` | 上 + LoRA 注入层 | 容量可调微调 |

- **条件头（y_* / cond_fusion）必须训**（新增层，随机初始化）。
- **adaLN/final_layer 必须训**（`reset_cond_head=true` 重置为 std=0.02 后，若冻结 → 输出糊，历史根因）。
- `--lora-target`：`all`(qkv+proj+fc1+fc2) / `attn`(qkv+proj) / `mlp`(fc1+fc2)，112 层可选注入范围。
- 启动时日志打印精确 `Trainable Parameters` / `Frozen Parameters`。

## 3. 配置文件（JSON → argparse 默认值）

`train.py --config <cfg.json>`，所有参数可被 CLI 覆盖（`--lora-r`, `--lr`, `--epochs` …）。

关键字段：

| 字段 | 说明 |
|---|---|
| `pretrained` | `null`=从零；路径=加载官方 body（过滤 y_* / cond_fusion） |
| `use_lora` / `lora_r` / `lora_alpha` / `lora_target` | LoRA 注入 |
| `reset_cond_head` / `train_cond_head` | adaLN 重置与训练开关 |
| `use_canny` / `use_skel` / `w_canny` / `w_skel` | 结构 loss（**默认关**，见 MEMORY_GUIDE.md 显存原因） |
| `latent_shards_dir` / `img_root` / `canny_root` / `skel_root` | latent-cached 数据路径 |
| `preload` | 启动时把 latent(+canny/skel) 读入内存（零磁盘 IO） |
| `ckpt_keep` | 只保留最近 N 个 ckpt（0=全留） |

## 4. Checkpoint 与调度

- **保存内容**：`use_lora=true` → `delta`（LoRA+条件头+adaLN）；`use_lora=false` → 完整 state_dict；均含 `opt`（CPU 化）与 `args`。
- **保存调度**：前 5000 步每 1000 存一个；之后每 4000 存一个（硬编码，见 `train.py`）。
- **恢复**：`--resume-full <ckpt.pt>` 恢复 delta + opt + step。
- **eval**：每 ckpt 自动 `eval_in_memory`（1k eval，MSE/SSIM，`t=150` 单步重建）；1000 张对比图落盘 `eval_<step>/` + `eval_latest.png`。

## 5. 启动示例

```bash
# S 从零（关结构 loss，大 batch）
/opt/conda/bin/python train.py --config exp_s_scratch.json

# XL-2 预训练 + LoRA r32（仅 attention）
/opt/conda/bin/python train.py --config exp_xl_head_r32.json \
    --lora-target attn --lora-r 32 --lora-alpha 32

# 远程后台（tmux）
tmux new-session -d -s exp "bash /root/Workspace/xy/DiT/_launch_exp.sh exp_xl_head_r32.json exp_xl_head_r32.log"
```

## 6. 显存与速度要点（详见 MEMORY_GUIDE.md）

- 结构 loss 的像素空间 VAE decode = 15.56G（batch8）是显存天花板 → 默认关闭。
- 纯 diff 训练：S batch64 ~6G / 12.4 steps/s；XL batch8 ~11G / ~7-10 steps/s。
- 日志每 20 步输出 `Mem: <当前>G/<峰值>G` 全程监控。
