# Training Pipeline（训练管线）

> 覆盖 `train.py`（根目录，远程使用）/ `src/train.py`（源真值，与根目录略有差异）、配置文件、训练循环、优化器/调度器/EMA、显存与速度、远程启动与 auto-eval。
> 当前主跑实验：**s7-klf4-top30-diffonly**（从零全参预训练，f4 VAE，latent-cached，diff-only）。

---

## 0. 文档定位与阅读顺序

本文档只讲「怎么训练、训练时发生了什么、为什么这么配」。与之相关的其它文档：

- `docs/legacy/TRAINING.md` —— 早期训练说明（冻结策略、LoRA、3cond），本文是其在 f4 + latent-cached 路线下的演进版。
- `docs/data/` —— 数据集与 latent shard 构建（`MCCDDataset` / `MCCDLatentDataset` 的字段约定）。
- `docs/model/CONTROLNET.md` —— ControlNet（骨架条件）分支训练，独立脚本 `tools/controlnet/train_controlnet.py`。
- `docs/experiments/2026-08-17-latent-vs-pixel-struct.md` —— latent vs pixel 结构监督的定量依据（`w_latent_canny`/`w_latent_skel` 动机）。

如无特别说明，下文「train.py」指根目录脚本（远程实际运行的版本），关键行为同时在 `src/train.py` 中存在。

---

## 1. 训练脚本

### 1.1 两份 train.py 的关系

| 路径 | 角色 | 说明 |
|---|---|---|
| `train.py` | **远程运行版** | 部署在服务器 `/root/Workspace/xy/DiT/train.py`，被 tmux 直接调用 |
| `src/train.py` | **源真值（source of truth）** | 本地仓库的规范化副本，含小幅整理（如 3cond 也传 `in_channels`） |

两者主体一致，差异是局部的参数透传整理（例如 `src/train.py` 在构造 `DiT_3Cond_models` 时显式传入 `in_channels=getattr(args, 'latent_channels', 4)`，根目录版在 2cond 分支才有）。**改动训练逻辑时应先改 `src/train.py`，再同步到根目录**；远程只拉根目录。若两份不一致，以 `src/train.py` 为准。

### 1.2 启动

```bash
python train.py --config <config.json>
```

- 配置 JSON 包含**全部**超参（模型、数据、优化器、调度、EMA、ckpt、eval、early-stop）。
- CLI 参数覆盖配置：`train.py` 先用 `--config` 指向的 JSON 作为各 argparse 项的默认值，再解析 CLI 覆盖（`train.py:1322-1339`）。

```python
# train.py:1322-1339  config → defaults → CLI override
parser.add_argument("--config", type=str, default="config.json", ...)
config_defaults = {}
cfg_path = parser.parse_known_args()[0].config
if cfg_path and os.path.isfile(cfg_path):
    with open(cfg_path, "r", encoding="utf-8") as f:
        config_defaults = json.load(f)
for action in parser._actions:
    if action.dest in ("help", "config"):
        continue
    if action.dest in config_defaults:
        action.default = _coerce(config_defaults[action.dest], action.default, action.type)
        action.required = False
args = parser.parse_args()
main(args)
```

`_coerce`（`train.py:39-51`）把 JSON 值强制成 argparse 默认值的类型（`bool`/`int`/`float`/`str`），避免 `"false"` 字符串被当真值。CLI 仍是最终仲裁。

---

## 2. 训练循环

### 2.1 总体流程

```
1. 解析 config → args（JSON 默认 + CLI 覆盖）
2. 构建模型：DiT_2Cond_models[args.model] 或 DiT_3Cond_models[args.model]
   - latent_size = image_size // vae_downscale
   - in_channels = latent_channels
3. 加载顺序（固定）：pretrained body → reset cond head → 注入 LoRA（若启用）→ load delta
4. 装配：EMA（可选）、optimizer（AdamW）、LR scheduler（cosine）
5. 构建数据集：MCCDLatentDataset（latent-cached）/ MCCDDataset（pixel）
6. 训练循环：
   a. 取 batch：x_latent（预编码）+ 条件（callig, char）
   b. 采样 timestep t，加噪：x_t = sqrt(alpha_t)·x_0 + sqrt(1-alpha_t)·eps
   c. 前向（bf16 autocast）：model(x_t, t, y_callig, y_char) → eps_pred
   d. 损失：MSE(eps_pred, eps)
   e. 反向、梯度裁剪（max_norm=1.0）、optimizer.step()
   f. EMA 更新（若启用）
   g. 释放计算图（del loss, loss_dict, pred_xstart）
   h. 每 20 步日志，每 5000 步存 ckpt
```

### 2.2 关键代码落点

| 阶段 | 位置（`train.py`） |
|---|---|
| 模型构造（2cond） | `:206-220` |
| pretrained body 加载 + 过滤条件键 | `:234-246` |
| reset adaLN/final_layer（std=0.02） | `:259-268` |
| 注入 LoRA | `:270-285` |
| full resume（load delta + opt + scheduler） | `:288-477` |
| 冻结/可训练策略 | `:297-322` |
| EMA 装配 | `:326-332` |
| MockVAE / 真实 VAE 选择 | `:357-372, :486-491` |
| 数据集 + DataLoader | `:492-538` |
| cosine LR schedule | `:546-567` |
| 训练主循环 | `:640` 起 |
| bf16 autocast 前向 + training_losses | `:683-686` |
| pred_xstart 立即 detach（无结构 loss 时） | `:695-718` |
| backward + grad clip + opt.step + EMA | `:896-908` |
| 计算图释放 `del` | `:917-923` |
| 日志（每 `log_every` 步） | `:941-1027` |
| 存 ckpt + `.done` + 轮转 | `:1029-1080` |
| early-stop 检查 | `:1086-1093` |

### 2.3 单步伪代码（diff-only 路径，s7）

```python
# train.py:670-923 精简版
t = torch.randint(0, diffusion.num_timesteps, (B,), device=device)
model_kwargs = dict(y_callig=y_callig, y_char=y_char)

# bf16 autocast：与 fp32 同指数范围，不会像 fp16 AMP 那样溢出，无需 loss scaling
with torch.autocast("cuda", dtype=torch.bfloat16):
    loss_dict = diffusion.training_losses(model, x_latent, t, model_kwargs)
    loss_diff = loss_dict["loss"].mean()

# pred_xstart 携带整条 DiT 前向计算图（12 block 的激活都被钉住），
# 无结构 loss 时立即 detach，并把 loss_dict 里的引用 pop 掉，避免计算图残留
pred_xstart_latent = loss_dict.get("pred_xstart", None)
if pred_xstart_latent is not None and not _need_x0_grad:
    pred_xstart_latent = pred_xstart_latent.detach()
loss_dict.pop("pred_xstart", None)

loss = loss_diff                       # diff-only：总损失 = 扩散损失
if torch.isfinite(loss):
    loss.backward()
    torch.nn.utils.clip_grad_norm_(trainable_params_list, max_norm=1.0)
    opt.step()
    if scheduler is not None: scheduler.step()
    if ema_model is not None:
        current_ema_decay = min(args.ema_decay, (1.0 + train_steps) / (10.0 + train_steps))  # warmup
        update_ema(ema_model, model, current_ema_decay)
else:
    nan_steps += 1                      # NaN 守卫：跳过该步，不更新参数
del loss, loss_dict, loss_diff, pred_xstart_latent   # 释放计算图
```

加噪本身由 `diffusion.training_losses`（`gaussian_diffusion`）内部完成，等价于
`x_t = sqrt(alpha_t)·x_0 + sqrt(1-alpha_t)·eps`，损失为 `MSE(eps_pred, eps)`。

---

## 3. 关键训练参数

### 3.1 精度

- **bf16 autocast** 用于前向（`train.py:684`），无需 loss scaling —— bf16 与 fp32 共享指数范围，不会像 fp16 AMP 那样溢出。
- VAE encode 保持 **fp32**（`train.py:666-668`），VAE 对低精度敏感。
- 结构损失保持 **fp32**（`train.py:820` 注释），在 autocast 外计算以保数值稳定。
- 模式开关一行：

```python
# train.py:683-685
with torch.autocast("cuda", dtype=torch.bfloat16):
    loss_dict = diffusion.training_losses(model, x_latent, t, model_kwargs)
```

- TF32 已开启以加速 matmul/cudnn：`train.py:6-7`
  (`torch.backends.cuda.matmul.allow_tf32 = True`、`torch.backends.cudnn.allow_tf32 = True`)。

### 3.2 优化器

- **AdamW**，`lr=2e-4`（默认），`weight_decay=0.02`（`train.py:448`）。
- **Cosine LR schedule + warmup**（`train.py:546-567`）：
  - warmup 段线性升到 base lr：`lr_scale = (step+1)/warmup_steps`；
  - 之后余弦衰减到 `min_lr_ratio × base_lr`：
    `cosine = 0.5*(1+cos(π·progress))`，`lr_scale = min_lr_ratio + (1-min_lr_ratio)·cosine`。
  - s7 配置：`warmup_steps=2000`，`min_lr_ratio=0.1`。
- **梯度裁剪**：`torch.nn.utils.clip_grad_norm_(trainable_params_list, max_norm=1.0)`（`train.py:898`）。
- NaN 守卫：`loss` 非有限时跳过该步、不更新参数、累计 `nan_steps` 并告警（`train.py:895-916`）。

### 3.3 EMA（Exponential Moving Average）

- `use_ema=true`（**从零预训练推荐开启**），`decay=0.9999`，`ema_warmup=true`（`train.py:902-908`）。
- warmup 限制早期衰减：`current_ema_decay = min(decay, (1+step)/(10+step))`，避免冷启动阶段 EMA 被随机初始化权重拖偏。
- EMA 权重**仅用于评估**（`ema_model.eval()`，`requires_grad=False`），训练仍用原模型。
- 代价：+170MB VRAM（fp32 全模型副本），计算量可忽略。
- 收益：评估更平滑、early-stopping 更可靠、最终模型质量更好。
- full-resume 时优先从 ckpt 恢复 EMA 权重（`train.py:329-331`）。

```python
# train.py:66-79  EMA 更新（含浮点 buffer）
@torch.no_grad()
def update_ema(ema_model, model, decay):
    source = model.module if hasattr(model, "module") else model
    source_params = dict(source.named_parameters())
    for name, ema_param in ema_model.named_parameters():
        ema_param.mul_(decay).add_(source_params[name].detach(), alpha=1.0 - decay)
    source_buffers = dict(source.named_buffers())
    for name, ema_buffer in ema_model.named_buffers():
        source_buffer = source_buffers[name].detach()
        if torch.is_floating_point(ema_buffer):
            ema_buffer.mul_(decay).add_(source_buffer, alpha=1.0 - decay)
        else:
            ema_buffer.copy_(source_buffer)
```

### 3.4 Batch Size 与显存

| 配置 | batch | VRAM（RTX 4090 24G） | 备注 |
|---|---|---|---|
| s7（f4 kl-f4，latent-cached） | 224 | **19.74G** | 当前主跑 |
| s5/s6（sd-vae f8，latent-cached） | 192 | 17.1G | 旧路线 |

- **f4 latent 每图比 f8 大 3×**：f4 为 64×64×3=12288 元素/图，f8 为 32×32×4=4096 元素/图（12288/4096=3）。
- 但 **DiT token 数相同（256）**：s7 用 `DiT-2Cond-S/4`（patch=4），64×64 latent → 16×16=256 token；s5/s6 用 patch=2，32×32 latent → 16×16=256 token。patch 补偿了空间尺寸。
- 因此 DiT 端注意力/激活显存接近，整体 VRAM 相近（差异主要来自 latent 张量与 VAE 编码侧）。
- 实测速度：s7 **3.51 steps/s**（diff-only，bf16 autocast）。

### 3.5 Latent-cached vs Pixel 模式

- **Latent-cached（首选）**：latent 在 shard 中预编码，启动时 preload 进内存（128k f4 约 5.9G RAM）。
  - VAE **不加载**，用 `MockVAE` 占位（`train.py:341-355, 489-491`），省 ~500MB VRAM。
  - 开关：`use_latent = bool(getattr(args, "latent_shards_dir", None))`（`train.py:482`）。
  - 数据集：`MCCDLatentDataset`（`train.py:493-505`）。
- **Pixel 模式**：每 batch 现场用 VAE encode（`train.py:666-668`），VAE 常驻 GPU。
  - 数据集：`MCCDDataset`（`train.py:509-510`）。
  - 仅在 `not use_latent` 或需要 pixel 结构损失（`w_repa`/`use_canny`）时加载真 VAE（`train.py:488`）。

### 3.6 Checkpointing

- **保存节奏**（`train.py:1029-1033`）：
  - `train_steps ≤ 5000`：每 1000 步存一个；
  - `train_steps > 5000`：每 5000 步存一个（`(train_steps - 5000) % 5000 == 0`）。
- **轮转**：`ckpt_keep=60`，只保留最近 60 个 ckpt 及其 eval 目录（`train.py:1064-1080`）。
- **保存内容**（`train.py:1041-1058`）：

| 键 | 内容 | 说明 |
|---|---|---|
| `delta` | LoRA 模式：LoRA + 条件头 + adaLN/final_layer；全参模式：完整 `state_dict()` | `use_lora` 决定 |
| `opt` | 优化器 state（`_state_to_cpu` 递归搬到 CPU） | resume 用 |
| `args` | 训练时 argparse Namespace | 复现/恢复 |
| `train_steps` | 当前步数 | early-stop/恢复 |
| `ema` | EMA `state_dict()`（若 `use_ema`） | 评估用 |
| `scheduler` | LR scheduler `state_dict()`（若 cosine） | resume 用 |

- 存盘前用 `_state_to_cpu`（`train.py:81-93`）把张量递归搬到 CPU，避免 `torch.save` 触发 GPU 显存尖峰。
- **`.done` 标记**：存盘后写 `ckpt_path + ".done"` 空文件（`train.py:1061`），向 `auto_eval_cpu` 信号「该 ckpt 已就绪可评估」。

```python
# train.py:1036-1062
if rank == 0:
    model_to_save = model.module if hasattr(model, 'module') else model
    if getattr(args, 'use_lora', True):
        delta = extract_full_inference(model_to_save)
    else:
        delta = model_to_save.state_dict()
    delta = _state_to_cpu(delta)
    _opt_cpu = _state_to_cpu(opt.state_dict())
    checkpoint = {"delta": delta, "opt": _opt_cpu, "args": args, "train_steps": train_steps}
    if ema_model is not None: checkpoint["ema"] = _state_to_cpu(ema_model.state_dict())
    if scheduler is not None: checkpoint["scheduler"] = scheduler.state_dict()
    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
    torch.save(checkpoint, checkpoint_path)
    open(checkpoint_path + ".done", "w").close()       # 信号 auto_eval_cpu
```

### 3.7 Early Stopping

- 基于 **eval SSIM**（`early_stop_metric="ssim"`，越大越好；MSE 则越小越好，`train.py:594-595`）。
- `patience=6`：连续 6 个 eval 点无改善则停（`train.py:625-636`）。
- `min_steps=60000`：仅 60k 步后才检查（`train.py:1086-1087`）。
- 检查频率：默认 `ckpt_every // 2`（最少 1000），可用 `--early-stop-check-every` 覆盖（`train.py:596-598`）。
- eval 数据来自 `auto_eval_cpu` 写入的 `eval_auto_{step}.json`（`train.py:605-622` 读取），由独立 CPU 进程产生，不阻塞 GPU。

```python
# train.py:600-636 精简
def _early_stop_check(force=False):
    ev_files = sorted(glob(os.path.join(checkpoint_dir, "eval_auto_*.json")))
    last_ev = ev_files[-1]
    ev_step = int(...)  # 从文件名解析 step
    if ev_step <= early_stop_last_eval_step: return False
    d = json.load(open(last_ev))
    val = float(d["ssim"]) if _es_metric == "ssim" else float(d["mse"])
    if early_stop_best is None or _es_better(val, early_stop_best):
        early_stop_best, early_stop_stale = val, 0
    else:
        early_stop_stale += 1
        if early_stop_stale >= args.early_stop_patience: return True   # 触发停止
    return False
```

---

## 4. VAE-Aware 参数（train.py）

f4（kl-f4）与 f8（sd-vae）在 train.py 里由一组 `--vae-*` / `--latent-*` 参数切换，模型与数据集都据此构造。

| 参数 | 默认 | f8（sd-vae） | f4（kl-f4） |
|---|---|---|---|
| `--vae-downscale` | 8 | 8 | **4** |
| `--latent-channels` | 4 | 4 | **3** |
| `--vae-in-channels` | 3 | 3 | 3 |
| `--vae-out-channels` | 3 | 3 | 3 |
| `--vae-scaling-factor` | 0.18215 | 0.18215 | **0.102079** |
| `--vae-path` | sd-vae-ft-ema | sd-vae-ft-ema | `pretrained_models/kl-f4` |

派生关系（`train.py:174-177`）：

```python
vae_downscale = getattr(args, 'vae_downscale', 8)
assert args.image_size % vae_downscale == 0
latent_size = args.image_size // vae_downscale     # 256//4 = 64  (f4); 256//8 = 32 (f8)
```

- 模型以 `in_channels=latent_channels` 构造（`train.py:219`），f4 时为 3。
- 模型名里的 `/4`、`/2` 是 **patch_size**，不是 VAE downscale。`DiT-2Cond-S/4`：latent 64×64，patch=4 → token 数 = (64/4)² = **256**；`DiT-2Cond-S/2`：latent 32×32，patch=2 → (32/2)² = **256**。两者 token 数相同，所以 s7（f4 + S/4）与 s5/s6（f8 + S/2）的 DiT 端显存接近。
- latent-cached 模式下 VAE 被 `MockVAE` 占位，其 `_vae_ds/_vae_lc/_vae_oc/_vae_sf` 是模块级动态变量（`train.py:336-340`），按 `args` 实际值确定编码/解码形状与 scaling。

---

## 5. 训练模式（冻结策略）

冻结策略与 `use_lora` **解耦**：`if use_lora or pretrained is not None` 才冻结 body（`train.py:304-322`）。

### 5.1 从零全参（当前 s7）

- `pretrained: null`，`use_lora: false`。
- **全部 41.8M 参数可训练**（`DiT-2Cond-S/4`）。
- 需要大数据（128k 图）+ 多步（`max_steps=600000`）。
- **EMA 强烈推荐**（冷启动平滑）。
- 启动日志会打印精确的 `Trainable Parameters` / `Frozen Parameters`（`train.py:319-322`）。

### 5.2 Warm-start 微调（s6 及更早）

- `pretrained: "path/to/DiT-XL-2-256x256.pt"`（ImageNet DiT body）。
- `use_lora: true`（在 transformer block 注入 LoRA）。
- `reset_cond_head: true`：加载预训练 body 后把 adaLN/final_layer **重置为 std=0.02**（`train.py:259-268`）。原因：ImageNet 的 adaLN 学的是「1000 类自然物分类 → 调制」，与书法（callig×char）条件正交，保留会乱码/跑偏。
- 可训练：仅 LoRA + 条件头（`y_callig_embedder`/`y_char_embedder`/`cond_fusion`/...）+ adaLN/final_layer（`train_cond_head=true` 时），约 2-4M 参数。

### 5.3 加载顺序（固定）

```
pretrained body → reset cond head → inject LoRA → load delta (full resume)
```

`train.py:228-296`：
1. **pretrained body**：过滤掉 `y_embedder/y_callig/y_script/y_char/cond_fusion` 等条件键，只加载 transformer 通用引擎（`x_embedder/pos_embed/t_embedder/blocks/final_layer`）。
2. **reset cond head**：adaLN/final_layer 重新初始化（仅在非 resume 时）。
3. **inject LoRA**：`inject_lora(model, r, lora_alpha, target)`，`target` 可选 `all`/`attn`/`mlp`。
4. **load delta**：`--resume-full <ckpt.pt>` 恢复 `delta` + `opt` + `scheduler` + `ema`，并从文件名推断起始步数（`train.py:468-477`）。

---

## 6. 显存优化

### 6.1 计算图管理（VRAM 关键）

`pred_xstart` 由 `training_losses` 返回，携带**整条 DiT 前向计算图**（每个 block 的激活都被钉住，因为梯度回流经 `_predict_xstart_from_eps → model_output`）。这是显存「20G→22G→14G 周期性波动」的根因。修复策略（`train.py:695-723`）：

- 无结构 loss 时（diff-only）**立即 detach**：`pred_xstart_latent = pred_xstart_latent.detach()`。
- `loss_dict.pop("pred_xstart", None)` 清掉 dict 内的重引用，避免残留计算图跨步存活。
- 每步结束后 `del loss, loss_dict, loss_diff, pred_xstart_latent` 等所有损失张量（`train.py:917-923`），在下次前向前释放。
- 结构 decoder 的计算图在反向前释放（`train.py:820` 注释）。

```python
# train.py:695-718 精简
_need_x0_grad = (latent_struct_loss_fn is not None
                 or getattr(args, 'w_latent_skel', 0) > 0
                 or getattr(args, 'w_latent_canny', 0) > 0
                 or getattr(args, 'w_std_mid', 0.0) > 0)
pred_xstart_latent = loss_dict.get("pred_xstart", None)
if pred_xstart_latent is not None and not _need_x0_grad:
    pred_xstart_latent = pred_xstart_latent.detach()   # 立即丢图
loss_dict.pop("pred_xstart", None)                     # 清 dict 内引用
```

### 6.2 DataLoader

`train.py:527-537`：

| 参数 | 值 | 作用 |
|---|---|---|
| `pin_memory` | `True` | CPU→GPU 传输走 page-locked 内存 |
| `persistent_workers` | `True`（`num_workers>0` 时） | worker 不在 epoch 间重启 |
| `prefetch_factor` | 4（`num_workers>0` 时，否则 2） | 每 worker 预取 batch 数 |
| `drop_last` | `True` | 保证 batch size 一致（避免尾批 OOM/shape 抖动） |
| `num_workers` | 8 | CPU 并行加载，与 GPU 计算重叠 |
| `batch_size` | `global_batch_size // world_size` | 单卡 batch |

### 6.3 其它

- `_state_to_cpu`（`train.py:81-93`）：递归把 `opt.state_dict()`（state→param_idx→tensors 两层嵌套）和张量搬到 CPU 再 `torch.save`，避免存盘时 GPU 显存尖峰。
- latent-cached + `preload=true`：shard 在启动时由 `preload_workers`（s7=24）读进 RAM，训练期零磁盘 IO（`train.py:501-507`）。
- `use_checkpoint`（梯度检查点）默认 `false`（历史 NaN 根因），s7 关闭；显存够用时不开启。

---

## 7. 远程训练配置

### 7.1 环境

| 项 | 值 |
|---|---|
| 服务器 | `root@10.176.54.17:36430` |
| GPU | RTX 4090（24G VRAM） |
| Python | `/opt/conda/bin/python`（3.10, torch 1.13.1+cu117） |
| 工作目录 | `/root/Workspace/xy/DiT` |
| 会话管理 | tmux（长时间训练） |
| 网络 | 远程无外网（不能 pip / huggingface 下载），SSH 偶发不稳定，命令失败就重试 |

### 7.2 启动命令

```bash
# 训练（GPU）
tmux new-session -d -s s7klf4 \
  'python train.py --config s7_klf4_top30_diffonly.json > run_s7_klf4.log 2>&1'

# auto eval（CPU，独立进程，不阻塞 GPU）
tmux new-session -d -s evalcpu \
  'python auto_eval_cpu.py \
    --results-dir 5script/results/s7_klf4_top30 \
    --workers 8 --worker-threads 8 \
    --seen5-csv 5script/seen5_top30.csv'
```

两个 tmux 会话**解耦**：训练在 GPU 上跑，eval 在 CPU 上轮询 `{checkpoint_dir}/*.done`，读最新 ckpt 做 SSIM/MSE，写 `eval_auto_{step}.json`。训练进程的 early-stop 读这些 JSON 决定是否停（`train.py:600-636`）。

`auto_eval_cpu.py` 关键参数（`auto_eval_cpu.py:447-459`）：

| 参数 | 作用 |
|---|---|
| `--results-dir` | ckpt 所在目录（=训练的 `results_dir`） |
| `--workers` | eval100 数据并行进程数（fork 继承，>1 启用） |
| `--worker-threads` | 每 worker 线程数 |
| `--seen5-csv` | seen5 对比 CSV（书家×字seen 组合） |

### 7.3 监控

- 训练日志：`run_s7_klf4.log` + tmux attach `s7klf4`。
- 日志每 `log_every=20` 步输出一行，含 `Total/Diff/...` 各损失、`Mem: <当前>G/<峰值>G`、`EMA: <decay>`、`steps/s`（`train.py:1003-1019`）。
- eval 结果：`5script/results/s7_klf4_top30/eval_auto_*.json` + `eval_<step>/` 对比图。

---

## 8. 当前训练配置（s7_klf4_top30_diffonly.json）

源文件：`configs/s7_klf4_top30_diffonly.json`。

| 参数 | 值 | 说明 |
|---|---|---|
| `experiment_name` | `s7-klf4-top30-diffonly` | 实验 slug |
| `model` | `DiT-2Cond-S/4` | depth=12, hidden=384, heads=6, patch=4 |
| `cond_mode` | `2cond` | callig + char |
| `condition_fusion` | `factorized_add` | callig 128 + char 256 → 嵌影相加 |
| `callig_embed_dim` / `char_embed_dim` | 128 / 256 | |
| `cond_drop_all_prob` / `cond_drop_one_prob` | 0.05 / 0.25 | CFG dropout |
| `image_size` | 256 | |
| `num_calligraphers` / `num_characters` | 1011 / 35130 | |
| `max_steps` | 600000 | |
| `global_batch_size` | 224 | |
| `lr` | 2e-4 | |
| `warmup_steps` | 2000 | |
| `lr_schedule` | `cosine`（`min_lr_ratio=0.1`） | |
| `weight_decay` | 0.02 | |
| `use_ema` | `true`（`ema_decay=0.9999`, `ema_warmup=true`） | |
| `use_lora` / `pretrained` | `false` / `null` | 从零全参 |
| `reset_cond_head` / `train_cond_head` | `false` / `false` | 从零无需 reset |
| `vae` / `vae_path` | `ema` / `pretrained_models/kl-f4` | f4 |
| `vae_downscale` | 4 | |
| `latent_channels` | 3 | |
| `vae_in_channels` / `vae_out_channels` | 3 / 3 | |
| `vae_scaling_factor` | 0.102079 | f4 |
| `bf16` | autocast | |
| `ckpt_every` / `ckpt_keep` | 5000 / 60 | |
| `early_stop` | `true`（`patience=6`, `min_steps=60000`, `metric=ssim`） | |
| `latent_shards_dir` | `final_latents_f4` | 预编码 latent shard |
| `img_root` / `skel_root` | `final_imgs_256` / `final_skeleton` | |
| `preload` / `preload_workers` | `true` / 24 | |
| `auto_eval` / `eval_csv` / `eval_n` / `eval_steps` | `true` / `5script/eval100_top30.csv` / 100 / 50 | |
| `eval_cfg` / `eval_seed` / `eval_batch` | 4.0 / 0 / 20 | |
| `num_workers` / `log_every` | 8 / 20 | |
| **速度** | **3.51 steps/s** | 实测 |
| **VRAM** | **19.74G / 24G** | 实测 |

> 注：`reset_cond_head=false` 是因为从零训练不存在「加载 ImageNet adaLN」这一步，adaLN 随模型整体正常初始化与训练；该开关只在 `pretrained is not None` 时有意义。

---

## 9. 常见问题速查

| 现象 | 可能原因 / 处理 |
|---|---|
| 显存周期性 20G→22G→14G 波动 | `pred_xstart` 计算图未及时释放 → 确认 diff-only 走 detach + `loss_dict.pop`（§6.1） |
| 早期 NaN | 预训练 adaLN 未 reset → `reset_cond_head=true`；或 lr 过高 → 降 lr / 加 warmup |
| EMA 早期发散 | 未开 `ema_warmup` → 设 `ema_warmup=true`，让 `decay=min(d,(1+s)/(10+s))` |
| 存盘时显存尖峰 | 未用 `_state_to_cpu` → 已内置（`train.py:1047-1048`），勿改 |
| eval 不更新 / early-stop 不触发 | `auto_eval_cpu` 未起 / `.done` 未写 → 检查 evalcpu tmux + ckpt 目录权限 |
| ckpt 被删太多 | `ckpt_keep` 过小 → s7 设 60，保留足够回溯点 |
| f4 vs f8 切换报错 | `image_size % vae_downscale != 0` 或 `latent_channels` 未改（§4） |

---

## 10. 变更与维护

- 改训练逻辑：先改 `src/train.py`（源真值），再同步根目录 `train.py`，远程拉取根目录运行。
- 新增超参：在 argparse 注册（`train.py:1114+`），同时加进配置 JSON；`_coerce` 会按类型转换。
- 新增 VAE：补 `--vae-*` 行（§4），`MockVAE` 的 `_vae_ds/_vae_lc/_vae_oc/_vae_sf` 会自动跟随。
- 新增结构 loss：在 `train.py:753-860` 区段扩展，注意 `_need_x0_grad` 与计算图释放（§6.1）。
- 早停指标变更：改 `early_stop_metric`（`ssim`/`mse`），`_es_better` 自动切换比较方向（`train.py:595`）。
