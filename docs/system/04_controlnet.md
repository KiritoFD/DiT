# ControlNet 骨架结构控制

> 对应源码：`src/model/controlnet.py`，训练入口 `src/train/train_controlnet.py`，配置 `src/train/configs/ctrl_skel_s18_flow.json`，评测 `src/eval/inference.py` + `src/eval/eval_ctrl_metrics_daemon.py`。

## 1. 设计：冻结主模型 + 逐层 zero-init 注入

```
        cond(1,256,256) skel
              │
        ctrl_encoder (12× DiTBlockSimple)
              │  逐层
        out_projs[0..11]  ── zero_init_linear(hidden, hidden)  ← 零初始化
              │
   x ── x_embedder+pos ──► DiT main blocks (冻结) ──► + ctrl_feats[i] ──► ... ──► final_layer ──► unpatchify
```

- `ControlNetDiT(main_model, cond_in_channels=1, train_ctrl_only=True)`：
  - `ctrl_encoder`：与主模型同深度的简化 DiTBlock 序列（`DiTBlockSimple`），吃 skel latent + 条件嵌入 `c`。
  - `out_projs`：**全部零初始化** → 训练开始时 ctrl 注入 ≡ 0，主模型行为完全不变，是完美的 warm-start 起点。
  - `forward(x, t, y_callig, y_char, cond=None)`：`cond=None` 时直接退化为主模型 forward（条件 dropout 仍可用）；否则复现主模型 embedding 路径（x_embedder+pos、t_embedder、2cond 条件融合），逐 block 跑主模型并 `x = x + ctrl_feats[i]`。
- **warm-start 内存图策略**：主模型冻结，`train_ctrl_only=True` 时 forward 不建主模型训练图，只对 ctrl 分支建图；每步 del 中间张量 + zero_grad，无 graph 残留。

## 2. 训练两种模式

| 模式 | 配置 | 训练范围 |
|---|---|---|
| **A. warm-start（默认/推荐）** | `train_ctrl_only=true` + `main_ckpt` | 冻结主模型，只训练 `ctrl_encoder` + `out_projs` |
| **B. from-scratch** | `train_ctrl_only=false` | 主模型 + ctrl 一起从零训练；可选 `pretrained` 只载入 body（滤掉条件头） |

`load_main_model(...)`（`src/model/controlnet.py` 顶部有 `import os` —— 重构时曾因缺这个 import 报 NameError，已修复）：

```python
model = DiT_2Cond_models[model_name](num_calligraphers=..., num_characters=...,
        condition_fusion=..., callig_embed_dim=..., char_embed_dim=...,
        char_proj_mode=..., freeze_char_table=...,
        cond_drop_all_prob=..., cond_drop_one_prob=...,
        cond_drop_which_glyph_prob=...,      # ← 与主模型一致，保证 ckpt 结构匹配
        use_checkpoint=..., learn_sigma=True)
```

- ckpt 加载：优先 `ck["ema"]`，回退 `ck["delta"]` 或裸 state_dict；`strict=False` 并打印 missing/unexpected（主模型 ckpt 与重建参数不一致时可第一时间发现）。
- `learn_sigma=True`：输出 8 通道，flow 训练/采样只取前 4（velocity）。

## 3. 训练循环要点（train_controlnet.py）

```python
# 数据：latent shards + 3px skel
ds = MCCDLatentDataset(csv_file=args.csv, latent_shards_dir=..., skel_root=...,
                       image_size=256, load_skel=True, is_train=True, ...)

# 统一时间步采样（第 2 节铁律）:
t = diffusion.sample_t(x_latent.shape[0], device)

# skel 条件 dropout（无需 skel 也能学）:
if args.cond_drop_struct_prob > 0:
    drop = torch.rand(N) < args.cond_drop_struct_prob
    skel = where(drop, zeros_like(skel), skel)

model_kwargs = dict(y_callig=..., y_char=..., cond=skel)
loss = diffusion.training_losses(ctrl, x_latent, t, model_kwargs)["loss"].mean()
```

- 优化器 AdamW + cosine 调度（`lr_lambda` 带 warmup），LR 1e-4，`min_lr_ratio 0.1`。
- EMA 只更新 trainable 参数（ctrl 分支），EMA decay 前 2000 步 ramp（`0.9999*(1-(1-step/2000)^4)`）。
- ckpt：只存 `ctrl_encoder` 权重（`"ctrl"` / `"ema"` 两套），from-scratch 模式另存 `"model"`（`main.*` 前缀）；带 `.done` 标记。
- **in-process GPU eval**：每 `gpu_eval_every` 步对 **GT skel（ctrl）** 与 **无 skel（base）** 各采一组，写 pending marker 交给 CPU daemon（见 `07_eval.md`）。

## 4. CFG（forward_with_cfg）

- skel **始终提供给两半**（结构条件不做 CFG），callig/char 有/无各跑一遍：批次翻倍 `[cond|uncond]`，输出 `[:in_channels]` 前缀组合。
- `cond=None` 时直接走主模型 `main.forward_with_cfg`。
- flow 采样不 clip（velocity 无界）。

## 5. 配置示例（ctrl_skel_s18_flow.json，供参考）

```json
{
  "main_ckpt": "5script/results/s18_s_flow_small/20260827-232003-s18-s-flow-small/checkpoints/0043000.pt",
  "csv": "5script/train_top6.csv",
  "latent_shards_dir": "final_latents",
  "skel_root": "final_skeleton_d3",
  "model": "DiT-2Cond-S/2",
  "diffusion_type": "flow",
  "cond_drop_all_prob": 0.05,
  "cond_drop_one_prob": 0.25,
  "cond_drop_struct_prob": 0.1,
  "train_ctrl_only": true,
  "lr": 1e-4, "batch_size": 152, "max_steps": 30000,
  "gpu_eval_cfg": 1.7, "gpu_eval_steps": 50, "gpu_eval_every": 2500
}
```

> 注意：该配置的 `main_ckpt` 指向旧的 s18 主模型。**下一轮 ControlNet 重训将以 s19 mid-clean 主模型为准**（s19 收敛后）。

## 6. 历史教训

- ~~`torch.randint(0, diffusion.num_timesteps)` 用于 flow 训练~~（t∈{0..49} 插入直线插值 + t*1000 OOD 到 49000）→ **已统一为 `sample_t`**，详见 `02_diffusion.md` §4。坏训练目录已删除，重训从干净起点开始。
- 指标：SkelIoU（骨架二值 IoU，thresh 0.5）、MSE(×4)、SSIM(σ=1.5)、LPIPS(vgg, CPU) —— 由 CPU daemon 计算，与主模型评测同一套约定（`07_eval.md`）。