# 评测体系

> 对应源码：`src/eval/inference.py`（**唯一推理核心**）、`src/eval/in_process_eval.py` / `in_process_ctrl_eval.py`（训练内 GPU 评测薄壳）、`src/eval/eval_metrics_daemon.py` / `eval_ctrl_metrics_daemon.py`（CPU 指标 daemon）、`src/eval/*.py`（其他壳）。
>
> 设计原则（用户要求）：**核心只写一个 inference，其他都是壳子** —— 采样/解码/落盘/指标只有一份实现，壳脚本只做「配置 + 调用」，杜绝复制-漂移。

## 1. 完整链路

```
训练进程（GPU）                              独立 CPU 进程
─────────────────                          ──────────────────
[每 ckpt 步]
  EMA 换权重（主模型）或 ema_ctrl（Ctrl）
  make_eval_cache：GT图+conds+skel+固定noise（CPU）
  sample_latents：bf16 DiT × ddim_sample_loop
                  （flow=Euler / ddpm=DDIM，统一 core）
  decode_and_save：fp32 VAE → PNG
  （ctrl 模式：{base,ctrl} 两组；主模型：{sample,gt}）
  write pending 标记 ───────────►  轮询 eval_pending_*.json
                                    compute_metrics（CPU: MSE/SSIM/
                                    skelIoU/LPIPS）
                                    write eval_auto_*.json
```

## 2. inference.py —— 核心函数（唯一实现）

| 函数 | 职责 |
|---|---|
| `load_eval_vae(device, vae_path)` | 进程内单例 VAE（sd-vae-ft-ema），fp32，eval 模式 |
| `build_diffusion(steps, diffusion_type)` | `create_diffusion_or_flow(str(steps), ...)` → flow=50 Euler / ddpm=DDIM |
| `sample_latents(model, diffusion, noise, conds, cfg_scale, batch, device, skel=None, seed=0)` | **bf16** 采样；`model` 传 `forward_with_cfg`（CFG 在模型层处理）；`skel` 可选（ControlNet 路径）；固定 noise + 固定 seed，结果可复现；输出 CPU float32 latents |
| `decode_and_save(vae, latents, scaling_factor, out_dir, tag, conds=None, gts=None, vae_batch=16, skels=None)` | **fp32** decode（`lat/0.18215`），输出 `{tag}{i}.png`/`gt{i}.png`/`skel{i}.png` |
| `compute_metrics(dec_dir, gt_dir, tag_prefix, n, use_lpips=True)` | CPU 指标（见 §4） |
| `make_eval_cache(eval_csv, img_root, skel_root, image_size, n, vae_downscale, latent_channels, scaling_factor)` | 预载 GT 图 + conds + skels + **固定 noise**（`Generator(seed=0)`），一次构建全实验复用 |
| `run_pair_eval(model, vae, diffusion, cache, device, step, checkpoint_dir, ...)` | 高层组合：采样→解码→落盘到 `eval_samples_ctrl/stepXXXXXXX/{tag}/` + `samples.json` |
| `write_pending_metrics_marker(checkpoint_dir, step, ...)` | 写 `eval_pending_ctrl_{step}.json`，**必须含 `step_tag` 字段**（daemon 靠它定位目录） |

### 采样约定（关键）

```python
mk = dict(y_callig=yc, y_char=yh, cfg_scale=cfg_scale)
if skel is not None:
    mk["cond"] = skel[i:j].to(device)          # ControlNet 结构条件
with torch.autocast("cuda", dtype=torch.bfloat16):
    samples = diffusion.ddim_sample_loop(model, z.shape, z,
                clip_denoised=False, model_kwargs=mk, device=device)
```

- **不分支**：flow → Euler（velocity 不可 clip）；ddpm → DDIM。`clip_denoised=False` 恒成立。
- cfg 在 `forward_with_cfg` 层执行，仅作用于 `[:in_channels]` 前缀（velocity/eps 子空间），sigma 通道原样。

## 3. 训练内 GPU eval 壳

- 主模型：`in_process_eval.py` 的 `run_gpu_eval`（旧签名）→ `eval_samples/stepXXXXXXX/{sample,gt}.png` + pending → `eval_metrics_daemon.py` → `eval_auto_{step}.json`。训练进程 **换 EMA 权重 → 采样 → 恢复**（不另开 GPU 进程）。
- ControlNet：`in_process_ctrl_eval.py` 的 `prepare_ctrl_eval_cache=make_eval_cache`、`run_ctrl_pair_eval`：
  1. **ctrl 组**：GT 骨架条件（`with_skel=True`，tag="ctrl"）——结构控制上限。
  2. **base 组**：无骨架（`with_skel=False`，tag="base"）——模型本色。
  3. `write_pending_metrics_marker` → daemon 写入 `eval_auto_ctrl_{step}.json`（含 base/ctrl/delta 字段）。

## 4. 指标约定（主模型与 ctrl 完全一致）

| 指标 | 算法 |
|---|---|
| MSE | `mean((pred-gt)^2) **×4**（×4 是为了和旧版像素空间习惯对齐的量纲约定） |
| SSIM | 自定义实现，win=11，σ=1.5，逐通道，均值 |
| skel IoU | 灰度 < 0.5 二值 → skeletonize → 与 GT 骨架（同法得到）逐图 IoU 平均 |
| LPIPS | vgg（CPU，可选；不可用时跳过） |

输出 json 含 mean/std + quartiles（q25/q50/q75）+ n。**对比口径**：以 `eval_strict_top6.csv` 全集（271：楷 169 / 隶 102）为准；严格同分布、固定 noise、固定 seed 保证跨实验可比。

## 5. 默认推理参数（已经配置持久化）

- `eval_cfg = 1.7`（flow 最佳；历史 ddpm 用 4.0，对 flow 过强）—— 默认值落在 `train.py --eval-cfg` 与 `train_controlnet.py --gpu-eval-cfg`。
- `eval_steps = 50`（Euler/DDIM 步数）。
- `eval_n = 271`、`eval_batch = 100`、`eval_vae_batch = 32`。

## 6. 独立评测壳（src/eval/）

| 脚本 | 用途 |
|---|---|
| `eval_auto.py` / `auto_eval_cpu.py` | 主实验的 CPU 批量评测量产（历史路径，指标与 daemon 一致） |
| `auto_eval_gpu.py` | GPU 批采样版 |
| `auto_eval_ctrl.py` / `auto_eval_ctrl_flow.py` | ControlNet 自动评测（flow 版） |
| `eval_gen.py` / `eval_compose.py` / `eval_full_3cond.py` / `eval_models.py` / `gpu_batch_eval*.py` / `eval_metrics.py` / `eval_test.py` / `eval_controlnet_cpu.py` / `sample_controlnet.py` / `test_controlnet.py` / `gradio_controlnet.py` / `backfill_eval.py` / `latent_condition_probe.py` | 采样/对比/回填等场景壳 |
| `eval_ctrl_metrics_daemon.py` | **只认 `step_tag`**：`eval_pending_ctrl_{step}.json → eval_samples_ctrl/{step_tag}/{base,ctrl}/ → eval_auto_ctrl_{step}.json` |

## 7. 当前基线（对比标准，同口径 eval_strict_top6，cfg=1.7）

| 模型 | MSE↓ | SSIM↑ | skel IoU↑ |
|---|---|---|---|
| ddpm s6 @195k（历史） | 0.7872 | 0.5276 | 0.0376 |
| flow s18 @43k | **0.7246** | **0.5476** | **0.0395** |
| s19 mid-clean（训练中） | — | — | — |
| ctrl（s19 收敛后重训） | 预期 skel IoU 大幅提升 | — | — |

> ControlNet 的判别标准是 **ctrl(base 差异) 与 skel IoU**：结构控制有效 ⇔ ctrl 组 skel IoU 显著高于 base 组，且 LPIPS 不劣化。