# DiT 训练环境 Infra 文档 (cu121 迁移版)

> 维护人: dsh agent · 最后更新: 2026-08-31
> 主机: `4090` (10.176.54.17:36430, Ubuntu 18.04, glibc 2.27, RTX 4090 24G, driver 590.48 / CUDA 13.1)
> 仓库: `/root/Workspace/xy/DiT` (本地镜像 `G:\GitHub\DiT`)

---

## ⚠ UPDATE 2026-09-16 (torch 已升级到 2.5.1, 本文件其余部分多为历史)

- **cu121 torch: 2.1.2+cu121 -> 2.5.1+cu121**（torchvision 0.20.1 / torchaudio 2.5.1 / triton 3.1.0）。
  2.6+ 需 glibc>=2.28（manylinux_2_28），本机 glibc 2.27，故 **2.5.1 是可用最高版**。
- **xformers 已从 cu121 卸载**：torch>=2.0 自带 SDPA 完全够用（`attn_impl=sdpa` 永远优先 SDPA），
  且旧 xformers 0.0.23 与 torch 2.5 ABI 不匹配。**base 环境的 xformers 0.0.16 未动。**
- 上一行表格里的 "torch 2.1.2 / 停在 cu121 2.1.2" 等描述**已过时**，以本 UPDATE 为准。
- 完整变更/回滚/证据/ MFU 定位见：**`_sync_work/INFRA_UPGRADE_20260916.md`**。
- 备份：`/opt/conda/envs/cu121_bak_20260916`（软链接克隆）；升级前 freeze：`/root/cu121_freeze_before_20260916.txt`。
- `src/train/train.py` 增加了 compile 前的 `pattern_matcher` env 门控（`DIT_PATTERN_MATCHER=1` 可恢复）。

---

## 0. 一页速览

| 项 | base 环境 | cu121 训练环境 |
|---|---|---|
| 路径 | `/opt/conda` | `/opt/conda/envs/cu121` |
| python | 3.10.8 | 3.10.18 |
| torch | **1.13.1+cu117** (别动!) | **2.1.2+cu121** |
| torchvision | 0.14.1+cu117 | 0.16.2+cu121 |
| torchaudio | 0.13.1+cu117 | 2.1.2+cu121 |
| xformers | (残留 0.0.23.post1, 忽略) | 0.0.23.post1 |
| 用途 | 所有旧脚本 / CPU metrics daemon / 数据工具 | **一切训练 (需 `--compile true`)** |

- **Golden Rule: 绝不动 base 的 torch 版本。** 它被当成系统组件供大量旧脚本依赖。
- cu121 是**自包含**环境（自带全部 nvidia-* 库），对系统其他部分零影响；base 里残留的 cu12 库是历史遗留，无需清理。
- 启动训练必须 `export PYTHONPATH=/root/Workspace/xy/DiT:$PYTHONPATH`（无 setup.py / pyproject）。

---

## 1. cu121 环境详情

### 1.1 重建命令（如需从零重建）

```bash
# 1) 创建环境 + 基础 python
conda create -n cu121 python=3.10 -y

# 2) torch 全家桶（官方 cu121 wheel）
/opt/conda/envs/cu121/bin/pip install \
  torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
  --index-url https://download.pytorch.org/whl/cu121

# 3) xformers（必须匹配 torch 2.1.x）
/opt/conda/envs/cu121/bin/pip install xformers==0.0.23.post1 \
  --index-url https://download.pytorch.org/whl/cu121

# 4) 项目依赖（关键 pin）
/opt/conda/envs/cu121/bin/pip install \
  numpy==1.24.4 \
  diffusers==0.27.2 timm==0.9.16 einops scipy==1.15.3 \
  opencv-python==4.7.0.68 scikit-image==0.25.2 \
  transformers==4.36.2 tqdm pandas==2.3.3 lpips==0.1.4 \
  safetensors==0.8.0 PyYAML==6.0 accelerate==0.20.0 omegaconf

# 5) 两个踩坑修复
/opt/conda/envs/cu121/bin/pip install huggingface-hub==0.23.5   # diffusers 需要 cached_download
/opt/conda/envs/cu121/bin/pip install 'setuptools<81'           # accelerate 需要 pkg_resources
```

### 1.2 关键约束

- **glibc 2.27 硬墙**：cu128 全家桶是 `manylinux_2_28`，**无法加载**。故停在 cu121（torch 2.1.2 是 glibc 2.27 能跑的最高 torch 2.x）。
- numpy 必须 **<2.0**（torch 2.1.2 依赖）。
- `attn_impl=sdpa` 在 torch>=2.0 下自动走 **SDPA flash**（不再用 xformers kernel），这是速度提升主因之一。
- 未装 flash_attn（用户拍板：xformers 即可）。

### 1.3 验证命令

```bash
/opt/conda/envs/cu121/bin/python -c "import torch,torchvision,xformers;print(torch.__version__, torchvision.__version__, xformers.__version__, torch.cuda.is_available())"
```

---

## 2. 代码改动（已同步远程，与本地 `G:\GitHub\DiT\src\...` 一致）

### 2.1 `src/train/train.py`
- 新增 `--compile`（bool, 默认 False）、`--compile-mode`（default/reduce-overhead/max-autotune）。
- 在 `model.to(device)` 后、DDP 前：`model = torch.compile(model, mode=...)`。
- EMA 模型**不编译**（eval-only，省编译时间/显存）。
- torch<2.0 时 warn 并跳过（兼容 base 环境运行旧流程）。

### 2.2 `src/train/train_controlnet.py`
- 新增与 2.1 相同的 `--compile/--compile-mode`，注入点也在 EMA deepcopy 与 optimizer 之前。
- **新增 ctrl 早停**：`--early-stop true --early-stop-metric {ssim,mse} --early-stop-patience N --early-stop-min-delta D --early-stop-min-steps S`。
  - 读 CPU daemon 写的 `checkpoints/eval_auto_ctrl_*.json`（`ctrl.ssim` 越高越好），连续 patience 次无改善则停。
  - 原脚本**本来没有早停**，这是本次新增。

---

## 3. 基准数据（s28 同配置冷启动, DiT-2Cond-S/2, flow, latent 32×32×4）

| 环境 | batch | Steps/Sec | Mem (训练侧) | 备注 |
|---|---|---|---|---|
| base (torch 1.13.1) | 192 | ~3.3 | 20.21G | 旧 s28 运行态 |
| cu121 (torch 2.1.2) | 192 | 3.60 | 19.89G | 纯迁移 |
| cu121 + compile | 192 | **8.50** | **9.73G** | 速度×2.6, 显存-52% |
| cu121 + compile | **384** | **4.22** | **18.88G** | 用户拍板 batch（显存顶到 ~20G） |
| cu121 + compile (ctrl s29) | **192** | ~9.7 | ~18G | ctrl 阶段拍板值 |

> 结论：迁移值得。显存大头节省来自 SDPA flash（不再物化完整 attention 矩阵）+ inductor 融合。

---

## 4. 串行训练 pipeline（本次任务核心）

脚本: `/root/Workspace/xy/DiT/_sync_work/run_s28_s29_pipeline.sh`（本地镜像 `G:\GitHub\DiT\_ot_scratch\run_s28_s29_pipeline.sh`）

**流程（串行，默认早停）：**

```
阶段A: train.py      --config s28_std_dino_pretrain.json
                     --compile true --compile-mode default --global-batch-size 384
       + 并行 eval_metrics_daemon.py (CPU, 写 eval_auto_*.json 供早停)
       → 默认早停 (ssim_lpips, patience 5, min_steps 20000)
阶段B: train_controlnet.py --config s29_ctrl_gt_skel_1px.json
                     --main-ckpt <阶段A最新ckpt>   # 自动查找
                     --compile true --compile-mode default --batch-size 192
                     --early-stop true --early-stop-metric ssim
                     --early-stop-patience 5 --early-stop-min-delta 0.002 --early-stop-min-steps 10000
       + 并行 eval_ctrl_metrics_daemon.py (CPU, 写 eval_auto_ctrl_*.json 供早停)
```

**关键点：**
- 阶段A用 base 配置但 `--global-batch-size 384`（CLI 覆盖 config 的 192）。
- 阶段B用 s29 配置但 `--batch-size 192`（覆盖 96）。
- s29 `main_ckpt` 由脚本从 `assets/results/s28_std_dino_pretrain/*/checkpoints/*.pt` 取最新自动填充。
- 每个阶段结束 kill 对应 daemon；daemon 都是 while-True 轮询，不会自己退出。

**tmux 启动（脱离 ssh）：**
```bash
tmux kill-session -t pipeline 2>/dev/null
tmux new-session -d -s pipeline 'bash _sync_work/run_s28_s29_pipeline.sh > /tmp/pipeline.log 2>&1'
```

---

## 5. 日志与产物位置

| 内容 | 路径 |
|---|---|
| pipeline 主日志 | `/tmp/pipeline.log` |
| 阶段A 训练日志 | `assets/results/s28_std_dino_pretrain/<ts>-s28-std-dino-pretrain/log.txt` |
| 阶段B 训练日志 | `assets/results/s29_ctrl_gt_skel_1px/<ts>-s29-ctrl-gt-skel-1px/log.txt` |
| base CPU 指标 | `<exp>/checkpoints/eval_auto_*.json` |
| ctrl CPU 指标 | `<exp>/checkpoints/eval_auto_ctrl_*.json` |
| base ckpt | `<exp>/checkpoints/<step>.pt`（含 ema / optimizer / scheduler） |
| ctrl ckpt | `<exp>/checkpoints/<step>.pt`（ctrl+injections / ema_ctrl / optimizer） |

---

## 6. 监控（agent 手动作法，无脚本）

每 ~30 分钟用 `bash /tmp/monitor.sh` 看一次快照；判断阶段用：
```bash
pgrep -f 'train_controlnet.py'       # → 阶段B
pgrep -f 'src/train/train.py --config src/train'  # → 阶段A
```
- 阶段切换信号：`/tmp/pipeline.log` 出现 `base train 退出` / `阶段B` / `ctrl train 退出`。
- 早停信号：日志出现 `[early-stop] ... early stopping`。
- 显存异常（`nvidia-smi` 显示 20097MiB 但 `--query-compute-apps` 无进程）是该容器环境的已知显示行为，以训练日志 `Mem:` 为准。

---

## 7. 已知注意事项 / 陷阱

1. **pwsh 转义地狱**：在 Windows 本地跑 ssh 命令时，`$()`、嵌套引号、反斜杠都会被 PowerShell 吃掉。一律先 `write` 一个 `.sh` 再 `scp` 过去执行。
2. **base 环境不要 `pip install/upgrade torch*`**——上次就是被误装 torch 2.1.2 到 base 才出的岔子。恢复命令（备用）：
   ```bash
   pip uninstall -y torch torchvision torchaudio
   pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --index-url https://download.pytorch.org/whl/cu117
   ```
3. **多 worker 数据加载**：batch 384 会起 8 个 DataLoader worker 进程，batch 192 起 4 个，均正常。
4. `_bench_*.json` 临时配置已清理；`run_s28_s29_pipeline.sh` 是唯一正式入口。
5. 若需从 checkpoint 续训：base 用 `--resume <ckpt>`；ctrl 用 `--resume <ckpt>`。
