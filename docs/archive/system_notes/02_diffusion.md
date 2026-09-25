# 统一扩散时间步设计（Flow Matching / DDPM）

> 对应源码：`src/loss/flow_matching.py`、`src/loss/gaussian_diffusion.py`、`src/loss/__init__.py`
>
> **这是整个系统最重要的正确性设计。** 训练代码里出现的同一个变量 `t`，在 flow 和 ddpm 下语义完全不同；历史上曾因此产生过一个直接毁掉训练的 bug（见第 4 节），已通过「统一采样入口」根治。

## 1. 两种扩散的时间步语义

| | DDPM（GaussianDiffusion） | Flow Matching（FlowMatching） |
|---|---|---|
| 训练时间步 `t` | **整数** `t ∈ {0, 1, …, T-1}`，`T=num_timesteps`（默认 1000） | **连续** `t ∈ [0, 1)` |
| 加噪过程 | `x_t = √(ᾱ_t) x0 + √(1-ᾱ_t) ε`（cosine/linear schedule，按 t 查表） | `x_t = (1-t) x0 + t·noise`（直线插值，t=0 纯数据，t=1 纯噪声） |
| 预测目标 | epsilon（或 learned sigma） | velocity `v = noise - x0 = dx_t/dt` |
| 模型输入 | 原始整数 `t`（TimestepEmbedder 内部正弦编码） | **`t × TIME_SCALE`**，`TIME_SCALE = 1000.0`（与 DDPM 相位范围对齐，正弦编码任意浮点皆可） |
| 采样 | DDIM，t: T-1 → 0 | Euler ODE，t: 1 → 0，`dt = 1/steps`，50 步 |
| clip 输出 | 可 clip（预测的是去噪结果方向） | **不可 clip**（velocity 是无界场） |

## 2. 统一采样入口 `sample_t`

两个类提供**接口一致的** `sample_t(n, device)`：

```python
# gaussian_diffusion.py
def sample_t(self, n, device):
    """Uniform integer timestep in [0, num_timesteps)."""
    return th.randint(0, self.num_timesteps, (n,), device=device)

# flow_matching.py
def sample_t(self, n, device):
    """Uniform t in [0, 1)."""
    return th.rand(n, device=device)
```

**铁律：训练调用方只写这一行，绝不自己分支：**

```python
t = diffusion.sample_t(x_latent.shape[0], device)
```

- `train.py`（主模型）：`src/train/train.py` 训练循环（约 line 813）。
- `train_controlnet.py`（ControlNet）：`src/train/train_controlnet.py` 训练循环（约 line 328）。

流程侧在 `FlowMatching.training_losses` 内部做 `t→t*TIME_SCALE` 再喂模型；ddpm 侧喂原始整数。**调用者永远不知道也不关心 diffusion 的类型**，这就是「消灭 bug 类」的位置。

## 3. 训练与采样 t 分布核对（数值验证过的对照表）

| 场景 | 时间步来源 | 模型实际收到的 `t` |
|---|---|---|
| flow 训练单步 | `sample_t` → 均匀 [0,1)，如 0.4979 | `t*1000 = 497.9` |
| ddpm 训练单步 | `sample_t` → randint，如 556 / 677 | 原始整数 556 / 677 |
| flow 采样 (50 步) | Euler 网格 `t0=1.0 → 0.02`，`dt=0.02` | `t_batch*1000` = 1000 → 20 |
| ddpm 采样 | DDIM 抽取 (1000→50) | 整数序列 999 → 0 |
| cfg 采样 | 与上相同（每步对 uncond/cond 各一次） | 相同 |

## 4. ⚠️ 历史 bug 复盘（用户发现）：flow 训练用了 DDPM 的整数采样

### 4.1 症状

`tools/controlnet/train_controlnet.py`（旧路径）训练时用了：

```python
t = torch.randint(0, diffusion.num_timesteps, (batch_size,), device=device)
```

而该次训练配置是 flow（`FlowMatching(num_steps=50)`）→ `num_timesteps = 50`：

- `t ∈ {0..49}` 的**整数**被当作连续 t 使用，插入直线插值 `x_t = (1-t)x0 + t·noise`：t 越大插值越偏向纯噪声；t=0 时却是纯数据 —— 分布完全错乱。
- 模型输入 `t*1000` 最大到 **49000**，远超训练见过的 `[0, 1000]` 区间（flow 主模型训练时 t∈[0,1)→输入∈[0,1000)），**TimestepEmbedder 严重 OOD**。
- 训练 loss 从 ~21 一路降到 ~0.1（看起来在收敛！），但学到的是垃圾：ControlNet 的 SkelIoU ≈ 0.04，LPIPS 比无控制更差 —— **结构控制根本没学到**。

### 4.2 根因

- 调用方在不知道 diffusion 类型的情况下，手工选择了「DDPM 风格」的整数采样。
- 相同 API 名（`num_timesteps`）在两类扩散下语义不同（ddpm=最大噪声级 1000，flow=Euler 步数 50），极易误用。
- loss 下降具有欺骗性：MSE 目标在错误插值路径上也能被压缩，单看 loss 无法发现。

### 4.3 修复

1. 两个类都实现 `sample_t(n, device)`，语义各自正确（ddpm 整数 / flow 连续）。
2. **所有训练调用方统一改为 `t = diffusion.sample_t(...)`**，不再手动 randint。
3. 删除因该 bug 产生的坏训练目录（`20260828-112115-ctrl-skel-s18-flow`）。
4. 代码审查清单加入「时间步相关」检查项：grep `randint` 不得出现在训练循环；统一走 `sample_t`。

### 4.4 同类设计要点（训练/推理一致性）

- **flow 训练与推理的 t 分布已数值验证一致**：训练 t 均匀 [0,1) ×1000 ∈ [0,1000)，推理 Euler 网格 [1000→20] 落在这个区间内（t=1000 是训练的合法上边界）。
- `learn_sigma=True` 的 DiT 输出 2C 通道（latent 4ch → 8ch）：训练时 flow 只取 `[:, :C]` 当 velocity；采样时同理丢弃 sigma 通道。**CFG 只作用在 `[:in_channels]` 前缀**（velocity/eps 子空间），sigma 通道原样保留。
- 采样统一 `clip_denoised=False`：flow 的 velocity 不可 clip；ddpm 由 `ddim_sample_loop` 内部按需处理。

## 5. 工厂与切换

```python
from src.loss import create_diffusion_or_flow
diffusion = create_diffusion_or_flow("50", diffusion_type="flow")   # flow, 50 Euler 步
diffusion = create_diffusion_or_flow("50", diffusion_type="ddpm")   # ddpm, DDIM 50 步
```

- 配置字段 `"diffusion_type": "flow" | "ddpm"`（主模型在训练配置 root；ControlNet 在 ctrl 配置）。
- 默认推理 cfg=1.7（flow 的最佳引导强度，4.0 会过强；s6 ddpm 时代曾用 4.0）。
- flow 相关的训练辅助项（canny/skel 像素损失、x0 结构损失等 DDPM 专属机制）会被 `_flow_disabled` 机制自动关闭，避免语义错配。