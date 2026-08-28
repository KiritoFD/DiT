# v2 架构现代化改造（2026-08-29）

> 对应代码：`src/model/modules.py`、`src/model/dit.py`、`src/model/controlnet.py`、
> `src/loss/flow_matching.py`、`src/train/train.py`、`src/train/train_controlnet.py`
>
> 背景与动机：[10_diagnosis_20260828.md](10_diagnosis_20260828.md)（问题分析）
> 关联：[12_dino_diagnosis_20260829.md](12_dino_diagnosis_20260829.md)（DINO 条件实测）

## 0. 一句话

把 2022 年的 DiT 原版实现换成现代组件，把 flow 求解器从一阶 Euler 换成二阶 Heun，
把「会静默污染训练」的几个坑补上。**参数量基本不变（46.22M vs 46.24M）**，
所有改动都可回退，便于 A/B。

## 1. 骨干现代化

新增 `src/model/modules.py`，`DiT_2Cond` 默认启用：

| 组件 | 原版 | v2 | 说明 |
|---|---|---|---|
| 归一化 | `LayerNorm`(无 affine) | `RMSNorm`(无 affine) | 更快更省显存；尺度仍由 adaLN 提供 |
| FFN | `GELU(tanh)` MLP | `SwiGLU` | `hidden = 2/3·4D`，**参数量与 GELU MLP 严格相等** |
| 位置编码 | 固定 2D sin-cos，加到 `x` | 2D axial RoPE，作用于 q/k | 相对位置，无外推问题 |
| Attention | `timm.Attention` | QK-Norm + SDPA | 抑制 logits 爆炸，bf16 更稳 |
| PatchEmbed | `timm` | 自写 | 去 timm 依赖 |

### 回退开关

全部走 config，关掉即与旧实现等价：

```json
"norm_type": "layer", "mlp_type": "gelu", "qk_norm": 0, "rope": 0
```

### RoPE 的 ckpt 兼容性

`rope_cos`/`rope_sin` 注册为 `persistent=False` buffer —— **不进 `state_dict`**，
因此不破坏任何已有 checkpoint 的 key 集合。

⚠ `pos_embed` 参数始终保留（即使 `rope=1` 也不加到 `x` 上），因为
`ControlNetDiT` 通过 `x_embedder.num_patches` 反推 T。

### SDPA 的静默回退（已修）

早期版本写的是：

```python
try:
    _SDPA = F.scaled_dot_product_attention
except AttributeError:
    _SDPA = None          # ← 静默吞掉
```

4090 服务器是 **torch 1.13.1**，没有这个 API，于是 `attn_impl="sdpa"` 悄悄变成
eager —— attention 矩阵被完整物化（B·H·N² 个 fp32 元素），**显存翻倍并 OOM**。

现改为 `resolve_attn_impl()` **显式告警**后回退。宁可吵，不要静默。

> 该服务器上 xformers/flash-attn 均不可用：
> - flash-attn 1.x 只支持 sm_75/80/86，4090 是 **sm_89**，需 2.x，而 2.x 要 torch≥2.0
> - xformers 0.0.16 兼容 torch 1.13，但装它要动环境，且对 N=256 的短序列收益有限

## 2. Flow 求解器

`src/loss/flow_matching.py` 重写：

| 项 | 原版 | v2 | 依据 |
|---|---|---|---|
| t 采样 | `U(0,1)` | **logit-normal**（另有 cosmap） | SD3：均匀 t 在两端浪费梯度预算（t→0 时 v≈−x₀，t→1 时 v≈ε，信息量都低） |
| 求解器 | Euler（一阶） | **Heun**（二阶 RK2） | 截断误差 O(dt) → O(dt²) |
| schedule | 均匀 | 支持 **timestep shift** | 默认 `shift=1.0` 不 shift（本项目细节主导，见下） |
| NFE | = steps | **= 2 × steps** | 新增 `nfe` 属性便于等算力对比 |

实测 t 分布变化：`p5/p95` 从 `0.049/0.949` 收窄到 `0.166/0.838`。

### shift 为什么默认 1.0

SD3 的 `shift=3.0` 是针对**高分辨率**（1024²+）设计的：把步数向高噪声端偏移，
让布局先定下来。本项目是 32×32 latent（256px 字形），笔画末端、飞白这些细节
在 t→0 形成，属于**细节主导**，shift 反而可能有害。若要做实验请显式写进 config 记录。

### heun_batch 的一个坑（已修）

`heun_batch` 把两个 RK stage 沿 batch 维拼成一次 forward，但 `model_kwargs` 里的
`y_callig`/`y_char`/`cond`/`g` 仍是 B。而 `forward_with_cfg` 假定
`x.shape[0] == y.shape[0]`，不复制会在 `c = t_emb + y_emb` 处 batch 不匹配。
新增 `_tile_kwargs()` 处理。

### train/eval 配置一致性（已修）

新增 `flow_kwargs_from(args)`，让 train.py / in_process_eval / inference /
auto_eval_ctrl_flow 共用同一份配置。**此前 eval 侧完全没传**，会静默退回默认参数。

## 3. 修掉的「静默污染」类问题

这几个都不是性能问题，而是**错了也不报错**：

| # | 问题 | 后果 |
|---|---|---|
| 1 | eval 异常时 `except` 分支只调 `model.train()`，没恢复 `_orig_sd` | 训练从 EMA 权重继续，而 Adam 动量仍对应旧权重；`_orig_sd` 泄漏显存 |
| 2 | ControlNet 主 ckpt 路径失效时 `os.path.exists(...) or None` | **静默用随机初始化的主模型训练** ControlNet |
| 3 | flow 下 `learn_sigma` 默认 `True` | final_layer 多输出 C 个通道，被 `[:, :C]` slice 掉 → 零初始化且**永不收梯度** |
| 4 | 早停用严格不等式 | 任何 ±0.0001 的噪声级抖动都算「改善」并重置计数器 |
| 5 | `attn_impl="sdpa"` 在 torch 1.13 静默回退 eager | attention 矩阵被物化，显存翻倍 OOM |
| 6 | `w.requires_grad_(False); w[-1].requires_grad_(True)` | **no-op**（非叶子张量）→ CFG null token 从未可训练 |

第 6 条影响面最大：null token 承担 4-way dropout 中约 30% 的样本
（cond_drop_one 25% + cond_drop_all 5%），是 CFG uncond 分支的核心。
改为独立 `nn.Parameter` + `torch.where` 覆盖，零额外拷贝开销。

## 4. ControlNet 重写

| 项 | 原版 | v2 |
|---|---|---|
| RoPE | ❌ 不透传 | ✅ 透传（旧代码 `block(x,c)` 让主模型**静默丢失位置信息**） |
| 注入方式 | `x + feat` | `x*(1+s) + t`（adaLN 式，同样 zero-init 恒等 warm-start，但可增强也可抑制） |
| null condition | zeros | gaussian（零 latent 解码是特定灰块，不是空白） |
| 主 ckpt | 失效时静默随机 | 硬断言 |
| EMA | `zip(parameters())` | `named_parameters()` 名字匹配（zip 长度不一致会静默截断） |
| 架构一致性 | 无约束 | norm_type/mlp_type/qk_norm/rope 必须与主模型训练配置一致 |

## 5. batch 选型实测

`tools/bench_batch_s20.py`（忠实复刻 train.py 训练步：bf16 autocast + AdamW + clip + EMA）：

| ckpt | batch | 峰值 | steps/s | img/s |
|---|---|---|---|---|
| False | 96 | 12.08G | 5.63 | **541** |
| False | 128 | 15.84G | 4.11 | **526** |
| False | 160 | 19.60G | 3.16 | 505 |
| False | 192 | OOM | | |
| True | 160 | **2.67G** | 2.42 | 387 |
| True | 448 | 6.38G | 0.85 | 380 |

**两个结论：**

1. **SwiGLU 的激活显存代价**：参数量与 GELU 相同，但**激活显存翻倍**
   （需同时保存 gate/up 两支输出给反向）。所以旧架构能跑 batch=240，
   v2 架构在 192 就 OOM。这是 SwiGLU 的真实成本，选型时应计入。

2. **不开 checkpoint 更快**：开 ckpt 后显存降到 2.67G，但吞吐恒定在 380 img/s
   （完全 compute-bound，加大 batch 无收益），比 no-ckpt 的 526 慢 **28%**。

→ 选 **no-ckpt + batch=128**（15.84G，留 7.7G 给 eval）。

## 6. 回归测试

`tests/test_arch_v2.py`，7 组断言，CPU 可跑，本地与远程 4090 均通过：

```
1. 组件：RMSNorm/SwiGLU 参数量对齐、RoPE 保范数、Attention 形状
2. DiT_2Cond：四种配置（modern/legacy）均可前向
3. FlowMatching：插值端点、1 步 Euler 精确性、heun_batch 的 kwargs 复制
4. schedule：shift 单调性
5. ControlNet：注入/Gaussian null/冻结主模型/ckpt 断言
6. 梯度流：adaLN-Zero 性质 → step1 只 inj 有梯度 → step2 传导到 ctrl blocks
7. DINO 冻结：表冻结 + null token 真的能拿到梯度
```

## 7. 下一步

v2 架构改动解决的是「把现有信号用得更好」，**没有改变 DINO CLS 本身字符信息
薄弱**这个事实（跨书体检索 top-1 只有 1.9%~2.6%，见
[12_dino_diagnosis_20260829.md](12_dino_diagnosis_20260829.md)）。

零样本的根本解法仍是**标准字形 latent 条件**（`w_glyph_cond` +
`std_glyph_latent_v2/`）—— 那里编码的是字的形状本身，而不是全局图像描述子。
基础设施已就绪，差接线。
