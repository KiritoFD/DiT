# Callig-DiT：进一步的 Infra 优化建议

看过当前源码的训练热路径后，我认为你们在 RTX 4090 上还有值得挖掘的性能空间，但方向应该和普通 DiT 项目有所不同。

你们已经启用了 BF16、SDPA、`torch.compile`、fused AdamW、latent preload 和异步 checkpoint。因此，下一阶段的重点不是换更快的 Attention，而是减少数据往返、GPU 同步、辅助 Loss 的重复计算，以及训练之外的 GPU 占用。

我会按实际收益潜力，将优化分成三档：

| 优先级 | 方向                                    | 我的判断                  |
| --- | ------------------------------------- | --------------------- |
| P0  | Critic/反事实训练的 CPU↔GPU 往返、动态 shape 和同步 | 最值得立即优化               |
| P0  | REPA 特征缓存、RAM 使用方式和 H2D 传输            | 基模训练中值得重点测试           |
| P1  | `torch.compile` 图稳定性、训练 step 编译粒度     | 可能有收益，必须用 profiler 验证 |
| P1  | 梯度探针、EMA、checkpoint 和训练内 eval         | 主要降低额外开销              |
| P2  | CUDA Graph、算子融合、自定义 Triton kernel     | 当前没有充分证据值得优先开发        |

这里有个重要判断：对单张 4090，优化端到端 samples/s 比单纯提高 GPU utilization 或理论 MFU 更重要。 特别是你们会在 FM、REPA、Writer Critic、反事实损失之间切换，单一 MFU 数字不足以反映真正的实验成本。

## 一、最明确的热点：Critic 训练中存在 GPU→CPU→GPU 往返

查看 `src/train/train.py:2299–2356`，当前流程近似为：

```
GPU：采样时间 t
       ↓
GPU：生成有效时间步 mask
       ↓
CPU：把 mask 搬回来
       ↓
CPU：按 mask 切出 STD 条件和书家标签
       ↓
CPU：writer/script ID → Python 字符串 → slot
       ↓
GPU：重新创建索引 Tensor
       ↓
GPU：运行 Critic
```

其中：

```
if _cmask.any():
```

会使 Python 等待 GPU 判断结果。

随后：

```
_cmask_cpu = _cmask.cpu()_cw_id.tolist()_sc_id.tolist()
```

再次发生设备同步和 CPU 参与。

接着，对每个样本执行：

```
_lut[(writer, script)]slots.index(...)
```

最后还要用 Python list 在 GPU tensor 上做动态索引。

这条路径对于一个小型 1.06M 参数的 Critic 来说，调度开销可能不成比例。

### 我推荐的改法：全 GPU 标签映射 + 静态形状

在训练启动时构建整数查表：

```
slot_lut = torch.full(    (num_writers, num_scripts),    -1,    dtype=torch.long,    device="cuda",)# 只在启动阶段根据字符串标签填充slot_lut[writer_ids, script_ids] = slot_ids
```

训练时直接：

```
target_slot = slot_lut[y_writer, y_script]assert_valid_slots_preflight(...)
```

不再做运行中的 Python 字符串查表。

再把 Critic Loss 改成 `reduction="none"`，使用 GPU 上的时间掩码：

\\[ L\_{\rm critic} = \frac{\sum_i m_i\ell_i} {\sum_i m_i+\epsilon} \\]

这样可以避免依赖 Python 的 `if mask.any()` 控制流程。

第一版建议测试整个 batch 进入 Critic，再用 mask 计算 Loss。虽然会多算一部分 Critic forward，但有机会通过固定 batch shape、减少 CUDA 同步和避免动态编译图获得更好吞吐。

这不是必然更快，因为 Critic 的计算量确实会增加。应与当前的“动态筛选有效子批”实现做相同 batch 的实测 A/B。

尤其建议同时记录：

- 每步 CUDA 同步次数；
- Critic forward/backward 时间；
- GPU kernel 之间的空闲时间；
- `torch.compile` 产生的 graph 数量和 recompilation 次数；
- 端到端 samples/s。

验收要求：两个实现的有效样本 Loss 与梯度一致，然后才比较性能。

## 二、强风格后训练还有一个更严重的同步热点

我认为这里甚至比旧 Critic 路径更值得优先修。

涉及：

- `src/train/strong_style.py:64–155`
- `src/train/gradient_budget.py:24–46`

### 1. `StrongStyle` 逐样本读取 GPU 时间步

当前代码：

```
for i, (w, s) in enumerate(...):    ...    args.critic_t_min <= float(t[i]) <= args.critic_t_max
```

`t` 在 GPU 上。`float(t[i])` 会把该标量同步到 CPU。

如果 batch=192，且大部分书家槽位有效，这段逻辑就可能在一个训练 step 内触发上百次 GPU→CPU 同步。

这是明确的源码问题，不需要等 Nsight 才能判断它应该被消除；但它究竟造成多少毫秒开销，还需要实测。

建议改为：

```
valid_slot = slot_lut[writer, script] >= 0time_valid = (    (t >= t_min)    & (t <= t_max))eligible = valid_slot & time_valid
```

之后使用 GPU 索引。

随机选取负书家也应使用启动时构造的整数候选矩阵，不再通过：

```
torch.randint(...).item()
```

反复回到 Python。

### 2. 梯度预算模块即使不记录日志，也可能每步同步

`src/train/gradient_budget.py:45–46`：

```
messages.append(    "%s=%.6g weight=%.6g ratio=%s" % (        name, loss.detach(), scale, ...    ))
```

这里 `loss.detach()` 仍然是 CUDA Tensor。

格式化为浮点字符串会触发标量读取，而 `messages` 每个 step 都构建，并不只发生在日志周期。

应该将数值记录改为：

```
metrics["writer_loss"] = loss.detach()
```

只在真正写日志时统一进行一次 CPU 读取。

另外，`GradientBudget.balance()` 在 probe 时还会用 CUDA Tensor 做 Python 条件判断与浮点转换。这些操作可以保留在低频诊断流程，但应明确与正常训练分离。

### 3. 建议实现 `GpuMetricsAccumulator`

让所有 Loss 组件只返回：

```
{    "loss": loss,    "writer": writer_loss.detach(),    "brush": brush_loss.detach(),    "active_count": active.sum().detach(),}
```

`TrainStep` 在 GPU 上累加指标，logger 每 50 或 100 步读取一次。

我建议把它设为一条全局规则：

正常训练热路径中禁止 `.item()`、`float(cuda_tensor)`、`.tolist()`、`bool(cuda_tensor)`；只有显式标记的同步边界允许使用。

可以在 CI 中对训练热路径做静态扫描，并对已批准的例外添加注释。

## 三、反事实训练：减少第二次 DiT forward 的调度成本

你们的 C 分支是：

```
za = predict(writer_a)zb = predict(writer_b)
```

两次使用完全相同的 `xt`、时间步和 STD，只改变 writer。

这提供了一个很自然的计算优化机会：

```
xt_pair = torch.cat([xt, xt], dim=0)t_pair = torch.cat([tt, tt], dim=0)cond_pair = combine_conditions(    writer_a,    writer_b,    same_std=g,    same_script=script,)z_pair = predict_once(xt_pair, t_pair, cond_pair)za, zb = z_pair.chunk(2, dim=0)
```

这将两次 DiT 调用合并为一次更大的 batch。

可能的收益有：

- 减少 Python 和 CUDA kernel launch 开销；
- 提高较小反事实子 batch 的 Tensor Core 利用率；
- 更容易维持固定的编译形状；
- 避免重复执行一部分与 writer 无关的条件准备。

但注意：这不意味着计算量减少一半。Transformer 仍然要计算两套 writer 条件下的输出，主要减少的是调度开销和某些可共享操作。

对 24GB 显存，必须比较峰值内存。建议只在反事实 batch 较小的情况下测试，观察 `K=16、32、64` 的吞吐和显存。

不要直接复用正常 FM forward 的 `pred_xstart` 代替第一条反事实 forward。 当前反事实分支特意使用完整 writer 条件、关闭独立条件 dropout，并重新构造时间步。直接复用会改变实验语义，必须单独作为算法消融，而不是声称数值等价的 Infra 优化。

### `model.eval()` 也值得单独检查

`strong_style.py:104–145` 会在辅助 forward 前将整个模型切换到 eval，然后逐个恢复模块的 training 状态。

对于 `torch.compile`，训练态和 eval 态可能对应不同的计算图；如果反事实 batch 大小同时变化，就会增加编译版本和维护开销。

[image](https://www.google.com/s2/favicons?domain=https://docs.pytorch.org\&sz=32)

PyTorch 2.14 documentation



更理想的设计是让条件模块显式支持 `drop_condition=False`，不必切换整个模型的模式。不过这必须确认模型中所有 dropout、归一化和条件行为的等价性，不能仅删除 `model.eval()`。

我建议先用：

```
TORCH_LOGS="recompiles,graph_breaks"
```

观察首次 warmup 后是否仍在反复编译，再决定是否改这个接口。

## 四、REPA Cache：一个可能几乎零损失的优化

涉及 `src/utils/dino_cache.py:25–118`。

我发现你们的 REPA 特征缓存优先使用：

```
feats.f32
```

而源码文档注明，这个文件是通过 `convert_dino_cache_f32.py` 从 `feats.f16` 转换得到的。

如果当前训练所用的 `feats.f32` 确实只是 FP16 数据上转换而来，那么 FP32 文件并没有增加任何特征精度，却让缓存和 H2D 流量翻倍。

例如，B=288，特征形状 `[288,256,384]`：

现有 FP32 传输

# 108 MiB

每个训练 batch

改为 FP16 缓存

# 54 MiB

每个训练 batch

建议：

1. 抽取相同 ID 的 1,000–2,000 条特征，验证 `feats.f16.astype(float32)` 与当前 `feats.f32` 是否逐元素相等。
2. 相等则优先加载 FP16 缓存。
3. GPU 上用 FP32 进行原来的相似度计算，保持 REPA 数值语义。
4. 对比单步 Loss、梯度以及稳定吞吐。
5. 验证通过后，才把 FP16 设为默认并移除重复的 FP32 文件。

如果 FP32 缓存确实来自更高精度的原始计算，不能直接假定这是一项无损优化。

### 进一步：不要永久 pin 住几十 GB 内存

当前实现将全量 REPA 缓存复制到 pinned RAM：

```
_t = torch.from_numpy(np.array(_mm))self._feats = _t.pin_memory()
```

这至少涉及普通 RAM 副本和 pinned RAM 副本的构建过程。

251 GB 内存让这种方式在容量上可行，但系统 RAM 多，不等于锁页内存越多越好。Pinned Memory 对操作系统内存管理有代价，PyTorch 文档也明确提醒不能过度使用。

[image](https://www.google.com/s2/favicons?domain=https://docs.pytorch.org\&sz=32)

PyTorch 2.14 documentation



我建议比较：

| 方案                          | 全量缓存          | Batch staging   | 特点           |
| --------------------------- | ------------- | --------------- | ------------ |
| A：现状                        | FP32 pinned   | FP32 pinned     | 最容易对照        |
| B：FP16 pinned               | FP16 pinned   | FP16 pinned     | 首推，直接减少一半数据量 |
| C：FP16 普通 RAM               | FP16 pageable | 双 pinned buffer | 减少长期锁页规模     |
| D：FP16 pinned + CUDA stream | FP16 pinned   | 双 pinned buffer | 尝试与计算重叠      |

优先做 B。如果收益已经充分，未必需要 C/D。

### CUDA 双缓冲应该如何接入？

当前 `DinoFeatureCache.gather()` 特意使用同步 `.to(device)`，原因是它会复用同一个 `_out_buf`。

这个考虑是正确的：简单改成 `non_blocking=True`，下一次 gather 就可能覆写尚未传输完的 pinned 数据。

正确的预取器至少需要两个 staging buffer，分别对应当前 batch 和下一 batch：

```
CPU staging[0] → CUDA copy stream → GPU batch n
CPU staging[1] → 等待 staging[1] 可安全复用
                                  ↓
                         GPU 计算 batch n
                                  ↓
                 copy stream 预取 batch n+1
```

用 CUDA Event 或 stream 依赖管理 buffer 生命周期，避免复制未完成时覆写。

还应确保计算 stream 等待 H2D 完成、GPU 张量的生命周期得到正确管理。单纯 `non_blocking=True` 并不能自动实现传输与 GPU 计算重叠。

[image](https://www.google.com/s2/favicons?domain=https://docs.pytorch.org\&sz=32)

PyTorch Tutorials 2.14.0+cu130 documentation



我建议先完成 FP16 Cache 验收，再上双缓冲。 否则很容易花大量工程时间解决一个只占整步几毫秒的问题。

## 五、C2OT：当前基模仍有一条同步 CPU 求解路径

`v84A_std_cond.json` 中：

```
"use_c2ot": true,
"c2ot_mode": "slot"
```

而 `src/loss/flow_matching.py:221–280` 的实现每步会：

1. 将真实 latent 和噪声从 GPU 搬到 CPU；
2. 将条件索引搬回 CPU；
3. 使用 NumPy 构造各书家×书体槽的距离矩阵；
4. 调用 SciPy `linear_sum_assignment()`；
5. 把匹配结果传回 GPU。

也就是说，C2OT 是训练中明确存在的一条同步 CPU 路径。

你们当前实现已经比最早的逐组 GPU→CPU 传输好很多，但仍值得测量。

### 这里 Zen2 的 64 核怎么用？

我的第一建议不是并行运行 64 个 Hungarian Solver，而是检查 NumPy BLAS 的线程数。

因为每个 writer×script 组通常不大，`np.dot()` 计算的是很多个小矩阵。让 OpenBLAS 对每个小矩阵启动几十个线程，可能反而增加线程调度成本。

建议测：

```
OPENBLAS_NUM_THREADS=1
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
```

对比 BLAS 线程数 1、2、4、8 时的 C2OT 耗时。

至少拆成三个时间段：

| 阶段     | 说明                           |
| ------ | ---------------------------- |
| D2H    | GPU latent、noise、keys 拷回 CPU |
| Solver | NumPy 距离计算 + SciPy 匹配        |
| H2D    | 将 permutation 传回 GPU         |

如果 C2OT 只占整步约 1–2%，我建议保持现状，不要重新实现 GPU Hungarian。

如果占比超过 5%，才考虑将下一批的 C2OT 计算与当前 GPU step 流水线重叠。

注意这种预处理不能简单移到 DataLoader：目前噪声由 GPU 端随机生成。若改为 CPU 提前生成，虽然仍可保持高斯分布，但随机序列与随机数生命周期改变了，属于需要重新验证的实现变更。

另外，是否保留 `use_c2ot=true` 是算法消融，而不是 Infra 等价优化；不能为了提速直接关闭后声称模型训练完全相同。

## 六、额外发现：`heun_batch` 现在存在重复模型计算

这是本轮进一步阅读采样器时发现的一个具体问题。

位置：

`src/loss/flow_matching.py:535–547`

代码写成：

```
v1 = self._v(model, x, t_batch, model_kwargs, C)x_euler = x + dt * v1x_cat = torch.cat([x, x_euler], dim=0)t_cat = torch.cat([t_batch, t2], dim=0)v_cat = self._v(model, x_cat, t_cat, kw_cat, C)v1_, v2 = torch.split(v_cat, B, dim=0)
```

你们已经算过一次 `v1`，第二个 `2B` forward 又重新算了一次 `v1_`。

因此：

| Heun 实现       | 模型 forward | B-equivalent NFE |
| ------------- | ---------- | ---------------- |
| 正常 Heun       | B + B      | 2                |
| 当前 heun_batch | B + 2B     | 3                |

当前 `heun_batch=True` 的理论网络计算量比普通 Heun 多约 50%，而且并没有减少 forward 调用次数。

Heun 的 corrector 状态需要先知道 predictor 的输出，因此这两个阶段本来就不能直接无条件并行成一次 forward。

好在 `heun_batch` 默认关闭，我没有在本次上传的实验 JSON 中查到显式开启它的配置，所以没有理由认为 v84A 的历史评测因此被拖慢。

建议：

- 将当前实现标记为实验性/废弃；
- 默认保留普通 Heun；
- 删除文档里“把两个 RK stage 合并成一次 forward”的错误描述；
- 后续可以优化 CFG 的 conditional/unconditional batch 合并，但不要与 Heun stage 合并混淆。

这是一个值得和这次重构任务一起修掉的隐藏性能陷阱。

## 七、利用 251 GB RAM：把 Dataset 改成批量索引，而不是增加 worker

当前 `src/utils/latent_dataset.py:560–713` 的 `__getitem__()` 每读取一个样本，都要完成多次 Python 字段处理和 Tensor 构造，包括：

- 从 CSV 行提取 `img_id`、书家、书体、字符等字段；
- 查找 latent；
- 获取随机 σ 的 STD；
- 创建多个标签 Tensor；
- 返回一个包含 `image`、`g`、`std_soft`、`canny`、`skeleton` 等字段的字典。

主线 STD-only 模型并不使用其中所有字段，但通用 Dataset 仍为这些路径准备了对象。

对于 B=288，这意味着每个 batch 有数百次 Python 调用，以及上千个小 Tensor 对象的构造或处理。

### 你们已经有一个值得保留的优化

`StdSoftGlyphStore` 已预先生成 13 档 σ 的模糊结果：

\\[ 13\times8451\times64\times64 \approx450\text{ MB} \\]

这个优化是正确的。不要再把 Gaussian blur 挪回每张图实时执行。

### 下一步：Vectorized Batch Fetch

把静态标签和索引在 Dataset 初始化时转成连续数组：

```
writer_ids = np.asarray(...)script_ids = np.asarray(...)char_ids = np.asarray(...)std_row_ids = np.asarray(...)latent_rows = np.asarray(...)
```

训练时直接按整个 batch 的索引收集：

```
def get_batch(indices):    z = latent_bank[indices]    writer = writer_ids[indices]    script = script_ids[indices]    sigma_levels = sample_sigma_levels(len(indices))    g = std_bank[sigma_levels, std_row_ids[indices]]    return pack_batch(z, g, writer, script)
```

这个设计减少 Python per-sample 逻辑，也避免重复创建无用的空张量。

推荐保留 STD 为 uint8，搬到 GPU 后再转换为 float32 / 255。

B=288 时，STD 的 H2D 数据从约 4.5 MiB 降到约 1.1 MiB；同时减少 CPU 端逐样本浮点转换。

要保持原有 σ 增强分布，必须继续先采样连续 σ，再映射到现有 0.05 间隔网格，不能直接均匀采样网格索引后声称完全等价。

### 为什么不是把 `num_workers` 提到 32？

源码 `train/data_setup.py:184–200` 已记录过一组实际测试：

| Workers | 数据准备耗时      |
| ------- | ----------- |
| 0       | 68 ms/batch |
| 4       | 40 ms/batch |
| 8       | 24 ms/batch |

这是历史合成测试，不是当前 latent-only 主训练的精确耗时，但说明此前选择 8 workers 有实际依据。

我建议先比较：

现有逐样本 Dataset + 8 workers 与 批量索引 Dataset + 2/4/8 workers。

如果新接口能降低 CPU 端 batch 组装时间，但原先这些时间已经完全被 GPU 计算重叠，端到端训练可能不会明显更快。

因此验收必须看 DataLoader wait time 和总 samples/s，不能只看 `__getitem__` 的 microbenchmark。

### Zen2 的 NUMA 优化

64C/128T Zen2 需要先确认是不是多个 NUMA 节点，以及 GPU 连接在哪个 PCIe Root Complex。

先执行：

```
lscpu
numactl --hardware
nvidia-smi topo -m
```

如果是多 NUMA 节点，优先测试训练进程、CPU staging 和 DataLoader workers 靠近 GPU 本地 NUMA 节点的配置。

但不要直接把全部 251 GB RAM 都绑定到同一 NUMA 节点。应该根据实际节点容量、REPA 缓存规模和其他负载决定。

## 八、`torch.compile`：重点减少图变体，不是盲目换 mode

当前模型：

```
torch.compile(model, mode="default")
```

还针对旧 PyTorch 的 Inductor 问题关闭了：

```
_ind_cfg.pattern_matcher = False
```

并提高 Dynamo 累积缓存上限。

这些都是已有的历史优化，不能简单认为打开 `max-autotune` 就会更快。

我建议只测以下三个配置：

| 配置                           | 目的                      | 注意       |
| ---------------------------- | ----------------------- | -------- |
| 当前 `default`                 | 对照                      | 保持历史条件   |
| `max-autotune-no-cudagraphs` | 检查 GEMM/autotune 收益     | 首次编译可能更久 |
| `reduce-overhead`            | 减少 CUDA launch overhead | 显存占用可能增加 |

对于 B=288 的主干训练，较大的 GEMM 已能有效使用 Tensor Core，因此 `reduce-overhead` 未必有明显收益。

但在 B=16–64 的反事实子 batch 中，Python 和 kernel launch 占比可能更高，值得单独测试。

PyTorch 官方也明确指出 CUDA Graph 带来的低开销模式有额外内存成本，并非所有图都能被成功捕获。

[image](https://www.google.com/s2/favicons?domain=https://docs.pytorch.org\&sz=32)

PyTorch 2.14 documentation



我更建议优先消除动态 batch、Python 条件和 train/eval 模式切换，再测试编译模式。

另外，你们当前 Attention 已使用 SDPA。按主干的粗略 FLOPs 模型，Attention 的二次项远小于 MLP/Linear 的参数相关矩阵乘，因此不要指望简单安装 FlashAttention 包带来 2× 整体加速。先确认 SDPA 实际使用的 CUDA kernel。

## 九、EMA、Checkpoint、Eval：重点减少训练被打断的时间

这部分我不建议大改算法，但值得加上完整的时间统计。

### EMA

目前每 4 步执行一次 EMA 更新。

保留现状作为对照，不应未经验证就换成每步 EMA 或 CPU EMA。CPU EMA 可能节省一些显存，却引入频繁的模型参数 PCIe 传输。

### Checkpoint

当前所谓异步保存，实际上分为：

- 同步阶段：把模型、optimizer、EMA 状态复制到 CPU；
- 异步阶段：后台序列化、写盘和计算文件哈希。

因此它并不是完全无阻塞。

不过 checkpoint 每 5k 步才发生一次，即使这次同步花几十到几百毫秒，摊销到整个训练通常也很小。

我建议优先保证保存正确性，不优先开发复杂的异步 GPU snapshot。

### 训练内评测

当前配置每 5k 步保存并运行一次 `eval200fix + seen20`，使用 Heun-50。

这会暂停训练，让 GPU 转去做推理。

对于准备跑 180k–200k 的新基模，我建议按用途分级：

| 训练阶段     | 评测策略                     |
| -------- | ------------------------ |
| 0–40k    | 每 5k：固定 20–40 字快速视觉与内容检查 |
| 40–160k  | 每 10k：固定 187 字完整评测       |
| 160–200k | 每 5k：完整评测                |
| 最终候选     | 三 seed + 完整风格/内容指标       |

这个方案可能减少在训评测占用，但要先检查它的真实 wall-time 占比。如果评测只占总训练不到 1%，就没必要为这项调度大改基础设施。

由于只有一张 GPU，不要让独立 eval 进程与主训练同时争抢同一张 4090。可以让 CPU 在后台计算 SSIM、墨量、BSD 等指标，但使用 GPU 的模型采样最好错开执行。

# 十、最值得实施的优化矩阵

下面的收益是待验证的性能目标或潜力判断，不是已经测出的加速比。

| 优化                            | 主要场景       | 性能潜力  | 开发风险 | 推荐     |
| ----------------------------- | ---------- | ----- | ---- | ------ |
| StrongStyle 去逐样本 CUDA 同步      | B/C/D 后训练  | 高     | 低～中  | 立即做    |
| 旧 Critic 全 GPU slot 查表        | Critic 后训练 | 中～高   | 低    | 立即做    |
| `GradientBudget` 延迟日志转换       | 强风格后训练     | 中     | 低    | 立即做    |
| FP16 REPA Cache               | 基模训练       | 中     | 低    | 优先做    |
| C2OT BLAS 线程数与同步 profiling    | 基模训练       | 不确定   | 低    | 优先测    |
| Batch Fetch / uint8 STD       | 基模训练       | 中     | 中    | 值得做    |
| 反事实合并 forward                 | C 分支       | 中     | 中    | 值得测    |
| `torch.compile` 图变体治理         | 所有训练       | 中     | 中    | 值得做    |
| REPA H2D 双缓冲                  | 基模训练       | 中～低   | 中高   | 测出等待再做 |
| 修改 CUDA Graph 模式              | 所有训练       | 不确定   | 中高   | 后置     |
| 自定义 Attention / Triton kernel | 基模训练       | 目前不明确 | 高    | 暂缓     |

## 十一、我建议用两天工程时间，先回答能快多少

不要在重构同时连续做十项 CUDA 优化。建议建立一个独立性能分支，每次只改变一项，保证能准确归因。

### Day 1：先消除明显的人为开销

上午：StrongStyle / Critic 热路径

消除逐样本 CUDA 标量同步、CPU 字符串映射和每步 Loss 格式化。测试 Writer-only 与反事实 C，比较输出 Loss、梯度和 samples/s。

下午：REPA 和 C2OT

核对 f16/f32 特征是否等价；测试 FP16 缓存；C2OT 比较 BLAS 线程数和同步耗时。

### Day 2：验证端到端收益

上午：数据与 compile

比较 Dataset Batch Fetch、DataLoader worker 数，以及编译图变体。固定真实样本顺序，不改变 Loss、VAE 或条件。

下午：完整训练 microbenchmark

对 v84A 类基模和 v85 强风格训练各执行短窗口测量，记录吞吐、显存、GPU/CPU 时间、编译次数与训练 Loss 对齐情况。

### 测试规范

建议每项配置：

- 先完成 compile warmup（例如 100 步，或以不再出现重新编译为准）。
- 正式测量 200 步，重复 3 次。
- 用 `time.perf_counter()` 测端到端耗时，CUDA Event 测 GPU 片段。
- 只在测量窗口边界同步 GPU，不要在每个 step 尾部额外调用 `torch.cuda.synchronize()`。
- 固定 batch、AMP、数据顺序、C2OT、Loss 和模型状态。
- 报告中位 step ms、p95、samples/s、显存峰值及 CUDA 同步次数。

现有训练器已经支持 `--profile-steps` 和 `--profile-skip`，可以直接复用；建议另外做一个专用 `tools/bench/train_step.py`，避免 benchmark 每次执行完整训练器的 checkpoint 和评测生命周期。

对 CUDA kernel 层面的问题，可以再使用 Nsight Systems；仅当确认某个 kernel 是热点时，才上 Nsight Compute。

## 十二、实际能节省多少 GPU 时间？

以 200k 步为例：

| 每步真正节约 | 200k 步节约 |
| ------ | -------- |
| 5 ms   | 16.7 分钟  |
| 10 ms  | 33.3 分钟  |
| 20 ms  | 1.11 小时  |
| 30 ms  | 1.67 小时  |

如果你们的新基模计划连续跑十几个小时，即使每步省 10–20 ms，也已经有明确价值。

但对一条只有 5k–10k 步的后训练，节省 10 ms/step 也不过一两分钟左右。所以我建议把复杂度高的性能优化放在反复使用、总样本量大的基模训练上，而强风格分支优先修复低成本的同步和编译问题。

### 我的最终排序

最值得立即让工程人员处理的四件事：

1. StrongStyle 全 GPU 化：尤其是每样本 `float(t[i])`、writer 槽位字符串查找，以及每步 CUDA Loss 格式化。
2. REPA FP16 缓存验收：当前 f32 可能只是 f16 上转换，值得核对并减少一半特征传输量。
3. C2OT + DataLoader profile：不要凭 64 核就扩大 worker/BLAS 线程，按真实等待时间决定改动。
4. 稳定 `torch.compile` 计算图：重点处理反事实子 batch、`model.eval()` 切换和动态条件，避免重新编译。

暂不值得投入的： 自写 FlashAttention/Triton、整个训练 step CUDA Graph 化、大规模 CPU EMA offload、盲目扩大 batch 或使用 64 个 DataLoader workers。

从源码和硬件规模看，你们现有 BF16 DiT 已经具备较好的计算基础。下一轮性能提升更可能来自减少 GPU 空转和冗余计算，而不是更换矩阵乘内核。

我的目标会设为：先验证基模端到端吞吐能否稳定提高 5–10%，同时让强 Critic/反事实后训练避免当前大量主机同步；达到这一目标后，再决定是否值得继续追求更高 MFU。