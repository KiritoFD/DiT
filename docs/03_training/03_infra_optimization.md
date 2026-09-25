# 基础设施极致优化：4.09 Steps/s 吞吐与硬件极限压榨

## 1. 硬件平台与吞吐总览

Callig-DiT 在单张消费级旗舰 **NVIDIA GeForce RTX 4090 (24GB VRAM)** 上实现了前所未有的训练效率：

| 性能指标 | 优化前基线 | 马良优化后 | 提升幅度 |
| :--- | :--- | :--- | :--- |
| **训练步频** | $1.28 \text{ steps/s}$ | **$4.09 \text{ steps/s}$** | **$3.2\times$ 吞吐加速** |
| **单秒生成吞吐** | $409 \text{ glyphs/s}$ | **$1,308 \text{ glyphs/s}$** | **超千字/秒级训练** |
| **稳态显存占用** | 23.4 GB (频发 OOM) | **18.59 GB / 24 GB** | 留存 5.4GB 安全余量 |
| **GPU 功耗** | 230W - 270W 抖动 | **368W 饱和满载 (82% TDP)** | 释放 GPU 算力极限 |
| **200k 训练周期** | 43.4 小时 | **13.6 小时** | 隔夜即可完成整轮收敛 |

---

## 2. 五大底层优化支撑架构

```
                 [RTX 4090 饱和运行架构]
                            │
   ┌────────────────────────┼────────────────────────┐
   ▼                        ▼                        ▼
[内存全预载 (RAM Preload)] [PyTorch Inductor 图编译] [SDPA 内核加速]
 50k 潜变量常驻内存 Host    全算子融合成单个 Triton 核 规避 Attention 显存展开
   │                        │                        │
   └────────────────────────┼────────────────────────┘
                            ▼
           [异步 CPU 权重落盘 (Async Checkpoint)]
            主训练循环零等待，后台线程并发写盘
```

### 2.1 PyTorch 2.0+ TorchDynamo & Inductor 图编译
- 配置开启 `"compile": true, "compile_mode": "default"`；
- **算子融合（Kernel Fusion）**：将 Transformer 中的 RMSNorm、SwiGLU、残差相加融合为一个单一的 Triton 算子，彻底消除了显存中间变量的频繁读写（Memory Bound $\to$ Compute Bound）；
- 编译开销仅发生在训练前 3 步，随后步频稳定锁定在 $4.09\text{ steps/s}$。

### 2.2 SDPA (Scaled Dot-Product Attention) 与零显存展开
- 设置 `"attn_impl": "sdpa"`；
- 调用 cuDNN / FlashAttention-2 原生 C++ 内核，在计算注意力权重时不实例化 $B \times H \times N \times N$ 的中间注意力图；
- 显存复杂度从 $\mathcal{O}(N^2)$ 骤降至 $\mathcal{O}(N)$，为全局 Batch Size 扩展至 320 创造了关键空间。

### 2.3 零 I/O 内存预加载流水线 (In-Memory Preloading)
- 设置 `"preload": true, "preload_workers": 16`；
- 启动时将 50,000 个图像与骨架潜变量完全映射并缓存在系统物理内存中（占用 $\approx 4.8\text{ GB}$ 主机内存）；
- 训练循环中的每个 Batch 读取只经过纯内存 DMA 搬运，耗时 $< 0.1\text{ ms}$，彻底斩断了 NVMe 磁盘小文件随机读带来的 I/O 停顿。

### 2.4 异步 CPU 检查点落盘 (Async Checkpoint Saving)
传统保存检查点时，`torch.save(model.state_dict())` 涉及显存向 CPU 拷贝及数秒的 NVMe 写盘，造成训练循环每隔 5000 步硬停顿 15-20 秒。
马良在 `src/train/train.py` 中实现了**非阻塞异步检查点保存**：
1. 主线程将 EMA 权重通过非阻塞流浅拷贝至 CPU 内存（耗时仅需 $8\text{ ms}$）；
2. 随后立即恢复 GPU 训练前向传播；
3. 后台守护线程（Daemon Thread）以异步方式将权重写入磁盘并记录评估 CSV，主训练循环实现**绝对零等待**。

---

## 3. 稳态显存预算分解表 (Batch=320, VRAM: 18.59GB)

| 显存占用分类 | 占用量 (GB) | 占比 | 优化策略 |
| :--- | :--- | :--- | :--- |
| **模型静态参数与 EMA** | $\approx 0.85\text{ GB}$ | $4.6\%$ | FP16 精度存储，梯度与优化器状态紧凑布局 |
| **AdamW 优化器状态 (FP32)** | $\approx 1.70\text{ GB}$ | $9.1\%$ | 一阶动量矩 + 二阶方差矩 |
| **前向激活值 (Activations)** | $\approx 11.20\text{ GB}$ | $60.2\%$ | 依靠 Inductor 自动重计算与算子融合压缩 |
| **SkelNet 变形头与特征图** | $\approx 1.84\text{ GB}$ | $9.9\%$ | $32 \times 32$ 紧凑网格采样与硬门控遮罩 |
| **PyTorch 缓存区与余量** | $\approx 3.00\text{ GB}$ | $16.2\%$ | 留存用于内存内评估（`in_mem_eval`）与采样 |
| **总计稳态显存** | **$18.59\text{ GB}$** | **$100\%$** | **完美适配 24GB 消费级显卡** |
