# 存储布局与显存/内存预加载流水线

## 1. 离线潜变量分片存储 (Latent Shards Layout)

实时通过 VAE 编码 $256 \times 256$ 图像极其耗费显存与计算力（单张图片 VAE 耗时约 4ms，占满 4090 计算能力的 30% 以上）。
因此，马良全面采用**全量离线编码分片存储（Pre-encoded Latent Sharding）**方案。数据物理分布在 `data/50k/` 目录下（共 515,185 个轻量分片，总体积 4.87GB）：

```
data/50k/
├── shards_img/          # 目标书法字真实图像潜变量 (z_img in R^{4x32x32}, float16)
│   ├── 00000.npy
│   └── ...
├── shards_std/          # 标准规范骨架潜变量 (z_skel in R^{4x32x32}, float16)
│   ├── 00000.npy
│   └── ...
└── shards_aux_skel3/    # 真实书法字形变骨架 (用于 SkelNet 中间监督 L_deform)
    ├── 00000.npy
    └── ...
```

---

## 2. 内存预加载机制 (In-Memory RAM Preloading)

在 batch size = 320、训练吞吐达到 $4.09\text{ steps/s}$ 的极限工况下，每秒需要从存储介质读取：
$$
320 \times 4.09 \approx 1,308 \text{ 组样本/秒}
$$
若通过 NVMe 磁盘文件系统进行随机小文件读取，文件系统元数据锁（dentry lock）会导致磁盘 I/O 成为主瓶颈，GPU 利用率会暴跌至 20% 以下。

### 2.1 预加载配置与实现
在训练配置 [`src/train/configs/v21_skelnet_200k.json`](file:///g:/GitHub/DiT/src/train/configs/v21_skelnet_200k.json) 中激活：
```json
{
  "preload": true,
  "preload_workers": 16,
  "num_workers": 8
}
```
1. **多线程并发载入**：在训练启动阶段，开辟 16 个并发 Worker，一次性将 50,000 个图像潜变量、标准骨架潜变量及形变骨架全部预加载至主机内存（Host RAM）；
2. **零拷贝内存对齐**：总内存常驻仅占用 $\approx 4.8\text{ GB}$ 内存，训练过程中数据读取直接通过内存指针索引完成（零磁盘 I/O 开销）；
3. **数据流水线加速**：PyTorch DataLoader `num_workers=8` 仅负责内存到 CUDA 显存的异步异步 DMA 传输，彻底释放 GPU 算力。

---

## 3. DINOv2 表征缓存 (DINO Feature Cache)

为了在训练中实时计算 REPA 表征自监督对齐损失（$\mathcal{L}_{repa}$），系统避免了在 GPU 上实时加载庞大的 DINOv2 ViT-B/14 骨干网络（否则将占用约 4GB 宝贵显存并消耗 30% 算力）：
- **离线预抽取**：将全量 50k 样本在 DINOv2 的第 8/12 层输出特征预先离线抽取并缓存；
- **存储路径**：`data/dino_cache/50k_v1`（包含 13 个汇总特征归档包，总计 98GB）；
- **动态按需映射**：训练进程通过内存映射（`mmap`）高效读取，以几乎为零的 GPU 开销实现顶尖的自监督表征语义对齐。
