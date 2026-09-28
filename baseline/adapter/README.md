# Baselines on top10_style23 — 三个对比方法的复现与启动

三个对比方法在 **top10_style23（38,583 张 / 30,304 个 (槽位,字) 对 / 23 槽位 / 256×256）**
上从零训练，与 v24 完全同口径。目录对应关系（服务器 `/root/Workspace/xy/DiT/baseline/`）：

| 目录 | 方法 | 说明 |
|---|---|---|
| `FontDiffuser/` | FontDiffuser (AAAI 2024) | **已改造为纯 latent 版**（官方是 pixel DDPM，256px 下注意力矩阵 8.6GB/层不可训练；latent 版 UNet/编码器全在 4×32×32 项目 latent 空间，用我们的 sd-vae-ft-ema + shards 基础设施） |
| `VQ-Font/` | VQ-Font (ICCV 2023) | GAN + 矢量量化部件风格，256×256，码本需自训 |
| `DG-Font/` | DG-Font (CVPR 2021) | GAN + 可变形卷积（DCN→torchvision 等价实现），256×256 |

> CF-Font（CVPR 2023）未纳入：依赖老式 DCN CUDA 编译且三阶段流程，其「内容融合」
> 对比叙事由 VQ-Font 承担。

## 0. 环境（一次性，已完成）

```bash
# conda env `baseline`（clone 自 cu121, torch 2.5.1+cu121）— 已建好
bash /root/Workspace/xy/DiT/baseline/adapter/env_setup.sh        # 幂等
bash /root/Workspace/xy/DiT/baseline/adapter/run_prep.sh         # 数据准备(幂等)
```

数据准备产出：`data/content_font/deng{,_num,_shards}/`（Deng 内容字渲染 + latent
shards，走项目 `src.data.vae_io` 基础设施）、`data/fontdiffuser/train/`、
`data/vqfont/{lmdb,meta}/`（36,154 条，灰度归一）、`data/dgfont/train/`（24 个
`id_XX` 域 + `domain_map.json`）、`data/eval/{strict84,seen20}/gt + refs.json`
（确定性 3 参考字协议，排除目标字）。并打补丁：
- DG-Font：att_to_use 400 类 bug、torchvision DCN、验证门控（每 25 epoch，
  `DG_SKIP_VAL=1` 可关）、`val_batch=min(val_batch,val_num)`、`DG_COMPILE` 钩子；
- FontDiffuser：数据集 .png、StyleRSI 配对、SDPA 注意力。
torchvision 缓存已预置 vgg16 / alexnet / pt_inception 权重。

## 1. 显存实测与推荐 batch（4090 24G，实测值，保证不 OOM）

| 方法 | 配置 | 峰值显存 | 状态 |
|---|---|---|---|
| FontDiffuser-Latent | **B192**（默认） | **16.2G** | ✅ 推荐，1/3 余量 |
| FontDiffuser-Latent | B256 | 21.5G | ✅ 实测上限，长训贴边 |
| FontDiffuser-Latent | B128 | 11.5G | ✅ 最稳 |
| FontDiffuser-Pixel（弃用） | B4 | >24G | ❌ OOM（改造动机） |
| VQ-Font | **B8**（默认） | **13.8G** | ✅ 推荐 |
| VQ-Font | B12 | 21.5G | ⚠️ 可跑但贴边 |
| VQ-Font | B16 | 23.9G | ❌ 长训必 OOM |
| DG-Font | **B8**（默认） | **16.6G** | ✅ 推荐 |
| DG-Font | B16 | >24G | ❌ OOM |

> 按项目要求统一 **不使用 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments`**。

## 2. 训练启动命令

```bash
# ---------- FontDiffuser-Latent（150k 步, B192, ~16G）----------
cd /root/Workspace/xy/DiT/baseline/FontDiffuser
STEPS=150000 BS=192 COMPILE=yes bash ../adapter/fontdiffuser/train_latent.sh
# ckpt: outputs/latent_top10/global_step_N/{unet,style_encoder,content_encoder}.pth

# ---------- VQ-Font（三步：VQ-VAE 码本 → 相似度矩阵 → GAN 训练）----------
cd /root/Workspace/xy/DiT/baseline/VQ-Font
python ../adapter/vqfont/pretrain_vqvae.py                    # 50k iter
python ../adapter/vqfont/build_similarity.py                  # npy 相似度矩阵
python ../adapter/vqfont/train_top10.py top10 ../adapter/vqfont/cfg_top10.yaml
# 加速可选: VQ_COMPILE=1 前缀 (ckpt 带 _orig_mod.* 键, 采样脚本自动剥掉)
# ckpt: work_dir/checkpoints/top10/{step}-top10.pth

# ---------- DG-Font（100k 步 = 100 epoch × 1000 iter, B8, ~16G）----------
cd /root/Workspace/xy/DiT/baseline/DG-Font
bash ../adapter/dgfont/train_top10.sh
# ckpt: logs/GAN_top10_256/model_{epoch}.ckpt
# 加速可选: DG_COMPILE=1 bash ../adapter/dgfont/train_top10.sh
```

三者共用一张卡时**串行**跑（tmux），不要并行。

## 3. 推理 + 评测

```bash
cd /root/Workspace/xy/DiT/baseline

# FontDiffuser-Latent
cd FontDiffuser && python ../adapter/fontdiffuser/sample_latent.py \
  --ckpt_dir outputs/latent_top10/global_step_150000 --eval_tag strict84 \
  --save_image_dir ../results/fontdiffuser_latent/strict84

# VQ-Font
cd VQ-Font && python ../adapter/vqfont/sample_top10.py ../adapter/vqfont/cfg_top10.yaml \
  --weight work_dir/checkpoints/top10/150000-top10.pth --eval_tag strict84 \
  --save_dir ../results/vqfont/strict84

# DG-Font
cd DG-Font && python ../adapter/dgfont/sample_top10.py \
  --log_dir logs/GAN_top10_256 --eval_tag strict84 --save_dir ../results/dgfont/strict84

# 统一指标（MSE / SSIM / LPIPS，按 {槽位}__{字}.png 对齐 GT）
cd /root/Workspace/xy/DiT/baseline
for m in fontdiffuser_latent vqfont dgfont; do
  python adapter/common/eval_metrics.py --gt data/eval/strict84/gt \
    --pred results/$m/strict84 --out results/$m/strict84/metrics.csv
done
# seen20 同理: --gt data/eval/seen20/gt
```

## 4. 冒烟测试（改代码后重跑）

```bash
bash /root/Workspace/xy/DiT/baseline/adapter/run_smoke.sh all         # 训练冒烟
bash /root/Workspace/xy/DiT/baseline/adapter/run_infer_probe.sh all   # 采样+eval 贯通
```

已全部通过：FD-latent 训练 6 步（B8/B12/B16/B20/B64/B96/B128/B192/B256）+ 采样
探针；VQ-Font VQ-VAE/相似度/训练 6 iter；DG-Font 训练 25 iter；三采样探针
4 张图 + eval 出数。

## 5. 评测口径（与 v24 完全一致）

- **训练**：top10 全量 38,583 行。基线数据模型是"每域每字一张"，取每对第一张
  （img_id 序）→ 30,304 对；FD-latent 的 target/style 直接读 `shards_img`，
  content 读 `deng_shards`。仅 5 个 Deng 缺字（CJK 扩展区，均不在评测集）涉及
  10 行被跳过。
- **评测 A（strict84）**：84 张 `data/50k` 未见样本图；**评测 B（seen20）**：
  20 张 top10 训练图。GT 在 `data/eval/*/gt/`。
- **参考协议**：`data/eval/refs.json`，每 (槽位,字) 确定性抽 3 个同槽位参考字
  （md5 种子，排除目标字，同 LF-Font FixedRef 语义）。FD-latent 单参考（one-shot
  设计，用第 1 个），VQ-Font 3 参考（kshot=3），DG-Font 风格域码（用第 1 个）。
- **内容字体**：Deng.ttf 渲染 256×256，latent 走与 shards_img 同款 sd-vae-ft-ema
  （SCALING=0.18215）。
- **CFG 丢弃**：FD-latent 用"空白字 latent"（白图过 VAE 编码的常量）替代 pixel
  版的 torch.ones；VQ/DG 按各自机制。

## 6. 方法改造点汇总（除以下各项均保持官方实现）

- **FontDiffuser → latent**：UNet in/out 4ch@32²；style/content 编码器改 4ch@32²
  输入（arch 表加 32 档 `[ch,2ch,4ch,8ch]`@`[16,8,4,2]`，尾通道 ch×8 与
  cross_attention_dim 对齐，MCA 下行 index 配对随之自洽）；DDPM betas 不变；
  pixel 感知损失(0.01) → x0-latent MSE(0.01)；SDPA 注意力（数值等价，去掉
  score 矩阵；cross-attention 的 k/v seq 单独处理）；CFG 空白 latent；`--compile`
  可选。phase-2 不训（官方 SCR 预训练未放出）。
- **VQ-Font**：官方 `weight/*.pth` 全为 0 字节 → VQ-VAE 按 notebook 等价脚本自训
  （结构逐层对齐保证 `load_pretrain_vae_model` 键位吻合）；相似度字典 → npy +
  `SimMatrix` 接口（官方 JSON 缺失且 5,855² 过大）；LMDB 写入统一灰度（top10 约
  37% 原图是 RGB）；**batch_size≥6 硬约束**（对比损失采 5 负样本）；VQ-VAE 与
  FFG 统一 Normalize(0.5,0.5)（官方两阶段不一致，取一致口径）。
- **DG-Font**：DCN → torchvision 等价实现（免 CUDA 编译）；域目录 `id_XX` +
  `domain_map.json`（其按 `int(name[3:])` 排序）；`att_to_use` 400 硬编码 bug 修复；
  验证每 25 epoch（原版每 epoch 跑 24×24 参考组合 FID，代价不可接受）；
  `val_batch=min(val_batch,val_num)`（原版 val_num<val_batch 时 AdaIN 参数错配
  直接崩）。
