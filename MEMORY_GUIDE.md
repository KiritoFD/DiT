# DiT 训练显存预测与参数设置指南

> 基于 2026-08-14 在 RTX 4090（24G）上的实测数据（`_mem_diag.py` / `_probe_eval_mem.py` / 训练日志 Mem 字段）。
> 目标：以后开启新实验时，能**在跑之前估出显存用量**，直接定对 batch / eval 参数，不再踩 OOM。

## 一、显存构成（从模型到 eval 的 6 项）

显存总占用 = 模型权重 + 优化器 + 梯度 + 前向激活 + 反向激活 + （结构 loss decode）+ （eval cache/推理）。

| # | 组件 | 占用量 | 说明 |
|---|---|---|---|
| 1 | 模型权重 | `N × 4B` | fp32 常驻；N=参数量 |
| 2 | AdamW 优化器 | `N_train × 8B` | 每可训练参数 2 个一阶/二阶矩×4B |
| 3 | 梯度 | `N_train × 4B` | 仅可训练参数 |
| 4 | 前向+反向激活 | ∝ `batch × depth × hidden² × tokens` | 主力，与 batch 线性（DiT 无 attention 爆炸时） |
| 5 | **结构 loss VAE decode** | ∝ batch，且 attention 层超线性 | **历史上 90% 显存的来源** |
| 6 | **eval cache + 推理** | encode ∝ encode-batch；decode ∝ decode-batch | 训练中每 ckpt 触发一次 |

示例：DiT-S（36.9M）全参 batch=8：模型 0.15G + opt 0.15G + 激活 1.3G ≈ **1.3G**；但一旦加结构 loss decode（batch=8）立即跳到 **15.6G**。

## 二、实测数据点（RTX 4090 / 24G）

| 场景 | 显存 peak | 备注 |
|---|---|---|
| DiT-S(36.9M) 全参 + AdamW | 0.15G | 权重+opt |
| DiT-S 前向+反向 batch=8（纯 diff） | 1.29G | 激活 |
| 结构 loss VAE decode batch=8（grad_ckpt 重放） | **15.56G** | 显存黑洞 |
| `prepare_eval_cache` encode 1000 图 batch=64 | **16.65G** | OOM 元凶 |
| `prepare_eval_cache` encode 1000 图 batch=16 | ~4.5G | 修复后 |
| `eval_in_memory` decode batch=32 | **10.80G** | |
| `eval_in_memory` decode batch=8 | 4.27G | 修复后 |
| S-scratch 训练 batch=64（关结构 loss）+ eval | **6.14G** | 修复后稳态 |
| 2Cond/3Cond-XL batch=8 纯 diff（TRAINING_NOTES） | ~12G | 大模型激活 |
| 3Cond-XL batch=8 含结构 loss | 18.5G | batch=4 时更安全 |

## 三、预测公式（开实验前先估）

**总预算 < 24G 即安全（留 ~2G 余量）。**

```
总显存 ≈ 权重(4N/1e9)G + 优化器(8N_train/1e9)G
         + 激活(随 batch/depth 线性)G
         [+ 结构 loss decode(15.6 × batch/8)G]     # 开了 use_canny/use_skel 才加
         [+ eval encode 峰值(batch_encode/16 × 4.5)G]  # 每 ckpt 触发
         [+ eval decode 峰值(batch_decode/8 × 4.27)G]
```

**三条铁律**（实测总结）：

1. **结构 loss 是显存天花板**：`use_canny/use_skel=true` 时，decode batch=8 就 15.6G，XL 巨模型下 batch 只能 ≤4。
   要吞吐/大 batch，先关结构 loss（纯 diff 显存骤降、吞吐可放大一个数量级）。
2. **eval 的 encode 才是隐性 OOM 源**：`prepare_eval_cache` 的 VAE encode batch 必须 ≤16（batch=64 会到 16.65G）。
   同理 eval decode ≤8。这是默认参数，**不要再调大**。
3. **大模型主要吃激活**：DiT-XL(707M) batch=8 纯 diff ~12G；S(37M) 同等 batch ~1.3G。扩 batch 时激活线性涨，
   显存与模型尺寸正相关，先估激活再定 batch。

## 四、参数设置清单（开新实验照抄）

| 参数 | 建议值 | 依据 |
|---|---|---|
| 结构 loss（use_canny/use_skel） | 默认 **false** | 关掉才有吞吐与大 batch；要开则 batch ≤4（XL）/≤8（S） |
| `global_batch_size` | XL=8、S=64（关结构 loss） | 显存实测；可再探 |
| `eval_auto.py: prepare_eval_cache batch_size` | **16**（勿 >16） | encode 峰值 4.5G |
| `eval_auto.py: eval_in_memory batch_size` | **8**（勿 >16） | decode 峰值 4.27G |
| `use_checkpoint` | false | 历史 NaN 根因（LoRA+随机 dropout） |
| `preload` | true | 内存 251G 充足，消除 IO |
| `ckpt schedule` | 前 5000 每 1000，后每 4000 | 前期密、后期省盘 |
| 显存日志 | 每 log_every 打印 `Mem: cur/peak` | 全程监控

## 五、监控方式

- **训练日志**：每 20 步打印 `Mem: <allocated>G/<max_allocated>G`（当前 + 峰值），`pull_log.py` 可解析。
- **nvidia-smi**：`--query-gpu=memory.used,utilization.gpu --format=csv,noheader` 看 reserved + 利用率。
- **OOM 定位脚本**：`_mem_diag.py`（分解模型/opt/VAE/前反向/decode）、`_probe_eval_mem.py`（eval 峰值）。
