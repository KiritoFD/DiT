# Latent vs Pixel 结构监督的 CPU 定量实验

## 问题
latent 空间(4×32×32)的 canny/skel 监督，能否达到和 pixel 空间(3×256×256)一样好的结构信号？

## 方法（远程 CPU, 128 核, 约 1-2 分钟）
- 数据：train_top30 取 40 个常见字(每字前 6 书家) = 240 样本
- 对每个样本：
  - pixel_canny  : 直接展平 GT canny 二值图(64×64) 作向量
  - latent_canny : GT canny 图用 VAE(sd-vae-ft-ema) 编码成 latent(4,32,32)×0.18215
  - latent_skel  : GT skeleton 同理
- 指标：同字对(L2)均距 D_same vs 异字对均距 D_diff，分开度 = D_diff/D_same
  （≈1.0 = 无结构判别力；越大越能把同字聚拢、异字分开）

## 结果
| 空间      | 同字均距 | 异字均距 | 分开度(diff/same) |
|-----------|---------|---------|-------------------|
| pixel_canny | 5164    | 5373    | 1.041             |
| latent_canny| 60.7    | 63.6    | 1.048             |
| latent_skel | 49.6    | 51.2    | 1.032             |
| Gaussian 噪声 | -      | -       | ≈1.0（参照）      |

## 结论
1. **latent 对 canny/skel 的结构信号保留 ≈ pixel**（分开度 1.048 vs 1.041），
   **latent 不劣于 pixel** → 可用 latent 版 canny/skel 监督替代 pixel 版，省去每步
   像素 VAE decode 的大开销（这是 train.py `w_latent_canny`/`w_latent_skel` 的动机）。
2. 但 canny/skel 的"字级判别力"本身很弱（分开度仅 1.03~1.05 vs 噪声 1.0）：
   结构监督主要引导"笔画/骨架清晰"，不编码字形 identity。要真正学字形，需更强条件
   (字符 id / 标准字形 / 更大模型)，结构 loss 仅是辅助。

## 复现
`/tmp/latent_canny_skel_probe.py`（远程已跑），结果 `/tmp/latent_canny_skel_probe.json`
