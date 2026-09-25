# 字泛化管线：标准字形 latent 字典 v2（2026-08-29）

> 目标：为 zero/few-shot 字符泛化提供实例级结构条件。标准字库渲染 256×256 →
> sd-vae-ft-ema encode → (4,32,32) latent 字典，与主模型 latent 同空间。

## 库

`src/utils/std_glyph_latent_v2/`（本地构建，已同步 4090）：

| font key | 字体 | 书体 | 字数 |
|---|---|---|---|
| kai_gb | 楷体 GB2312 (simkai) | 楷(0) | 8,118 |
| kai_st | 华文楷体 | 楷(0) | 7,842 |
| wei_st | 华文新魏 | 楷(0) | 6,651 |
| xing_st | 华文行楷 | 行(3) | 6,651 |
| li_gb | 隶变 (SIMLI) | 隶(4) | 7,842 |
| li_st | 华文隶书 | 隶(4) | 6,651 |

合计 **43,755** latents（float16 4×32×32，manifest.json 索引）。

## 字符集与覆盖

- 字符集 = 通用规范汉字表 8105 ∪ mid_clean 三书体字表 → **8,118 字**
- mid_clean 训练字（楷 2102 / 行 2031 / 隶 1328）：**100% 覆盖**
- 8105 规范字：**100% 覆盖**
- eval_strict_top6 263 字：覆盖 170（64.6%）。缺的 93 字全部是**繁体/异体**
  （聽軍録詩補…），6 个简体字库均不含——与 s19 的 93 个 zero-shot eval 字
  **完全重合**。
- 渲染归一：白底黑字，墨迹 bbox 等比缩放到 0.88×256 正方形居中
  （对齐数据集构图：ink h≈0.81 / w≈0.93 / 居中）。

## 代码

- 构建：`tools/build_std_glyph_latents.py`（渲染 + 本地 GPU encode，断点续跑；
  8GB 卡用 `--encode-batch 8 --mem-cap-gb 4.5`，总显存 ~4.2G）
- 查询：`src/utils/glyph_latent_v2.py` → `get_glyph_lookup_v2()`
  ```python
  lk.get(script_id, char)                  # script 默认字体
  lk.get(script_id, char, font="wei_st")   # 指定字体
  lk.get(script_id, char, random=True)     # 该书体可用字体随机 (训练增广)
  ```
  已验证 decode 往返无损（永/中 字形完整）。

## 待办 / 下一步

1. **训练接入**：dataset `g` 条件改走 v2 字典（当前 use_glyph_cond 走 v1），
   训练时 `random=True` 字体随机 = 免费增广；eval 时喂 g（当前 2cond eval 不喂）。
2. **繁体缺口**：下载开源繁体字库（Noto Serif TC / 霞鹜文楷 TC）重跑构建脚本
   （字符集不用变），可把 eval 覆盖从 170 → 263，正好补齐 93 个 zero-shot 字。
3. 质量对照：字体渲染的墨迹密度 0.15~0.24 vs GT 0.227，同一量级；
   如需更贴近书法笔画粗细变化，可加骨架化/笔画宽度扰动增广。
