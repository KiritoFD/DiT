# S4/S5/J2/J3 poster — what is and is not comparable on it

**图：** `assets/metrics_index/poster_s4s5j2j3.png`（1534×3271 raw，0.62× 交付版）
**脚本：** `tools/d_20261101/poster_s4s5j2j3.py`（可复现，`missing cells: 0`）
**日期：** 2026-11-02

## 列构成（10 列）

| # | 列 | 臂 | 面板 |
|---|---|---|---|
| 1 | GT 真迹 | — | val100 |
| 2 | GAR@100 | 冻结 K + 训练解码器 | val100 |
| 3 | GAR@336 | 同上 | val100 |
| 4 | GAR@672 | 同上 | val100 |
| 5 | K+D1dec | 换解码器 | val100 |
| 6 | **S4** | LocalCNN→Spatial AdaGN | val100 |
| 7 | **S5** | LocalAttention→XAttn | val100 |
| 8 | **K ★** | abc A@43429（基线）| val100 |
| 9 | **J2** | 解冻解码器 + 图像损失 + GT 重建 | 仅指标 |
| 10 | **J3** | 解冻解码器 + 图像损失，无 GT 重建 | 仅指标 |

## 必须说清楚的一件事：S4/S5 可对比，J2/J3 不可

行轴是 21 个共有 (书家, 字) 单元，**前提是每列的 GT 是同一张图**。这个前提只对 S4/S5 成立：

| | 覆盖面 | GT 是否同源 | 可否与 K 逐行相减 |
|---|---|---|---|
| **S4 / S5** | 21/21 | **是——逐位相同** | **可以** |
| **J2 / J3** | 21/21 | **否——0/21 同源** | **不可** |

**S4/S5 的校验（不是假设）：**
- 索引语义一致：S4/S5 的 `idx` 与 val100 单元 21/21 匹配，字符也 21/21 吻合。
- GT 逐像素相同：对多个索引做哈希与 `mean|diff|` 比对，**全部 0.0000**。
- 列身份正确：重算 SSIM 得 S4 0.6037 / S5 0.6079，与公布值 0.5988 / 0.5975 一致；
  且 **S4 与 S5 的 21 张图 0/21 相同**，证明不是同一列被贴了两次。

**J2/J3 的处境：** 属 heldout187 面板，21 个 (书家,字) 全都能对上，
但**每一个用的都是不同的 GT 实例**（0/21 与 val100 同源）。
把它贴在 K 旁边逐行比较会得出错误结论，所以图上用竖线把它们与 val100 组分开。

## ⚠ J2/J3 没有生成图，格子里是指标

J2/J3 的**生成图未归档、checkpoint 也不存在**（本机与 4090 均已确认）：
`docs/1028/raw/{J2,J3}/` 只有 `batch.csv`/`geometry.json`/`result.json`/`writer.json`。

**处理方式：** 不静默删列、更不拿别人的图冒充。这两列渲染成**指标条**——
格内印该行自己的 `SSIM / LPIPS / ink-IoU / frag`，并标注「无生成图」。
21 个单元**全部**有指标（`cells missing metrics = 0`）。

## 图上有一处刻意的设计

val100 组与 J2/J3 组之间有一条**橙色分隔线**，列头分别标绿 `val100` / 橙 `仅指标`。
这样看图的人不会在两组之间横向比。**如果只想看可直接对比的部分，看分隔线左侧的 8 列。**

## 数据源

- S4/S5 生成图：`scratch/panels/poster_imgs/{S4_50k,S5_50k}/g{idx}.png`（val100 idx）
- K 及解码器臂：`assets/d_20261101/reports_assets/gar_crossdecode/*/eval_samples_ctrl/*/eval200fix/`
- GT：同上 `gt{idx}.png`
- J2/J3 指标：`docs/1028/raw/{J2,J3}/batch.csv`
- 行轴定义：`scratch/panels/cells.json`

## 复现

```bash
python tools/d_20261101/poster_s4s5j2j3.py
# 输出 assets/metrics_index/poster_s4s5j2j3{,_raw}.png
```

注：S4/S5 的生成图在 `scratch/panels/poster_imgs/`（该目录被 `.gitignore` 忽略），
故**仅凭仓库无法重跑这两列** —— 需要先恢复那批图。其余列（GT/K/GAR/K+D1dec）
与 J2/J3 指标全部在版本控制内，可独立重建。
