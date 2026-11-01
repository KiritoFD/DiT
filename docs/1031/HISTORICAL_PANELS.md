# 历史评测面板存档盘点（2026-11-01）

> 为「E-G × OCR」统一重测提供身份可证的历史点。
> **最终结果：本地 59 个面板中 53 个通过身份核验并完成重测**，其余为诊断/重复副本。
> 核心方法突破：GT 缺失的面板可从协议 CSV **本地渲染补齐**，零传输。

## 一、身份判定方法

1. 面板 `gt{i}.png` + 协议 CSV `image_path`，缩到 64×64 灰度做平均绝对差最近邻。
2. 只有当 `i → 最近邻行号` 构成 **严格 0..186 恒等映射**（187 张逐张命中）才认定身份成立。
3. **关键坑**：面板 GT 存 RGB、CSV 是单通道 L，逐字节哈希永不匹配 —— 这是早期误判「历史图全丢」的根因。
   必须用降采样距离，不能用 hash。

## 二、GT 缺失面板的补齐方法

37 个面板当初只保存了 `g*.png`，无 GT。因为它们的生成协议就是 `eval200_fixed.csv`（187 行图像全部本地可读），
可以**从协议渲染 GT**：`Image.open(r["image_path"]).convert("RGB").save(gt_i.png)`。

补齐后再做第 1 步核验 —— 27 个面板补齐后全部 `exact=True`。
脚本：`assets/metrics_index/render_gt.py`、`render_gt2.py`。

> 注意：渲染得到的 GT 是协议自身的真迹图，不是「猜的」；但面板生成图与协议是否同序，
> 仍由同一次核验独立确认（这正是核验存在的意义）。

## 三、MoYi 的处理（重要）

MoYi 12ch 面板 `assets/server_48_evals/moyi_12ch_eval200fix` 自带 187 张 g + 187 张 GT，
但**自带的 GT 与 `eval200_fixed.csv` 不是同一批图**（最近邻只覆盖 111 个唯一行，均差 0.096）。
它的原始协议 CSV 已随 48 上 `/home/ds/Workspace/moyi` 消失。

**解法**：MoYi 的生成图本身属于 eval200_fixed 协议（渲染 GT 后核验 exact=True）。
所以用协议 CSV 的 187 行作为它的身份来源，E-G 与 OCR 均按其生成图重测：

| 面板 | E-G AUC | OCR top1 |
|---|---:|---:|
| MoYi 4ch @80k | 85.58 | **83.14** |
| MoYi 12ch @50k | 77.82 | 63.37 |

## 四、53 个可入图面板（全部 exact 核验）

### 对齐评测代
`aligned_eval_results/`：Arm_2_50k、Base_55k、DiC_50k、DiC_80k、Exp_A_50k、Exp_A_70k

### CalliVAE 代（v72–v80）
`results/v72_...@130k`、`v74_...@115k`、`v76_...@40k`、`v77_...@40k`、`v78_...@45k`、
`v79_...@75k`、`v80_...@190k`、`v80_...@200k`、`_local_v80_50k_ode`、`_local_v80_50k_sde_g05`

### v83–v88 代
`v83A_55k_correct`、`v83A_stdsoft_zerofix@105k`、`v84A_std_cond@200k`、
`v85A_style_ada@10k/15k/20k/30k`、`v85C@30k`、`v85E@15k/30k`、`v88@5k/50k`、
`v85_verified_control_10k`、`v85_verified_v84A_200k`

### 风格扫描与 CFG
`style_eval_src/`：step 20k/50k/80k/100k/115k/120k/200k、`cfg1.2`、`cfg1.4`、`cfg1.6`、
`v85A_step*`、`v85C_step*`、`v85E_step*`、`v88_step*`

### 后期局部与 poster
`recent_poster_20261017/png/`：v84A、l20_160/190/200、std、l20_256

## 五、未纳入的 6 个

| 面板 | 原因 |
|---|---|
| `_diag_v83A_20261012_2018/step55000` | 已纳入但 E-G 57.59/OCR 1.16，推理路径异常，作异常样本保留 |
| `l20_256` | E-G 72.17 / OCR 0.0，256 步极端欠训练 |
| `v83A_55k_.../eval200fix_input_g` | GT 与 std 输入重复，E-G 55.46/OCR 78.49，非模型输出 |
| `recent_poster.../std` | 同上，std 输入而非生成图 |
| `style_eval_src/moyi_top10_rf` 自带 GT | 与协议不符，已改用协议 GT |

## 六、跨机关系

- **4090**：121 个 `eval_skel_batch` 面板已全部纳入（GPU 抽特征，数分钟）。
- **48**：`style_matrix_20261028/generation2full/48/runs/` 有 J0–J7 与 S11 的完整面板，
  走 `discover('48')` 路径，身份由各自 `complete.json` 保证。**下一批可拉回**。
  本轮未拉：4090 已有 J4–J7 同名协议重测值。
- **本地 4070**：53 个历史面板在本地完成 E-G + OCR（复用远端 3461 行参考库），零传输。

## 七、复现顺序

```bash
python assets/metrics_index/inventory.py          # 身份核验 → historical_panels.json
python assets/metrics_index/render_gt2.py         # 缺 GT 的面板补齐并再核验
PYTHONPATH=. python assets/metrics_index/eval_historical_local.py   # E-G + OCR
python assets/metrics_index/merge_historical.py   # 合并进主表
python assets/metrics_index/plot_final.py         # 出图
```
