# 推理/评测的两个条件分支（写清楚：table 就 table，std 就 std）

> 起因：2026-10-12 的"白化"误判。真相是**评测时把 std 的字形条件拔掉了**（`g=None`），
> 而模型是用条件训出来的 —— 条件一缺，输出就大面积空白（33/187 近白）。
> 分数难看得像训练坏了，其实是评测喂错了输入。**本文把两条分支的契约钉死。**
>
> 同一 55k ckpt、同一协议，只把条件接回去：
> `ink IoU 0.0937 → 0.2768`、近白 `33/187 → 1/187`、`LPIPS 0.4071 → 0.3556`、**SSIM 几乎不变**
> （再次证明 SSIM 不能用来判"有没有墨"）。

---

## 一、两条分支是什么

| | **分支 T：table（字表）** | **分支 S：std（标准字软掩码）** |
|---|---|---|
| 代表 run | v68 家族（`v68_aug_sp_c2ot` 等）| v81 / **v83-A / v83-B** |
| 条件来源 | 三张 embedding 表：书家 / 汉字(4690) / 书体 | 三张表 **＋** 标准字渲染的 `(N,1,64,64)` 软掩码 |
| 判定键 | `std_soft_cond == False`（无 `glyph_embedder`）| **`std_soft_cond == True`**（有 `glyph_embedder`）|
| 注入方式 | 表条件走 `factorized_cat`，`cond_inject_at=2,4,5,6` | 同上 **＋** `enc(g)` 走 glyph 分支 → `adaln` |
| **`g` 的契约** | **必须是 `None`**（有 `g` 才是 bug ✗）| **必须是非空 `(N,1,64,64)`**；`None` = 条件被拔掉 ✗✗ |
| 拔掉条件的后果 | **无影响** ✓ —— 这就是它训练时的样子 | **大面积空白** ✗（实测 33/187 近白）|
| 训练时的处理 | 无此概念 | `glyph_drop_prob`（A=**0.0** ✗ / B=**0.1** ✓）；A 从没见过"无条件"状态 |
| 评测工具 | `tools/eval_table_checkpoint.py` | `tools/eval_std_soft_checkpoint.py` |
| 统一入口 | `tools/eval_checkpoint.py --ckpt <ckpt>` ← **自动判分支并说明理由** ✓ | 同左 |

**同一个模型里两条分支是并存的**（v83 的 std run 也照常用三张表 ✓）；这里的"分支"指的是
**评测/推理时该喂什么条件**，以及**喂错了要立刻报错**。

---

## 二、分支 S 的硬契约（评测器必须自证）

1. **`g` 必须真的到达编码器** ✓
   `eval_std_soft_checkpoint.py` 用 `glyph_embedder.register_forward_pre_hook` 记录前两次输入，
   断言 `mean > 0` ✓ —— **这是模板做法**（否则"声明 std、实际 `g=None`"会静默通过 ✗）。
2. **`g` 的形状/取值域** ✓ —— `(N,1,64,64)`、非全零、σ 网格内（`std_soft_eval_sigma`）。
3. **LPIPS 必须可用** ✓（否则该次结果不予采信，而不是留空 ✗）。
4. **协议必须显式记录** ✓ —— γ、steps、cfg、sampler、VAE、种子、源文件 sha256 全写进 `manifest.json` ✓。
   > 血的教训：v83-A 训练注释写 ODE(γ=0)，实际日志与配置是 **SDE γ=0.5**（`--sde-gamma` 的 CLI 默认 0.5 ✗），
   > 评测也沿用了它 ⇒ 与 ODE 评测**不可比** ✓（现已在 config 里显式 `sde_gamma: 0.0` ✓）。

## 三、分支 T 的硬契约

1. **`g` 必须是 `None`** ✓ —— 构造 cache 时 `g_kind` 若不是空/None，说明喂错了 ✗（表条件 run 不该有字形输入）。
2. **不许有 `glyph_embedder`** ✓ —— 有就说明它是 std run，应改用分支 S ✓。
3. 其余协议与 S **逐项一致** ✓（heun / steps 50 / cfg 1.0 / EMA / 同 eval sets）—— 否则两个家族不可比 ✗。

## 四、怎么跑（一条命令，自动判分支）

```bash
PY=/opt/conda/envs/cu121/bin/python          # ★ 4090 上只能用它（cu117/2.7/3.6 都不行）
$PY tools/eval_checkpoint.py --ckpt <ckpt.pt> --out <新的空目录> \
    --vae-path data/pretrained/pretrained_models/sd-vae-ft-ema
```
它会：打印**判定的分支与依据** → 校验该分支的硬契约 → 调对应的评测器 → 产物落 `<out>/{manifest,metrics}.json`。

**禁止**：把两种分支的结果混进同一个目录 ✗；用不同协议的数字互比 ✗；用 SSIM 判"有没有墨" ✗。

## 五、和"白化"的关系（结论修正）

- 先前把"输出空白"归因于**编码器零输入响应膨胀**（H2）✗ —— 该归因**证据不足**：
  修好评测后，同一个 55k ckpt 是健康的 ✓；而背景/内容比在 **v83-A 里同样一路涨**（0.077→0.303 @20k→55k）
  却并不影响生成质量 ⇒ **该比值不能预测失效** ✗。
- 现在最强的解释是：**条件缺失**（评测 bug 触发）+ **`glyph_drop_prob=0` 导致模型没学过无条件** ✗。
  ⇒ 所以 **v83-B（drop=0.1）** 才是"条件鲁棒性"的对照 ✓。
- 任何未来的 std 评测，**必须**先过第二节的硬契约 ✓。
