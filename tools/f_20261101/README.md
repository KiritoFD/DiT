# F 计划：以 8×8 Writer 注入为核心的 12 小时后训练

依据：[p125 第二阶段机制诊断](../../docs/1031/p125_phase2_diag.md)。
一句话：书家条件**接通但弱**（direction 0.41 / amplitude 0.59），且身份**几乎全由 8×8 注入承载**
（关掉后 direction 0.1659），而 32×32 与 16×16 两个注入**是净负担**
（关掉后 FM loss −0.0154 / −0.0085，生成指标无损）。VAE 已排除。

因此本轮不碰架构、不换 VAE、不扩主干，只做：**摘掉有害注入 + 强化 8×8 书家通路**。

## 阶段与命令

| 阶段 | 内容 | 产物 |
|---|---|---|
| s0 | K 选择：A@43429 vs P3A@20000 同协议评测 | `k_choice.json` |
| s1 | K 上四种推理消融（原样 / 关0 / 关1 / 关0+1）+ 简化结构内容复核 | `eval/abl_*` |
| s2 | 由 K 生成 F0/F1 spec | `specs/F0.json`、`F1.json` |
| s3 | 冒烟：60 步 → resume 80 步 → 断言 gain 仍为 0（raw+EMA） | `smoke/` |
| s4 | F0：FM only，5k | `runs/F0` |
| s5 | F1：FM + 弱 E-G（ratio 2%），5k | `runs/F1` |
| s6 | 同协议评测 F0@5k / F1@5k → 选优 | `arm_choice.json` |
| s7 | 赢家续训 5k → 12k（同一 cosine horizon，resume 续调度） | `runs/<W>` |
| s8 | 最终验收 + 推理扫描（CFG 1.0/1.3/1.6 × site2 gain 1.0/1.2） | `eval/<W>_12000` |

```bash
# 全量串行（幂等：state/*.done 存在即跳过）
tmux new -d -s f 'cd /root/Workspace/xy/DiT && bash tools/f_20261101/run_f.sh \
  > assets/f_20261101/logs/f_driver.log 2>&1'

# 单阶段调试
bash tools/f_20261101/run_f.sh s3_smoke
```

## 两臂设定

| | F0 | F1 |
|---|---|---|
| 起点 | K | K |
| 32×32 / 16×16 | **关闭并冻结**（gain=0，参数 requires_grad=False） | 同 F0 |
| 8×8 | 保留，`lr 8e-5` | 同 F0 |
| writer table | `lr 5e-5` | 同 F0 |
| 主干 | `lr 5e-6` | 同 F0 |
| Loss | FM | FM + E-G writer（梯度比 2%） |
| 首轮 | 5k | 5k |

其余全部继承 A 的成熟配方：`batch 512`、`microbatch 512`、`compile`、`ema 0.999`、
`warmup 200`、`min_lr_ratio 0.1`、`grad_clip 1.0`、`std_shape_aug strong`、
`stage_p2.csv`（writer 0–9）、`shards_all`、`vae sd-vae-ft-ema`。

## 关键实现细节

- **gain 是 buffer**：`injections.py` 里 `delta = delta * gain * valid`，置零即精确关闭。
  buffer 进 state_dict，因此**每次 load 之后都要重新置零**（raw 与 EMA 各一次），
  `run_f.sh s3` 用 checkpoint 断言验证 resume 后仍为 0。
- **冻结靠 `requires_grad_(False)`**：`optimizer()` 只收 `requires_grad` 的参数，
  于是 `lr_style_inj` 自动只作用于存活下来的 site 2。
- **E-G 复用 P3B 的 `EGStyleObjective`**：`refs_npz` 覆盖 writer 0–9 × 三书体（23 key），
  首个辅助步自动校准到 `target_ratio=0.02`（冒烟实测 achieved 0.0200）。
- **续训不重置调度**：s7 用 `--resume 0050000.pt --stop 12000`，同一 run 目录、
  同一 identity，cosine 沿 12k horizon 继续，不会二次 warmup。

## 验收顺序（严格按此优先级）

1. **内容不退化**：val100 的 SSIM 不比对照低 0.010 以上、Ink-IoU 不低于 0.010、
   frag 不劣于 +15%；OCR Top-1 不明显下降。
2. **Writer 方向改善**：direction 从 ~0.41 提到 ≥0.50（F1 需比 F0 高 ≥0.03 才算赢）。
3. **Writer 幅度合理**：amplitude 向 0.8–1.0 靠，但不许只放大错误方向。
4. **画质不退化**：SSIM / LPIPS / Ink-IoU / 笔宽维持或改善。

注意：900 张 swap 只含 30 个字符，1350 个 writer-pair 不是 1350 个独立字符，
小幅 direction 变化不能宣布为显著进步；比较时看 `pick.py` 输出的配对结论。

## 已知边界

- 若 F1 只提高 E-G 分数而 direction / 内容不改善 → 选 F0。
- 若两臂都未超过 K → 保留 K 或其无训练简化版（s1 的关 0+1 结构），
  本轮结论即为“清理注入路径 + 弱判别监督不足”，下一代才需要动数据组织或条件监督。
- 本轮 LR 是工程候选，未经消融证明最优。
