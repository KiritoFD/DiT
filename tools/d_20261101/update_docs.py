"""Append the D-plan result entry to training.md / ckpts.md (one-off doc edit)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

TRAIN = """2026-11-01 11:15（**D 计划首轮完成，4090 已空闲**）：D1/D2/D3 串行跑完并全部评测，
外加不额外训练的交叉组合 D3+D1dec。实测步速与 ETA 与预估一致：D1 04:30→06:29（2500 步）、
D2 06:29→08:45（3000 步）、D3 08:45→10:44（11000 步）、评测 10:44→10:59。**结论：K 仍最优，
D 计划没有产出可替换它的权重**——配对检验（100 张同图同种子，bootstrap 1e4）里只有退化达到显著：
D1 的 LPIPS −0.0100 与 Ink-IoU −0.0256、D2 的 Ink-IoU −0.0132；D3 全部 n.s.；
D3+D1dec 观感最好（SSIM 0.5965 / LPIPS 0.3428 / OCR top5 98.44 / 碎片比 2.389，均为四臂最优）
但 CI 全部含 0。**机制**：图像域损失奖励平滑高墨量渲染，decoder 靠加粗+合并笔画把碎片比压下去
（端点密度 0.0008→0.0004、连通块 10.16→8.28、笔宽 0.970→1.004 与 GT 齐平），
所以碎片比下降是假改善、粘连加重，Ink-IoU 与 OCR 同步下滑；D1 单独使用时 Writer 身份崩塌
（direction 0.5454→0.1607）。D3 是唯一不伤 Writer 身份的臂（0.5372/0.653）。
**不建议据此追加 2–4h**，增益在 100 字面板上与噪声不可分。另修一处评测缺陷：
`t3_writer_swap` 原先不认 `--decoder-from`，交叉组合的 t3 报告与 D3 md5 完全相同；
direction/amplitude 在 latent 空间算本就不受影响，但 OCR 走图像，原先拿的是 D3 的原样结果，
补跑后为 96.22/98.44。结果表与判据见 [docs/d_results.md](docs/d_results.md)，对照图见
`assets/d_20261101/review/`。

"""

CKPT = """2026-11-01 11:15（D 计划三臂权重落盘，4090 `/root/Workspace/xy/DiT`）：均从 K
（`assets/abc_20261031/execution/runs/A/checkpoints/0043429.pt` 的 EMA）出发，
路径相对 `assets/d_20261101/`。**三臂均未超越 K，不作为交付权重保留，仅供诊断复现。**

| 臂 | 权重 | 步数 | ssim | lpips | ink_iou | 备注 |
|---|---|---|---|---|---|---|
| D1 | `execution/runs/D1/checkpoints/0002500.pt` | 2,500 | 0.5839 | 0.3562 | 0.2630 | decoder-only；**Writer 身份崩塌**（direction 0.1607） |
| D2 | `execution/runs/D2/checkpoints/0003000.pt` | 3,000 | 0.5932 | 0.3466 | 0.2754 | U-Net+decoder 联合；ink_iou 显著变差 |
| D3 | `execution/runs/D3/checkpoints/0011000.pt` | 11,000 | 0.5927 | 0.3477 | 0.2847 | 纯 FM + 困难字过采样；**不伤 Writer 身份**（0.5372） |

交叉组合 `D3 U-Net × D1 decoder` 不占权重（评测时用 `--decoder-from` 临时拼装），
指标为 ssim 0.5965 / lpips 0.3428 / ink_iou 0.2854 / OCR 96.22，四臂最优但全部不显著。
完整评测在 `eval/{D1_2500,D2_3000,D3_11000,D3_11000_plus_D1dec}/`。

"""

for name, text, anchor in (('training.md', TRAIN, '# 训练进度\n\n'),
                           ('ckpts.md', CKPT, '# Checkpoint登记\n\n')):
    p = ROOT / name
    s = p.read_text(encoding='utf-8')
    if 'D 计划首轮完成' in s or 'D 计划三臂权重落盘' in s:
        print(f'{name}: already present')
        continue
    i = s.index(anchor) + len(anchor)
    p.write_text(s[:i] + text + s[i:], encoding='utf-8', newline='\n')
    print(f'{name}: inserted at top entry')
