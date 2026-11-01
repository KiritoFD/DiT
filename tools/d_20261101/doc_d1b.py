"""Insert the D1b launch entry and mark the superseded 11:15 conclusion."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
p = ROOT / 'training.md'
s = p.read_text(encoding='utf-8')

NEW = """2026-11-01 14:05（**D1b 拉起：修正起点后的 decoder-only，4090 tmux `d1b`**）：复查发现
D1/D2/D3 三臂**从未从 K 出发** —— `build_specs.py` 剥掉了 `transfer_style`，而
`common.py:34-46` 只在 `transfer_style` 为真时才复制 `style_representation` / `style_injections`，
于是三臂都是「K 的主干 + 随机初始化的 style 分支」。三个 injection gain 实测为
59.11/66.84/63.86（K 是 14.81/24.27/37.06），且三臂完全相同（`construct()` 用 `config.seed`
确定性初始化），这就是指纹。**不是 EMA 漂移**：`train.py:198` 冻结时跳过 `update_ema`，
实测 `ema_D1 == raw_D1`（差 0.0）。`check_init.py` 证明：当前 spec 与 K 有 33/368 个张量不同
（max 36.06 @ `style_injections.2.gain`），加 `transfer_style=True` 后 368/368 全等。
该混淆推翻了首轮结论 —— 用 `--decoder-from` 把 D1 的 decoder 挂到 K 上，SSIM +0.0036
（96/100 胜）、LPIPS +0.0047（90/100）、Ink-SSIM +0.0033、Ink-IoU +0.0007、碎片比 +0.2813
五项显著改善，只有容差骨架 F1 −0.0005 显著退化；且 K+D1dec 每项都优于 D3+D1dec，
说明 decoder 从未在 K 上退化。decoder 步数扫描（K 生成器 × D1 的 500/1000/1500/2000/2500）
单调改善且 1500 步已得约 90%，面板 OCR 全程 77.78 不动，骨架 F1 单调缓降 0.0258→0.0253。
**D1b** 从 4090 上已调好的 D1.json 派生（保留 `image_batch 64 / image_micro 8`），
仅改 `transfer_style=True` + 身份，2500 步约 2h。启动即核验：`transfer_style: 34`、
`strict_backbone_equal: True`、`frozen 362 generator tensors`、`grad_norm 0.0`、
`decoder_grad 0.44–3.40`、19.8 GiB / 95% / 416 W。首 checkpoint（500 步）落盘后用
`verify_d1b.py` 断言生成器与 K **逐位相同**。详见 [docs/d_results.md](docs/d_results.md)。

"""

OLD_HEAD = '2026-11-01 11:15（**D 计划首轮完成，4090 已空闲**）：'
MARK = ('2026-11-01 11:15（**D 计划首轮完成，4090 已空闲**）'
        '〔**已被 14:05 条目推翻**：三臂起点错误，结论不成立〕：')

if 'D1b 拉起' in s:
    print('already present')
else:
    i = s.index('# 训练进度\n\n') + len('# 训练进度\n\n')
    s = s[:i] + NEW + s[i:]
    s = s.replace(OLD_HEAD, MARK, 1)
    p.write_text(s, encoding='utf-8', newline='\n')
    print('training.md updated')

c = ROOT / 'ckpts.md'
cs = c.read_text(encoding='utf-8')
CK = """2026-11-01 14:05（D1b 起跑，4090）：`assets/d_20261101/execution/runs/D1b/checkpoints/`，
spec `assets/d_20261101/specs/D1b.json`。与 D1 的唯一区别是 `transfer_style: True` ——
D1 的起点实际是「K 主干 + 随机 style 分支」，D1b 才是真正的「从 K 出发」。
生成器全程冻结，落盘后须用 `tools/d_20261101/verify_d1b.py` 断言与 K 逐位相同。

"""
if 'D1b 起跑' not in cs:
    j = cs.index('# Checkpoint登记\n\n') + len('# Checkpoint登记\n\n')
    cs = cs[:j] + CK + cs[j:]
    c.write_text(cs, encoding='utf-8', newline='\n')
    print('ckpts.md updated')
else:
    print('ckpts.md already present')
