"""Global comparison on the two metrics that actually matter: writer style and OCR.

SSIM is deliberately absent from the ranking logic here. Style comes from the frozen
E-G writer discriminator (ResNet18, same-script/different-character, K=64 bank, top5
mean) and OCR from the panel recogniser. Their sources:

  style: docs/1029/domain_metrics/REPORT.md   (AUC/EER/mAP/Style-FID, per model)
         docs/1030/eval_style_v2/EG_RANKING.md (saturation, GT ceilings)
  OCR:   docs/1031/01_EVIDENCE_MATRIX.md + eval_arm outputs, two panels

Two protocols exist and their GT ceilings differ (100-char 92.38, 187-char 89.76), so
raw AUC is NOT cross-comparable. saturation = (AUC-50)/(GT-50) is, and it reproduces
the published sat column exactly (S5 0.896, J0 0.930, DeepSym4@85k 0.895).
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT_CSV = ROOT / 'assets/metrics_index/style_ocr_global.csv'
OUT_MD = ROOT / 'docs/1031/GLOBAL_STYLE_OCR.md'
OUT = ROOT / 'assets/metrics_index'

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

CEIL = {'100字': 92.38, '187字': 89.76}

# model, protocol, auc, eer, map, style_fid, note
EG = [
    ('S5@50k', '100字', 87.98, 20.98, 63.55, 0.07054, 'Stroke×Spatial；style_matrix 风格冠军'),
    ('S4@50k', '100字', 84.91, 23.70, 61.75, 0.06941, 'LocalCNN + Spatial AdaGN'),
    ('S7@50k', '100字', 83.84, 24.71, 60.45, 0.06508, 'Style-FID 最低但身份核验不是最好'),
    ('S1@50k', '100字', 83.50, 24.84, 56.11, 0.06976, 'Global AdaGN'),
    ('P2@20k', '100字', 83.19, 24.46, 57.45, 0.08391, '注意：E-G 测于 20k，OCR 测于 70k'),
    ('S8@50k', '100字', 82.26, 25.22, 57.86, 0.06965, ''),
    ('S0@50k', '100字', 82.13, 25.83, 58.22, 0.08379, '内容基线'),
    ('S6@50k', '100字', 81.89, 26.16, 61.05, 0.07493, ''),
    ('S2@50k', '100字', 81.81, 26.48, 56.51, 0.06977, ''),
    ('S3@50k', '100字', 81.26, 27.93, 57.88, 0.07074, ''),
    ('S9@50k', '100字', 80.35, 27.66, 53.06, 0.07684, '复杂结构反而低于基线'),
    ('S11@50k', '100字', 80.32, 27.52, 55.51, 0.08239, ''),
    ('P1@10k', '100字', 76.32, 31.12, 46.92, 0.08733, '注意：E-G 测于 10k，OCR 测于 55k'),
    ('J0@5k', '187字', 87.01, 20.33, 64.47, 0.05877, 'FM 对照，5k 即高饱和'),
    ('J2@5k', '187字', 86.86, 20.22, 64.19, 0.05319, 'Decoder 联训'),
    ('J7@5k', '187字', 86.58, 20.29, 62.11, 0.05794, ''),
    ('U100Base@55k', '187字', 86.50, 20.38, 62.18, 0.06840, '★历史成熟模型'),
    ('J1@5k', '187字', 86.47, 20.86, 63.23, 0.05934, ''),
    ('J3@5k', '187字', 86.40, 20.92, 63.12, 0.05743, ''),
    ('J4@5k', '187字', 86.36, 20.66, 60.98, 0.05840, ''),
    ('J6@5k', '187字', 86.29, 20.66, 61.34, 0.05908, ''),
    ('J5@5k', '187字', 86.25, 20.87, 61.63, 0.05863, ''),
    ('DeepSym4@85k', '187字', 85.57, 21.92, 61.61, 0.07022, '★成熟教师模型'),
    ('DeepSym4@75k', '187字', 85.15, 22.25, 59.90, 0.07596, ''),
    ('S11@50k(187)', '187字', 81.61, 25.80, '', '', '同模型 187 面板复测'),
    ('S9@50k(187)', '187字', 81.21, 26.77, '', '', '同模型 187 面板复测'),
]

# OCR: model -> (top1, panel, n, note)
OCR = {
    'D3@11k(起点错)': (78.89, 'val100', 90, 'X 级阴性对照，但 OCR 最高'),
    'D3+D1dec': (78.89, 'val100', 90, 'X'),
    'S4@50k': (77.78, 'val100', 90, ''),
    'K@43,429': (77.78, 'val100', 90, '★当前主线节点'),
    'K+D1dec': (77.78, 'val100', 90, '★decoder 换装，OCR 不变'),
    'P2@70k': (75.56, 'val100', 90, ''),
    'D1@2.5k(起点错)': (75.56, 'val100', 90, 'X'),
    'P3A@75k': (74.44, 'val100', 90, ''),
    'D2@3k(起点错)': (74.44, 'val100', 90, 'X'),
    'S0@50k': (71.11, 'val100', 90, 'SSIM 第一但 OCR 偏低'),
    'P0@50k': (67.78, 'val100', 90, ''),
    'P1@55k': (67.78, 'val100', 90, ''),
    'S0@85k': (81.29, 'n187', 187, '187 面板，不与 val100 比'),
    'J5/J6/J7@5k': (82.00, 'n187', 187, '区间 81.4–82.6，取中值绘图'),
    'S5@85k': (77.20, 'n187', 187, '区间 75.4–79.0，三 seed'),
}


def sat(auc, proto):
    return (auc - 50.0) / (CEIL[proto] - 50.0)


def main():
    rows = []
    for name, proto, auc, eer, m, fid, note in EG:
        rows.append(dict(metric='E-G AUC', model=name, protocol=proto, value=auc,
                         sat=round(sat(auc, proto), 4), eer=eer, map=m,
                         style_fid=fid, note=note))
    for name, (v, panel, n, note) in OCR.items():
        rows.append(dict(metric='OCR top1', model=name, protocol=panel, value=v,
                         sat='', eer='', map='', style_fid='', note=note))

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['metric', 'model', 'protocol', 'value',
                                         'sat', 'eer', 'map', 'style_fid', 'note'])
        w.writeheader()
        w.writerows(rows)
    print(f'wrote {OUT_CSV} ({len(rows)} rows)')

    md = ['# 全局对比：书家风格 × OCR（不看 SSIM）\n',
          '> 由 `tools/poster_20261101/build_style_table.py` 生成。SSIM 不参与任何排序。\n',
          '\n## 0. 两条轴的定义与不可比性\n',
          '\n**风格轴 = E-G 书家身份核验。** 冻结 ResNet18 书家判别器的 512 维特征；'
          '协议为同书体、排除同字、K=64 真迹参考、top5 均值；AUC 为 pooled。'
          '已验收可靠（real 跨字 95.98，非低阶捷径）。\n',
          '\n**OCR 轴 = 面板识别率 top1。** 两块面板不可混：`val100`（n=90，10 张不可分割）'
          '与 `n187`（n=187）。\n',
          '\n**跨协议必须看饱和度。** 两块面板的 GT 天花板不同（100字 92.38、187字 89.76），'
          '原始 AUC 不能横比。归一化 `sat = (AUC−50)/(GT−50)`，本表复算与已发布值逐位一致'
          '（S5 0.896 / J0 0.930 / DeepSym4@85k 0.895）。\n',
          '\n## 1. 风格轴：E-G 书家身份核验（按饱和度排序）\n',
          '\n| # | 模型 | 协议 | AUC↑ | sat↑ | EER↓ | mAP↑ | Style-FID↓ | 备注 |\n',
          '|---:|---|---|---:|---:|---:|---:|---:|---|\n']
    for i, r in enumerate(sorted(rows, key=lambda r: -r['sat'] if r['metric'] == 'E-G AUC'
                                else 0)[:len(EG)], 1):
        if r['metric'] != 'E-G AUC':
            continue
        md.append(f'| {i} | {r["model"]} | {r["protocol"]} | {r["value"]:.2f} | '
                  f'{r["sat"]:.3f} | {r["eer"]:.2f} | {r["map"] or "—"} | '
                  f'{r["style_fid"] or "—"} | {r["note"]} |\n')

    md.append('\n## 2. OCR 轴\n')
    md.append('\n### 2.1 val100 面板（n=90，可与 S/P 组横比）\n\n')
    md.append('| # | 模型 | OCR top1 | 备注 |\n|---:|---|---:|---|\n')
    v = sorted([(k, x) for k, x in OCR.items() if x[1] == 'val100'],
               key=lambda t: -t[1][0])
    for i, (name, (val, _, _, note)) in enumerate(v, 1):
        md.append(f'| {i} | {name} | {val:.2f} | {note} |\n')
    md.append('\n### 2.2 n187 面板（不可与 2.1 比）\n\n')
    md.append('| 模型 | OCR top1 | 备注 |\n|---|---:|---|\n')
    for name, (val, _, _, note) in OCR.items():
        if OCR[name][1] == 'n187':
            md.append(f'| {name} | {val:.2f} | {note} |\n')

    md.append('\n## 3. 两条轴的联合覆盖：这才是真正的问题\n')
    md.append('\n| 模型 | E-G AUC | val100 OCR | 同一步同协议都有？ |\n|---|:-:|:-:|:-:|\n')
    joint = [
        ('S0@50k', '82.13', '71.11', '✅'),
        ('S4@50k', '84.91', '77.78', '✅'),
        ('S5@50k', '87.98', '—', '❌ 缺 OCR'),
        ('S7@50k', '83.84', '—', '❌ 缺 OCR'),
        ('S1@50k', '83.50', '—', '❌ 缺 OCR'),
        ('S2/S3/S6/S8/S9/S11', '80.3–82.3', '—', '❌ 缺 OCR'),
        ('P1', '76.32@10k', '67.78@55k', '⚠ 步数不同'),
        ('P2', '83.19@20k', '75.56@70k', '⚠ 步数不同'),
        ('P3A@75k', '—', '74.44', '❌ 缺 E-G'),
        ('P0@50k', '—', '67.78', '❌ 缺 E-G'),
        ('K@43,429', '—', '77.78', '❌ 缺 E-G'),
        ('K+D1dec', '—', '77.78', '❌ 缺 E-G'),
        ('J0–J7@5k', '86.3–87.0', '81.4–82.6 (n187)', '✅ 但面板不同'),
        ('U100Base@55k', '86.50', '—', '❌ 缺 OCR'),
        ('DeepSym4@85k', '85.57', '—', '❌ 缺 OCR'),
    ]
    for name, eg, ocr, ok in joint:
        md.append(f'| {name} | {eg} | {ocr} | {ok} |\n')

    md.append('\n## 4. 结论\n')
    md.append('\n1. **风格冠军是 S5@50k**（AUC 87.98、sat 0.896、EER 20.98、mAP 63.55），'
              '按原始 AUC 超过全部历史模型。但按饱和度它只是与 DeepSym4@85k（0.895）持平，'
              '而 J 组（0.91–0.93）更高 —— 公平说法是「S5 用 50k 达到了成熟模型 85k 的相对水平」，'
              '不是「S5 超过成熟模型」。\n')
    md.append('2. **S 家族内部排序真实存在且与结构假设一致**：'
              'S5(0.896) > S4(0.824) > S7(0.791) ≈ S1(0.791) > S8(0.762) ≈ S0(0.758) > '
              'S6(0.753) ≈ S2(0.751) ≈ S3(0.739)。**S0 作为内容基线，风格只排到第 7**。\n')
    md.append('3. **OCR 与风格是两条独立的轴，不能互相替代**：S0 的 SSIM 全场第一、'
              'OCR 只有 71.11（val100 倒数第三）；S5 风格全场第一、OCR 从未在 val100 测过。'
              'P0/P1 的 OCR 只有 67.78，说明 P 链这一路在可读性上是明显退步的。\n')
    md.append('4. **唯一同时具备强风格 + 强 OCR 的节点是 S4@50k**'
              '（E-G 84.91 / sat 0.824 / OCR 77.78）。这是目前唯一可以完整主张的「双轴都好」。\n')
    md.append('5. **K 的风格轴是完全空白**，不是「低」。E-G、E-L、Style-FID/KID、'
              'Density/Coverage 四套指标一次都没在 K 或任何 D 臂上跑过。'
              '所以「K 的风格如何」现在无法回答，任何断言都是缺数据。\n')
    md.append('6. **K 的 `direction` 不能代替 E-G。** direction 在 latent 上算（同噪声换书家），'
              'E-G 在读解码后的图像 —— 两者测的不是同一件事，数值也无换算关系。'
              '正因为如此，decoder 换装能改 E-G（读图像），却**证明性地不能改 direction**'
              '（latent 不变，实测 K 与 K+D1dec 都是 0.5571）。\n')

    md.append('\n## 5. 正确的 D1 结果（按风格 + OCR 两条轴重述）\n')
    md.append('\nD1 原先跑错了起点（style 分支随机初始化），所以它的 0.5839 / direction 0.1607 '
              '都不是「从 K 只训 decoder」。正确版本有两层：\n')
    md.append('\n**(a) 零训练：把 D1 的 decoder 直接装到 K 上**（`K + D1dec`）—— 已完成，'
              '100 张同图同种子配对 bootstrap 1e4：\n')
    md.append('\n| 指标 | K | K+D1dec | delta | 95% CI | 胜/平/负 | 判定 |\n')
    md.append('|---|---|---|---|---|---|---|\n')
    md.append('| SSIM ↑ | 0.5946 | 0.5982 | +0.0036 | [+0.0023, +0.0052] | 96/0/4 | 显著改善 |\n')
    md.append('| LPIPS ↑ | 0.3462 | 0.3416 | +0.0047 | [+0.0040, +0.0053] | 90/0/10 | 显著改善 |\n')
    md.append('| Ink-SSIM ↑ | 0.4423 | 0.4456 | +0.0033 | [+0.0022, +0.0046] | 97/1/2 | 显著改善 |\n')
    md.append('| Ink-IoU ↑ | 0.2886 | 0.2893 | +0.0007 | [+0.0005, +0.0009] | 83/0/17 | 显著改善 |\n')
    md.append('| 碎片比 ↓ | 2.6578 | 2.3765 | −0.2813 | [−0.6695, −0.0417] | 29/61/10 | 显著改善 |\n')
    md.append('| 容差骨架 F1 ↑ | 0.0258 | 0.0253 | −0.0005 | [−0.0010, −0.0000] | 41/0/59 | 显著退化 −2% |\n')
    md.append('| **面板 OCR** | **77.78** | **77.78** | 0 | — | — | **完全不变** |\n')
    md.append('| **direction** | **0.5571** | **0.5571** | 0 | — | — | **构造上必然不变** |\n')
    md.append('| **E-G AUC** | 未测 | 未测 | — | — | — | **未知** |\n')
    md.append('\n**(b) 正确起点重训 decoder**（`D1b`，生成器与 K 逐位相同，已用 checkpoint 断言 '
              '368/368 tensor max|diff|=0）：跑到 1810/2500，ETA 约 16:50。\n')
    md.append('\n### 5.1 按你关心的两条轴，正确的 D1 结果是：\n')
    md.append('\n- **OCR：没有任何收益，也没有任何损失**（77.78 → 77.78，逐位相同）。'
              '这说明 decoder 的改进全在像素保真与墨迹连续性上，没有改变字形可读性。'
              '所以「D1 能提升 OCR」这个期望是不成立的。\n')
    md.append('- **风格：目前无法回答。** K+D1dec 与 D1b 都没有 E-G AUC。'
              '而 E-G 恰好是读图像的，所以 decoder 是有可能改变它的 —— 这是唯一还有悬念的一项。\n')
    md.append('- **内容/墨迹：确实显著改善**（SSIM +0.0036、LPIPS +0.0047、碎片比 −0.28，'
              '五项显著、胜率 83–97%），只有容差骨架 F1 轻微退化 −2%。\n')
    md.append('- **decoder 步数扫描显示已饱和**：SSIM 增量 +0.0014/+0.0009/+0.0008/+0.0003/+0.0002，'
              '1500 步就拿到约 90% 的收益。所以把 D1b 训到 2500 步是够的，不需要再加。\n')
    md.append('- **D1 从未在 K 上退化。** 之前「decoder 配 K 会退化」的读法来自随机 style 分支；'
              'K+D1dec 在每一项上都优于 D3+D1dec（SSIM 0.5982 vs 0.5965、LPIPS 0.3416 vs 0.3428、'
              'Ink-IoU 0.2893 vs 0.2854、碎片 2.377 vs 2.389）。\n')

    md.append('\n## 6. K 的 E-G 补测：脚本就位但拒绝发布数字\n')
    md.append('\n已定位官方管线（`tools/eval_unpaired/run.py`，冻结 ResNet18 + `top10_style23` '
              '3454 张真迹参考库 + `assets/style_disc_ce_holdout.pt`），并写了 '
              '`tools/poster_20261101/eg_for_arm.py` 用现成缓存给任意面板算 E-G。\n')
    md.append('\n**但它复现不出已公布值，所以不发布。** 用 S4 面板做对照：\n')
    md.append('\n| 来源 | S4 E-G AUC | EER | mAP |\n|---|---:|---:|---:|\n')
    md.append('| 已公布（`eval_domain_20261029/4090/S4_50000_heldout/result.json`, pooled） | '
              '84.91 | 23.70 | 61.75 |\n')
    md.append('| `eval_style_v2.evaluate_feature`（k=64，同一面板同一库） | 85.89 | 22.42 | 72.70 |\n')
    md.append('\n差 0.98pp，`n_valid=98` 与公布值一致，说明面板与库都对了，是**聚合实现不同**'
              '（mAP 差 11pp 尤其明显）。按「指标不得用私有实现重算」的铁律，'
              'K 的 E-G 必须走 `tools/eval_unpaired/run.py` 原路重算，'
              '而不是接受 85.89 这一支。\n')
    md.append('\n> 下一步：确认 `run.py` 的 `--reference-csv`（参考库来自 `data/top10_style23` '
              '3454 张，需找到对应清单 csv），然后对 K_A 与 K_plus_D1dec 两个面板各跑一次'
              '（CPU 即可，不抢训练 GPU）。\n')

    OUT_MD.write_text(''.join(md), encoding='utf-8', newline='\n')
    print(f'wrote {OUT_MD}')
    charts(rows)


def charts(rows):
    eg = [r for r in rows if r['metric'] == 'E-G AUC']
    eg.sort(key=lambda r: -r['sat'])
    fig, ax = plt.subplots(figsize=(11, 8))
    names = [r['model'] for r in eg]
    vals = [r['sat'] for r in eg]
    cols = ['#c0392b' if r['protocol'] == '100字' else '#2980b9' for r in eg]
    bars = ax.barh(names, vals, color=cols, edgecolor='white')
    for b, r in zip(bars, eg):
        ax.text(r['sat'] + .006, b.get_y() + b.get_height() / 2,
                f'{r["sat"]:.3f}  (AUC {r["value"]:.2f})', va='center', fontsize=8)
    ax.set_xlabel('饱和度 sat = (AUC−50)/(GT−50)　—— 跨协议唯一可比值')
    ax.set_title('风格轴：E-G 书家身份核验（红=100字面板 GT 92.38，蓝=187字面板 GT 89.76）\n'
                 'S5 按原始 AUC 第一，按饱和度只与 DeepSym4@85k 持平',
                 fontsize=12.5, pad=13)
    ax.set_xlim(0.55, 1.0)
    ax.grid(axis='x', alpha=.25, ls=':')
    fig.tight_layout()
    p = OUT / 'fig_style_eg.png'
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print('wrote', p)

    # joint scatter: sat vs OCR, with the unmeasured region made explicit
    both = [('S0@50k', 0.758, 71.11), ('S4@50k', 0.824, 77.78)]
    fig, ax = plt.subplots(figsize=(9.5, 7))
    ax.axvspan(0.55, 0.60, color='#f4f6f8', zorder=0)
    ax.text(0.575, 80.5, 'K / K+D1dec\nE-G 未测\n(OCR 77.78)', fontsize=9,
            color='#c0392b', ha='center', va='center', fontweight='bold')
    ax.scatter([0.575], [77.78], marker='X', s=200, c='#c0392b', zorder=4)
    ax.text(0.93, 71.11, 'S5@50k\n风格第一\nOCR 未测', fontsize=9, color='#2980b9',
            ha='center', va='center', fontweight='bold')
    ax.scatter([0.93], [71.5], marker='X', s=200, c='#2980b9', zorder=4)
    for name, s, o in both:
        ax.scatter(s, o, s=260, c='#16a085', edgecolor='white', linewidth=1.4, zorder=5)
        ax.annotate(f'{name}\nsat {s:.3f} / OCR {o:.2f}', (s, o), fontsize=10,
                    xytext=(12, -14), textcoords='offset points', fontweight='bold',
                    color='#16a085')
    ax.set_xlabel('风格：E-G 饱和度 ↑')
    ax.set_ylabel('OCR top1 (%) ↑  (val100, n=90)')
    ax.set_title('两条轴的联合覆盖：只有 S0 与 S4 同时被测\n'
                 'X 标记 = 该轴完全没测过，不是分数低', fontsize=12.5, pad=13)
    ax.set_xlim(0.70, 0.95)
    ax.set_ylim(64, 82)
    ax.grid(alpha=.25, ls=':')
    fig.tight_layout()
    p = OUT / 'fig_style_vs_ocr.png'
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print('wrote', p)


if __name__ == '__main__':
    main()
