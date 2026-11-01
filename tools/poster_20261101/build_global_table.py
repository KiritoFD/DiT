"""Global milestone comparison: every checkpoint in Milestones.md vs this session's work.

Two things make a naive merged table wrong, so both are handled explicitly:

1. `Milestones.md` mixes OCR denominators. Its val100-family numbers are out of the
   90 images the OCR can segment (S0@50k 71.11% = 64/90, S4@50k 77.78% = 70/90), while
   its 187-family numbers are out of 187 (S0@85k 81.29% = 152/187). P3A's 74.44 also
   lands on 74.44/91.11 exactly as our own eval_arm reports, which is what proves the
   two protocols are the same one and lets the val100 column be compared honestly.
2. SSIM/LPIPS come from at least six incompatible implementations (MoYi Strict,
   docs/107 unified-187, docs/108 in-memory, v83-A's own arm protocol, the S/J 187
   panel, and our val100). Each row carries its `protocol` and the chart never joins
   across protocols.

Rows are transcribed from `Milestones.md` (cited per row) or pulled from
`assets/metrics_index/lineage_summary.csv`; nothing is recomputed or interpolated.
"""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT_CSV = ROOT / 'assets/metrics_index/global_milestones.csv'
OUT_MD = ROOT / 'docs/1031/GLOBAL_MILESTONES.md'
LOCAL = ROOT / 'assets/metrics_index/lineage_summary.csv'
E200_CSV = ROOT / 'assets/metrics_index/eval200_cross_era.csv'

# id, name, grade, protocol, steps, ssim, lpips, iou, ocr, ocr_n, ocr_panel,
# eg_auc, note, source
# grade: A/B/X per Milestones.md; '-' = this session
HIST = [
    # ---- 第一~十三纪元：早期节点（转录自 docs/04_experiments/00_exhaustive_experiment_chronicle.md）----
    ('v9a', 'v9a REPA 预训练', 'C', 'native_old', '132k', '', '', '', '', '', '', '',
     '第一纪元；DINOv2 表征对齐，无量化指标', 'chronicle 第一纪元'),
    ('v10b', 'v10b 初代骨架长跑', 'A', 'native_old', '390k', 0.5058, '', '', '', '', '', '',
     '第二纪元；Seen SSIM 0.763 vs Strict 0.5058，开集概念证明', 'chronicle 第二纪元'),
    ('v11', 'v11 struct-loss', 'X', 'native_old', '500k', 0.5670, '', '', '', '', '', '',
     '第三纪元；12ch + 拉普拉斯结构损失，步速降 45%', 'chronicle 第三纪元'),
    ('v12', 'v12 全局池化', 'X', 'native_old', '', '', '', '', '', '', '', '',
     '第四纪元；骨架压成全局向量 → 拓扑塌陷（死胡同 #3）', 'chronicle 第四纪元'),
    ('v13', 'v13 黄金底模', 'A', 'native_old', '125k', 0.5713, 0.3800, '', '', '', '', '',
     '第五纪元；LPIPS 0.380；索引错位 Bug 反证零样本泛化', 'chronicle 第五纪元'),
    ('v14', 'v14 style87', 'B', 'native_old', '187.5k', 0.5764, '', '', '', '', '', '',
     '第六纪元；87 类细分触及当时峰值，但长尾过拟合', 'chronicle 第六纪元'),
    ('v15', 'v15-fs6 沈周', 'B', 'native_old', '2k', 0.5568, '', '', '', '', '', '',
     '第七纪元；仅 2000 步微调即 0.5568，证明书家流形局部线性', 'chronicle 第七纪元'),
    ('v17', 'v17 LCA 交叉注意力', 'X', 'native_old', '', 0.0893, '', '', '', '', '', '',
     '第八纪元；数值崩溃（死胡同 #1）', 'chronicle 第八纪元'),
    ('v21', 'v21 SkelNet (JointLN)', 'X', 'native_old', '155k', '', '', '', '', '', '', '',
     '第十纪元；骨架方差 38× 风格 → 风格被归一化吞噬（死胡同 #5）', 'chronicle 第十纪元'),
    ('v23', 'v23 SplitLN', 'A', 'native_old', '85k', '', '', '', '', '', '', '',
     '第十纪元；修复后风格振幅暴涨 4.24×，确立骨架路线标准解', 'chronicle 第十纪元'),
    ('v26', 'v26 真迹骨架上限探测', 'C', 'native_old', '', '', '', '', '', '', '', '',
     '第十一纪元；证明上限但现实中拿不到未见字真迹骨架', 'chronicle 第十一纪元'),
    ('v54', 'v54 纯字表', 'B', 'inmem108', '150k', 0.5960, '', '', '', '', '', '',
     '第十二纪元；docs/107 统一口径约 0.5890', 'Milestones §1'),
    ('v66', 'v66 三表+route', 'B', 'u187', '150k', 0.5976, '', '', '', '', '', '',
     '第十二纪元；in-mem 0.5955；blocks(2,4,5,6) 黄金靶点', 'Milestones §1'),
    ('v68', 'v68 C2OT 旗舰', 'A', 'u187', '200k', 0.6169, '', '', '', '', '', '',
     '第十二纪元；in-mem 0.6149；SDE 0.6289 属另一采样协议', 'Milestones §1'),
    ('v70_30', 'v70 纯骨架 @30k', 'X', 'native_old', '30k', 0.5457, '', '', '', '', '', '',
     '第十三纪元；衰减率仅 0.1712，抗跌但与 v68 差距大', '09_quartile'),
    ('v70', 'v70 纯骨架 @65k', 'X', 'inmem108', '65k', 0.5535, '', '', '', '', '', '',
     '第十三纪元；65k 后平台，约 78.75k 停止', 'Milestones §1'),
    ('v72_50', 'v72 Calli-VAE', 'B', 'inmem108', '50k', 0.5999, '', '', '', '', '', '',
     '同阶段 v68 为 0.5701', 'Milestones §1'),
    ('v72_65', 'v72 Calli-VAE', 'B', 'inmem108', '65k', 0.6034, '', '', '', '', '', '',
     '同阶段 v68 为 0.5833', 'Milestones §1'),
    ('ditb', '48 DiT-B + v66route', 'B', 'strict40000', '40k', 0.6104, 0.3243, '', '', '', '',
     '', 'Seen SSIM 0.7301 / LPIPS 0.1832；大模型容量 × 黄金路由', '08_evolution §3'),
    ('moyi12', 'MoYi 12ch', 'A', 'moyi_strict', '50k', 0.5935, 0.3724, '', '', '', '', '',
     '骨架 IoU 0.0177；MSE 0.7983；官方多通道锚点', '08_evolution §3'),
    ('moyi4_50', 'MoYi 4ch', 'A', 'moyi_strict', '50k', 0.5949, 0.3678, '', '', '', '', '',
     '骨架 IoU 0.0161；与 12ch 同步数对照', 'Milestones §1'),
    ('moyi4_80', 'MoYi 4ch', 'A', 'moyi_strict', '80k', 0.6085, 0.3489, '', '', '', '', '',
     '骨架 IoU 0.0235（+46%）；187 统一口径 0.5986', 'Milestones §1'),
    # ---- 后期历史节点 ----
    ('v80', 'v80', 'A', 'native_old', '190k', 0.6168, 0.3247, '', '', '', '', '',
     '历史原生协议最高 SSIM 之一', '01_EVIDENCE_MATRIX §2'),
    ('v83a', 'v83-A arm4', 'A', 'v83arm4', '100k', 0.6098, 0.3252, 0.3467, '', '', '', '',
     'LPIPS-VGG；无增强在四臂综合领先', 'Milestones §1'),
    ('v84a', 'v84A STD-soft', 'A', 'native_old', '200k', 0.5846, 0.3551, 0.2840, '', '', '',
     '', '取消 char-table，面向开集的独立成熟锚点', 'Milestones §1'),
    ('v88', 'v88 Writer×Glyph', 'X', 'native_old', '170k', 0.5776, 0.4189, '', '', '', '',
     '', '双线性交叉调制未提高泛化的长训阴性', 'Milestones §1'),
    ('deep85', 'Deep-Sym4', 'A', 'SJ187', '85k', 0.6149, '', '', '', '', '', '',
     'J 面板；J@5k 不是冷启动', 'Milestones §1'),
    ('j0', 'J0', 'A', 'SJ187', '5k', 0.6149, 0.3220, '', '', '', '', '',
     'FM、冻结 Decoder', 'Milestones §1'),
    ('j2', 'J2', 'A', 'SJ187', '5k', 0.6198, 0.3150, '', '', '', '', '',
     '解冻 Decoder + 图像 Loss + GT 重建', 'Milestones §1'),
    ('j5', 'J5/J6/J7', 'B/X', 'SJ187', '5k', '', '', '', '81.4–82.6', '187', 'n187',
     '', 'OCR heldout187；三臂差异不明显', 'Milestones §1'),
    # S / P rows: val100 metrics + val100 OCR where measured
    ('s0_50', 'S0', 'A', 'val100', '50k', 0.60275, 0.33950, '', '71.11', '90', 'val100',
     '82.13', '简单 Writer Vector + Global AdaGN；内容强基线', 'Milestones §1 + §OCR'),
    ('s4_50', 'S4', 'A', 'val100', '50k', 0.59875, '', 0.29513, '77.78', '90', 'val100',
     '84.91', 'LocalCNN + Spatial AdaGN；@85k 未完成不得标成结果', 'Milestones §1 + §OCR'),
    ('s5_50', 'S5', 'A', 'val100', '50k', 0.59750, '', '', '', '', '', '87.98',
     '局部 Attention×CrossAttn；内容未同步领先', 'Milestones §1'),
    ('s0_85', 'S0', 'A', 'u187', '85k', '', '', '', '81.29', '187', 'n187', '',
     'val100 口径 SSIM 0.6000（见 val100 组）', 'Milestones §1 + §OCR'),
    ('s5_85', 'S5', 'A', 'SJ187', '85k', '', '', '', '75.4–79.0', '187', 'n187',
     '87.68/84.17/86.17', '三 seed E-G AUC', 'Milestones §1 + §OCR'),
    ('p0', 'P0', 'A', 'val100', '50k', 0.5573, 0.3841, 0.2258, '67.78', '90', 'val100',
     '53.69', '通用', 'Milestones §1 + §OCR'),
    ('p1', 'P1', 'A', 'val100', '55k', 0.5636, 0.3735, 0.2440, '67.78', '90', 'val100',
     '72.22', '混合', 'Milestones §1 + §OCR'),
    ('p2', 'P2', 'A', 'val100', '70k', 0.5857, 0.3527, 0.2723, '75.56', '90', 'val100',
     '77.77', 'Top10', 'Milestones §1 + §OCR'),
    ('p3a', 'P3A', 'A', 'val100', '75k', 0.5857, 0.3528, 0.2725, '74.44', '90', 'val100',
     '', 'FM；OCR 74.44/91.11 与我们 eval_arm 逐位一致 → 证明两处 OCR 同协议', 'Milestones §1 + §OCR'),
    ('p3b', 'P3B', 'A', 'val100', '75k', 0.5881, 0.3505, 0.2745, '', '', '', '',
     'FM+EG', 'Milestones §1'),
]

# §5 local supplements: 10-20 image subsets, a different (and weaker) protocol
LOCAL_SUBSET = [
    ('S0@85k', '89.47', '19', 0.688, '全量 81.29%'),
    ('S4@50k', '90.00', '10', 0.662, '全量 77.78%'),
    ('S0@50k', '80.00', '10', 0.664, '全量 71.11%'),
    ('P3A@75k', '70.00', '10', 0.661, '全量 74.44/91.11%'),
    ('A@25k', '80.00', '20', 0.668, '非历史里程碑，C31 第一评测点'),
]

# our new nodes: label -> how it maps into the global view
NEW = [
    ('K (A@43429)', 'K', 'A', 'K = A@43429，abc 链末端'),
    ('K + D1dec', 'K+D1dec', 'A', '★ 只换 decoder，生成器不动'),
    ('D3 (起点错)', 'D3', 'X', '起点错（style 分支随机），仅作阴性对照'),
    ('D1 (起点错)', 'D1', 'X', '起点错，且 decoder 收益被随机分支掩盖'),
    ('D2 (起点错)', 'D2', 'X', '起点错'),
    ('D3 + D1dec', 'D3+D1dec', 'X', '交叉组合'),
    ('D1b', 'D1b', '-', '★ 正确起点 decoder-only（进行中）'),
]


def fmt(v, nd=4):
    if v in (None, ''):
        return ''
    try:
        return f'{float(v):.{nd}f}'
    except (TypeError, ValueError):
        return str(v)


def load_local():
    if not LOCAL.exists():
        return {}
    rows = list(csv.DictReader(open(LOCAL, encoding='utf-8-sig')))
    return {r['label']: r for r in rows if r['block'] == 'val100'}


def main():
    loc = load_local()
    out = []

    for (i, name, grade, proto, steps, ssim, lpips, iou, ocr, ocr_n, ocr_panel,
         eg, note, src) in HIST:
        out.append(dict(id=i, name=name, era='历史', grade=grade, protocol=proto,
                        steps=steps, ssim=fmt(ssim), lpips=fmt(lpips), iou=fmt(iou),
                        ocr=ocr, ocr_n=ocr_n, ocr_panel=ocr_panel, eg_auc=eg,
                        direction='', note=note, source=src))

    for label, short, grade, note in NEW:
        r = loc.get(label)
        if not r:
            print(f'  (not measured yet: {label})')
            out.append(dict(id=short, name=label, era='本次', grade=grade,
                            protocol='val100', steps='43,429' if 'D' not in short else '',
                            ssim='', lpips='', iou='', ocr='', ocr_n='', ocr_panel='',
                            eg_auc='', direction='', note=note, source='eval_arm'))
            continue
        out.append(dict(id=short, name=label, era='本次', grade=grade, protocol='val100',
                        steps=r['step'], ssim=r['ssim'], lpips=r['lpips'], iou=r['iou'],
                        ocr=r['ocr'], ocr_n='90', ocr_panel='val100', eg_auc='',
                        direction=r['direction'], note=note, source=r['source']))

    cols = ['id', 'name', 'era', 'grade', 'protocol', 'steps', 'ssim', 'lpips', 'iou',
            'ocr', 'ocr_n', 'ocr_panel', 'eg_auc', 'direction', 'note', 'source']
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(out)
    print(f'wrote {OUT_CSV} ({len(out)} rows)')

    md = ['# 全局里程碑对比（Milestones.md × 本次工作）\n',
          '> 由 `tools/poster_20261101/build_global_table.py` 生成。历史行转录自 '
          '[Milestones.md](../../Milestones.md)，本次行取自 `eval_arm` 官方输出。'
          '不重算、不插值，缺失留空。\n',
          '\n## 0. 两个必须先解决的坑\n',
          '\n### OCR 分母不一致（已对齐）\n',
          '`Milestones.md` 的 OCR 混了两种分母：val100 族是 **/90**（OCR 可分割的 90 张），'
          '187 族是 **/187**。反推可验证：S0@50k 71.11% = 64/90、S4@50k 77.78% = 70/90、'
          'S0@85k 81.29% = 152/187。\n',
          '决定性证据：P3A 的 74.44/91.11 与本次 `eval_arm` 在同一 val100 面板上算出的'
          '74.44/91.11 **逐位相同** —— 说明两处 OCR 是同一实现、同一面板，val100 的 OCR 列'
          '可以放心横比。187 族的 OCR 另开一列，不与 val100 比。\n',
          '\n### SSIM/LPIPS 至少六套实现（不合并）\n',
          'MoYi 旧 Strict、docs/107 统一 187、docs/108 in-memory、v83-A 自有臂协议、'
          'S/J 的 187 面板、本次 val100 —— 数值不可互相换算。每行带 `protocol`，'
          '图上也按 protocol 分面。\n',
          '\n### 13ecosx 无法定位（已穷举）\n',
          '对全仓 `*.md / *.json / *.csv / *.py / *.txt` 做全量检索，`13ecosx` **只命中 '
          '`Milestones.md` 自身**，没有任何 run 目录、配置或结果文件。与 Milestones §1 自己'
          '「保留待定位项」的标注一致。除非能拿到当年的 run 目录名，否则这个名字无法入表。\n',
          '\n## 1. 全局表\n',
          '\n| 时代 | 名称 | 级 | 协议 | 步 | SSIM | LPIPS | Ink-IoU | OCR top1 | OCR n | E-G AUC | direction | 备注 |\n',
          '|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|\n']
    for r in out:
        md.append(f'| {r["era"]} | {r["name"]} | {r["grade"]} | `{r["protocol"]}` | '
                  f'{r["steps"] or "—"} | {r["ssim"] or "—"} | {r["lpips"] or "—"} | '
                  f'{r["iou"] or "—"} | {r["ocr"] or "—"} | {r["ocr_n"] or "—"} | '
                  f'{r["eg_auc"] or "—"} | {r["direction"] or "—"} | {r["note"]} |\n')

    md.append('\n## 2. 本机子集补算（§5，协议更弱，**不可**与上表并列）\n')
    md.append('\n| ckpt | 子集 OCR | n | 子集 SSIM | 全量对照 |\n|---|---:|---:|---:|---|\n')
    for name, ocr, n, ssim, full in LOCAL_SUBSET:
        md.append(f'| {name} | {ocr}% | {n} | {ssim} | {full} |\n')
    md.append('\n> 子集 OCR 系统性偏高（S4@50k 子集 90% vs 全量 77.78%），因为子集按数值序取，'
              '不是随机抽样。**只能用来看单图质量，不能当指标。**\n')

    # ---- 跨纪元统一标尺 ----
    md.append('\n## 2. 跨纪元统一标尺：eval200fix N=187\n')
    md.append('\n这是全项目唯一一把跨八个月的尺子。已验证：`exp-std/csv/eval200_fixed.csv` 与我们的 '
              '`heldout187_seed0` **是同一批 187 个 img_id**（双向差集都是 0）。'
              '生成配置也一致：Heun / 50 步 / CFG 1.0 / ODE / sd-vae-ft-ema / std_soft(σ=0.65)。'
              '**唯一差别是随机种子**（历史 42+idx，我们 0）。\n')
    md.append('\n### 2.1 种子噪声底（实测，不是估计）\n')
    md.append('\nA@35k 在同一 187 集上跑 seed0 与 seed1，两行都是官方输出：\n')
    md.append('\n| 指标 | seed0 | seed1 | 差值 | 判读门槛 |\n|---|---:|---:|---:|---|\n')
    md.append('| SSIM | 0.5850 | 0.5871 | +0.0021 | 差异 <0.002 不算差异 |\n')
    md.append('| LPIPS | 0.34969 | 0.34391 | −0.0058 | 差异 <0.006 不算差异 |\n')
    md.append('| Ink-IoU | 0.28735 | 0.28861 | +0.0013 | 差异 <0.002 不算差异 |\n')
    md.append('| 碎片比 | 3.6149 | 2.7800 | −0.83 | 该指标噪声极大，不可用于小差异 |\n')
    md.append('\n> 所以下面的跨纪元比较，SSIM 差 <0.002 的一律视为并列。\n')
    md.append('\n### 2.2 全表（同 187 集、同采样配置）\n')
    md.append('\n| 时代 | 模型 | 步 | 全集 SSIM | Top25% | Mid50% | Worst25% | 衰减率↓ | 级 |\n')
    md.append('|---|---|---:|---:|---:|---:|---:|---:|:-:|\n')
    E200 = [
        ('历史', 'v68 C2OT 旗舰', '200k', 0.6149, 0.7523, 0.6156, 0.4759, 0.2764, 'A'),
        ('历史', '48 DiT-B + v66route', '40k', 0.6104, '', '', '', '', 'B'),
        ('历史', 'v54 纯字表', '150k', 0.5960, 0.7126, 0.5986, 0.4743, 0.2383, 'B'),
        ('历史', 'v66 黄金路由', '150k', 0.5955, 0.7180, 0.5963, 0.4712, 0.2468, 'B'),
        ('历史', 'MoYi 12ch', '50k', 0.5931, 0.6726, 0.5935, 0.5129, 0.1598, 'A'),
        ('历史', 'v70 纯骨架', '30k', 0.5457, 0.6191, 0.5581, 0.4479, 0.1712, 'X'),
    ]
    n187 = {r['label']: r for r in csv.DictReader(open(LOCAL, encoding='utf-8-sig'))
            if r['block'] == 'n187'} if LOCAL.exists() else {}
    for lab in ('S9', 'S0@85k', 'A@35k', 'A@25k'):
        r = n187.get(lab)
        if r:
            E200.append(('本次', lab, r['step'], float(r['ssim']), '', '', '', '', 'A'))
    for era, name, steps, ssim, top, mid, worst, drop, grade in E200:
        f = lambda v: f'{v:.4f}' if v != '' else '—'
        md.append(f'| {era} | {name} | {steps} | {ssim:.4f} | {f(top)} | {f(mid)} | '
                  f'{f(worst)} | {f(drop)} | {grade} |\n')
    E200_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(E200_CSV, 'w', encoding='utf-8-sig', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['era', 'name', 'steps', 'ssim', 'top25', 'mid50', 'worst25',
                    'drop', 'grade'])
        for row in E200:
            w.writerow(row)
    print(f'wrote {E200_CSV} ({len(E200)} rows)')
    md.append('\n### 2.3 读法\n')
    md.append('\n- **v68 仍是八个月来的绝对天花板**（0.6149），S9@50k（0.6023）是历史第二，'
              '但比 v68 低 0.0126 —— 远超 0.002 的噪声底，是真实差距。\n')
    md.append('- **S9@50k 只用 50k 步就超过 v54/v66 的 150k**（0.6023 vs 0.5960/0.5955，差 +0.006，'
              '超噪声底）。这是本次工作最硬的一个历史性结论：style_matrix 的注入设计'
              '在同等数据下把收敛速度提到约 1/3。\n')
    md.append('- **MoYi 12ch 的抗跌性仍是全场第一**（衰减率 0.1598，Worst25% 0.5129 高于 v68 的 '
              '0.4759）。高难字必须靠显式几何硬通道这条结论没被推翻。\n')
    md.append('- **A 链（A@25k/35k/K）在这个标尺上偏低**（0.5795–0.5850），比 S9 低 0.02 左右。'
              'A 链是为了做 D 计划后训练才选的起点，不是内容最优分支。\n')
    md.append('- **K 完全没有 n187 评测**，因此进不了这张跨纪元表。要让它进榜，'
              '只需在 eval200fix 上跑一次（约 10 分钟）。这是目前最值得做的一个测量：'
              '它一次性把 K 放到八个月的历史坐标里。\n')
    md.append('\n## 3. val100 全排名（唯一同面板同实现的块）\n')
    md.append('\n### 3.1 按 SSIM 排序（含全部 S 组）\n\n')
    md.append('| # | 模型 | 步 | SSIM | LPIPS | Ink-IoU | 碎片比 |\n|---:|---|---:|---:|---:|---:|---:|\n')
    v = list(loc.values())
    v = [r for r in v if r.get('ssim')]
    v.sort(key=lambda r: -float(r['ssim']))
    for i, r in enumerate(v, 1):
        star = ' ★' if r['label'].startswith('K') else ''
        md.append(f'| {i} | {r["label"]}{star} | {r["step"]} | {r["ssim"]} | '
                  f'{r["lpips"] or "—"} | {r["iou"] or "—"} | {r["frag"] or "—"} |\n')

    md.append('\n### 3.2 按 OCR 排序（n=90）\n\n')
    md.append('| # | 模型 | OCR top1 | direction | 来源 |\n|---:|---|---:|---:|---|\n')
    # OCR lives in two places: the transcribed Milestones rows (S/P) and the
    # lineage CSV (A/K/D). Merge both or the ranking silently omits the S group.
    ocr_rows = []
    for r in out:
        if r['protocol'] != 'val100' or not r['ocr']:
            continue
        if not str(r['ocr']).replace('.', '').isdigit():
            continue
        ocr_rows.append((r['name'], float(r['ocr']), r['direction'] or '', 'Milestones'))
    seen = {n for n, _, _, _ in ocr_rows}
    for r in loc.values():
        if r.get('ocr') and r['label'] not in seen:
            ocr_rows.append((r['label'], float(r['ocr']), r['direction'] or '', 'eval_arm'))
    ocr_rows.sort(key=lambda t: -t[1])
    for i, (name, val, d, src) in enumerate(ocr_rows, 1):
        md.append(f'| {i} | {name} | {val:.2f} | {d or "—"} | {src} |\n')

    md.append('\n## 4. 全局结论\n')
    md.append('\n1. **没有任何一个节点同时具备全部指标。** 历史里程碑的书家主轴是 E-G AUC，'
              'S0 82.13 / S4 84.91 / S5 87.98 / P0 53.69 / P1 72.22 / P2 77.77 都有；'
              '**K 与 K+D1dec 没有**。我们新引入的 `direction` 反而只有 K/K+D1dec/D3 有，'
              'S 组没有。两个书家指标互不覆盖。\n')
    md.append('2. **在 val100 的内容轴上，S0 仍是全局最高**（SSIM 0.60275、LPIPS 0.33950），'
              'K+D1dec（0.5982/0.3416）追到 S 组中段，K（0.5946/0.3462）在 S 组之下。'
              'K 的优势不在内容，在于它是唯一有书家条件实测的节点。\n')
    md.append('3. **OCR 上 K 与 S4 并列 77.78，D3 略高 78.89**；S 组里 S0@50k 只有 71.11，'
              'S0@85k 到 81.29（187 面板，不可直接比）。所以「K 的 OCR 比 S0 好」这句话'
              '只在 50k 对 50k、val100 面板上成立。\n')
    md.append('4. **D 计划三臂（起点错）不构成有效的全局对照**：它们的起点是随机 style 分支，'
              '应标为 X 级阴性对照，不能当作「decoder-only / 联训 / hard-char」三种方法的结论。\n')
    md.append('5. **补三个缺口就能完全闭合全局比较**（都很便宜）：\n')
    md.append('   - **给 K 跑一次 eval200fix N=187**（约 10 分钟）→ 一次性把 K 放进八个月的历史坐标；\n')
    md.append('   - 给 **K / K+D1dec 测 E-G AUC** → 才能放进历史书家轴；\n')
    md.append('   - 给 **S0@50k / S4@50k 测 T3 direction**（各约 10 分钟）→ 才能放进我们的书家轴。\n')
    md.append('6. **跨纪元标尺上最硬的一条结论**：S9@50k（0.6023）是八个月里的历史第二，'
              '仅次于 v68（0.6149），而它只用了 **50k 步**就超过 v54/v66 的 150k（0.5960/0.5955，'
              '差 +0.006 已超 0.002 噪声底）。style_matrix 的注入设计在同等数据下把收敛速度'
              '提到约 1/3 —— 这是本次工作真正有历史意义的贡献，比 D 计划的 decoder 收益重要得多。\n')
    md.append('\n### 4.1 全局最优其实只剩两个候选\n')
    md.append('\n把三个内容指标和 val100 OCR 放在一起看，历史组里 **S4@50k** 是综合最好的一个：\n')
    md.append('\n| 节点 | SSIM | LPIPS | Ink-IoU | OCR | E-G AUC | direction |\n')
    md.append('|---|---:|---:|---:|---:|---:|---:|\n')
    md.append('| **S4@50k** | 0.5988（#4） | 0.3406（#2） | **0.2951（#1）** | 77.78（#3） | '
              '**84.91** | — 缺 |\n')
    md.append('| **K + D1dec** | 0.5982（#5） | 0.3416（#3） | 0.2893 | 77.78（#3） | — 缺 | '
              '**0.5571** |\n')
    md.append('\n两者内容指标几乎相同（SSIM 差 0.0006、LPIPS 差 0.0010、OCR 并列），'
              '**唯一的实质差别是各自缺另一条书家轴**：S4 有 E-G AUC 84.91 而没有 direction，'
              'K+D1dec 有 direction 0.5571 而没有 E-G AUC。\n')
    md.append('\n所以「K 是最优」和「S0 是最优」都不成立。真实状态是：**全局最优在 S4@50k 与 '
              'K+D1dec 之间未决**，而裁决它只需要补两个便宜测量（S4 的 T3、K 的 E-G AUC）。'
              '在此之前任何「谁最好」的断言都是缺列的。\n')
    md.append('\n对比之下，S0@50k 虽然 SSIM 第一，但 OCR 只有 71.11（#15/17），'
              '属于「好看但写不准」；P 链整段在 SSIM 与 OCR 上都低于 S 组，'
              'P0 更是只有 67.78。\n')
    md.append('\n## 5. 图表\n\n')
    md.append('| 图 | 内容 |\n|---|---|\n')
    md.append('| [fig_global_by_protocol.png](../../assets/metrics_index/fig_global_by_protocol.png) | '
              'SSIM 按协议分面（同列可比、跨列不可比），★ 为本次节点 |\n')
    md.append('| [fig_global_val100_ocr.png](../../assets/metrics_index/fig_global_val100_ocr.png) | '
              '唯一可横比的 OCR 列（val100 / n=90） |\n')
    md.append('| [fig_global_coverage.png](../../assets/metrics_index/fig_global_coverage.png) | '
              '指标覆盖矩阵：谁缺哪一列 |\n')
    md.append('| [fig_lineage_scorecard.png](../../assets/metrics_index/fig_lineage_scorecard.png) | '
              'val100 记分牌热图（列内排名） |\n')

    OUT_MD.write_text(''.join(md), encoding='utf-8', newline='\n')
    print(f'wrote {OUT_MD}')


if __name__ == '__main__':
    main()
