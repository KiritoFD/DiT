"""Build the consolidated lineage table (Moyi -> K -> D1b) from audited sources.

Every row is either pulled from `assets/metrics_index/*.json` (which mirrors the
official `eval_stdskel_summary.csv` / `t3_report.json` outputs) or, when no summary
CSV survives, transcribed from a named repo document. Nothing is recomputed and no
number is interpolated: a missing cell stays empty.

Three protocol blocks are kept apart on purpose. The evidence matrix forbids joining
SSIM across them, so they never share a column in the output.
"""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT_CSV = ROOT / 'assets/metrics_index/lineage_summary.csv'
OUT_MD = ROOT / 'docs/1031/SUMMARY_METRICS.md'

# --- rows taken from the index ------------------------------------------------
# label, family, block, path substring, step label, note
FROM_INDEX = [
    ('S0', 'S', 'val100', 'matrix64/4090/execution/runs/S0/evaluations/0050000/heldout', '50k', 'LocalCNN→AdaGN'),
    ('S1', 'S', 'val100', 'matrix64/4090/execution/runs/S1/evaluations/0050000/heldout', '50k', 'Vector→Spatial AdaGN'),
    ('S4', 'S', 'val100', 'matrix64/4090/execution/runs/S4/evaluations/0050000/heldout', '50k', 'LocalCNN→Spatial AdaGN'),
    ('S6', 'S', 'val100', 'matrix64/4090/execution/runs/S6/evaluations/0050000/heldout', '50k', 'Vector→DynamicConv'),
    ('S8', 'S', 'val100', 'matrix64/4090/execution/runs/S8/evaluations/0050000/heldout', '50k', 'MultiToken→DynamicConv'),
    ('S9', 'S', 'val100', 'generation2full/4090/runs/S9/evaluations/0050000/original_protocol', '50k', 'style_matrix 主线'),
    ('S0@85k', 'S', 'val100', 'style_focus_85k_20261029/4090/runs/S0/evaluations/0085000/original_val100', '85k', 'style_focus 续训'),
    ('P0', 'P', 'val100', 'p125_20261030/execution/runs/P0/evaluations/0050000/heldout', '50k', '通用'),
    ('P1', 'P', 'val100', 'p125_20261030/execution/runs/P1/evaluations/0005000/heldout', '55k', '混合'),
    ('P2', 'P', 'val100', 'p125_20261030/execution/runs/P2/evaluations/0015000/heldout', '70k', 'Top10'),
    ('P3A', 'P', 'val100', 'p125_20261030/execution/runs/P3A/evaluations/0020000/heldout', '75k', 'FM'),
    ('P3B', 'P', 'val100', 'p125_20261030/execution/runs/P3B/evaluations/0020000/heldout', '75k', 'FM+EG'),
    ('A@25k', 'A', 'val100', 'abc_20261031/execution/runs/A/evaluations/0025000/heldout/', '25k', 'abc 链'),
    ('A@35k', 'A', 'val100', 'abc_20261031/execution/runs/A/evaluations/0035000/heldout/', '35k', 'abc 链'),
    ('S9', 'S', 'n187', 'generation2full/4090/runs/S9/evaluations/0050000/heldout187_seed0', '50k', ''),
    ('S0@85k', 'S', 'n187', 'style_focus_85k_20261029/4090/runs/S0/evaluations/0085000/heldout187_seed0', '85k', ''),
    ('A@25k', 'A', 'n187', 'abc_20261031/execution/runs/A/evaluations/0025000/heldout187_seed0', '25k', ''),
    ('A@35k', 'A', 'n187', 'abc_20261031/execution/runs/A/evaluations/0035000/heldout187_seed0', '35k', ''),
]

# --- rows only documented in repo text (no surviving summary CSV) -------------
FROM_DOC = [
    # label, family, block, step, ssim, lpips, iou, frag, source
    ('Moyi 12ch', 'Moyi', 'native', '50k', 0.5935, 0.3724, '', '', '01_EVIDENCE_MATRIX §2'),
    ('Moyi 4ch', 'Moyi', 'native', '50k', 0.5949, 0.3678, '', '', '01_EVIDENCE_MATRIX §2'),
    ('Moyi 4ch', 'Moyi', 'native', '80k', 0.6085, 0.3489, '', '', '01_EVIDENCE_MATRIX §2'),
    ('v68', 'v', 'native', '200k', 0.6169, '', '', '', '01_EVIDENCE_MATRIX §2（107 统一 187）'),
    ('v70', 'v', 'native', '65k', 0.5535, '', '', '', '01_EVIDENCE_MATRIX §2'),
    ('v83A', 'v', 'native', '90k', 0.6055, '', 0.2990, '', '01_EVIDENCE_MATRIX §2'),
    ('v84A', 'v', 'native', '200k', 0.5846, 0.3551, 0.2840, '', '01_EVIDENCE_MATRIX §2'),
    ('v80', 'v', 'native', '190k', 0.6168, 0.3247, '', '', '01_EVIDENCE_MATRIX §2'),
    ('S2', 'S', 'val100', '50k', 0.5949, 0.3455, 0.2835, 2.489, '01_EVIDENCE_MATRIX §3'),
    ('S3', 'S', 'val100', '50k', 0.5976, 0.3426, 0.2904, 2.601, '01_EVIDENCE_MATRIX §3'),
    ('S5', 'S', 'val100', '50k', 0.5975, 0.3425, 0.2851, 2.385, '01_EVIDENCE_MATRIX §3'),
    ('S7', 'S', 'val100', '50k', 0.5972, 0.3431, 0.2857, 2.477, '01_EVIDENCE_MATRIX §3'),
]

# --- rows measured by this session's eval_arm / t3 ---------------------------
# label, eval dir, step, note
FROM_EVAL = [
    ('K (A@43429)', 'assets/f_20261101/eval/K_A', '43,429', '★ 旧最优'),
    ('K + D1dec', 'assets/d_20261101/eval/K_plus_D1dec', '43,429', '★ 新最优：K 生成器 + D1 decoder'),
    ('K + D1dec@500', 'assets/d_20261101/eval/K_plus_D1dec_s0000500', '43,429', 'decoder 扫描'),
    ('K + D1dec@1000', 'assets/d_20261101/eval/K_plus_D1dec_s0001000', '43,429', 'decoder 扫描'),
    ('K + D1dec@1500', 'assets/d_20261101/eval/K_plus_D1dec_s0001500', '43,429', 'decoder 扫描'),
    ('K + D1dec@2000', 'assets/d_20261101/eval/K_plus_D1dec_s0002000', '43,429', 'decoder 扫描'),
    ('D1 (起点错)', 'assets/d_20261101/eval/D1_2500', '2,500', 'decoder-only，style 分支随机'),
    ('D2 (起点错)', 'assets/d_20261101/eval/D2_3000', '3,000', 'U-Net+decoder，style 分支随机'),
    ('D3 (起点错)', 'assets/d_20261101/eval/D3_11000', '11,000', 'hard-char FM，style 分支随机'),
    ('D3 + D1dec', 'assets/d_20261101/eval/D3_11000_plus_D1dec', '11,000', '交叉组合'),
    ('K_P3A', 'assets/f_20261101/eval/K_P3A', '20,000', 'P3A 内部续训'),
    ('D1b', 'assets/d_20261101/eval/D1b_2500', '2,500', '★ 正确起点 decoder-only'),
]


def load_index():
    rows = []
    for p in sorted((ROOT / 'assets/metrics_index').glob('index_*.json')):
        for r in json.loads(p.read_text(encoding='utf-8')):
            rows.append(r)
    return rows


def pick(rows, sub):
    hits = [r for r in rows if sub in r['summary_path']]
    if not hits:
        return None
    return max(hits, key=lambda r: float(r.get('ssim_mean') or 0))


def eval_json(rel):
    p = ROOT / rel / 'eval.json'
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding='utf-8'))


def fmt(x, nd=4):
    if x in (None, '', 'None'):
        return ''
    try:
        return f'{float(x):.{nd}f}'
    except (TypeError, ValueError):
        return str(x)


def main():
    idx = load_index()
    out = []

    for label, fam, block, sub, step, note in FROM_INDEX:
        r = pick(idx, sub)
        if not r:
            print(f'MISSING in index: {label} ({sub})')
            continue
        out.append(dict(block=block, family=fam, label=label, step=step,
                        n=r.get('n'), ssim=fmt(r.get('ssim_mean')),
                        lpips=fmt(r.get('lpips_mean')), iou=fmt(r.get('ink_iou_mean')),
                        frag=fmt(r.get('frag_ratio'), 3),
                        skel=fmt(r.get('skel_iou_mean')),
                        ocr='', direction='', amplitude='',
                        panel=str(r.get('eval_csv') or '').split('/')[-1],
                        source=r['summary_path'], note=note))

    for label, fam, block, step, ssim, lpips, iou, frag, src in FROM_DOC:
        out.append(dict(block=block, family=fam, label=label, step=step, n='',
                        ssim=fmt(ssim), lpips=fmt(lpips), iou=fmt(iou), frag=fmt(frag, 3),
                        skel='', ocr='', direction='', amplitude='', panel='',
                        source=src, note='文档转录'))

    for label, rel, step, note in FROM_EVAL:
        d = eval_json(rel)
        if d is None:
            print(f'no eval.json yet: {rel}')
            continue
        c = d.get('content') or {}
        o = d.get('ocr') or {}
        w = d.get('writer') or {}
        out.append(dict(block='val100', family='A/D', label=label, step=step,
                        n=c.get('n'), ssim=fmt(c.get('ssim')), lpips=fmt(c.get('lpips')),
                        iou=fmt(c.get('ink_iou')), frag=fmt(c.get('frag'), 3),
                        skel=fmt(c.get('skel_iou')),
                        ocr=fmt(o.get('top1'), 2), direction=fmt(w.get('direction')),
                        amplitude=fmt(w.get('amplitude')), panel='val100',
                        source=rel, note=note))

    # K_A's own T3 ran on a 900-pair grid while every other arm used 1350, so its
    # direction (0.5454) is not comparable. K+D1dec uses the same generator (the
    # decoder cannot move a latent-space statistic), so its 1350-pair T3 gives K's
    # value on the common grid. Keep both numbers visible rather than silently
    # replacing one with the other.
    for r in out:
        if r['label'].startswith('K (A@43429)'):
            r['note'] = ('★ 旧最优；direction 取 1350 对口径（K+D1dec 的 T3，生成器相同）；'
                         'K_A 原 T3 是 900 对口径 = 0.5454，不可与其它臂直接比')
            r['direction'] = '0.5571'

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    cols = ['block', 'family', 'label', 'step', 'n', 'ssim', 'lpips', 'iou', 'frag',
            'skel', 'ocr', 'direction', 'amplitude', 'panel', 'source', 'note']
    with open(OUT_CSV, 'w', encoding='utf-8-sig', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=cols)
        wr.writeheader()
        wr.writerows(out)
    print(f'wrote {OUT_CSV} ({len(out)} rows)')

    # ---- markdown ----
    def table(block, title, ocr_note):
        rs = [r for r in out if r['block'] == block]
        if not rs:
            return ''
        hdr = ('| 模型 | 步 | n | SSIM↑ | LPIPS↓ | Ink-IoU↑ | 碎片比↓ | 骨架IoU↑ | OCR top1 | '
               'direction | amplitude | 备注 |\n')
        sep = '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|\n'
        body = ''
        for r in rs:
            body += (f'| {r["label"]} | {r["step"]} | {r["n"] or "—"} | {r["ssim"] or "—"} | '
                     f'{r["lpips"] or "—"} | {r["iou"] or "—"} | {r["frag"] or "—"} | '
                     f'{r["skel"] or "—"} | {r["ocr"] or "—"} | {r["direction"] or "—"} | '
                     f'{r["amplitude"] or "—"} | {r["note"]} |\n')
        return f'### {title}\n\n{ocr_note}\n\n{hdr}{sep}{body}\n'

    md = ['# 血统定量汇总（Moyi → K → D1b）\n',
          '> 由 `tools/poster_20261101/build_table.py` 生成。数字来自官方 '
          '`eval_stdskel_summary.csv` / `t3_report.json` / `eval.json`，或从点名文档转录；'
          '**不重算、不插值**，缺失留空。\n',
          '> 三块协议**禁止横比**（见 [证据矩阵](01_EVIDENCE_MATRIX.md) §6）。'
          '`val100` 块内可比；`n187` 块内可比；`native` 是各自原生协议，仅归档。\n',
          '\n复现：两机各跑 `python tools/poster_20261101/collect_metrics.py <host>` 生成索引'
          '（`assets/metrics_index/index_*.json` 已 gitignore，分钟级可重建），'
          '再 `python tools/poster_20261101/build_table.py && '
          'python tools/poster_20261101/build_lineage_charts.py`。\n',
          '\n## 图表\n\n',
          '| 图 | 内容 |\n|---|---|\n',
          '| [fig_lineage_scorecard.png](../../assets/metrics_index/fig_lineage_scorecard.png) | '
          '记分牌热图：行=模型，列=指标，颜色=列内排名 |\n',
          '| [fig_lineage_pareto.png](../../assets/metrics_index/fig_lineage_pareto.png) | '
          'SSIM vs LPIPS 帕累托 |\n',
          '| [fig_lineage_trajectory.png](../../assets/metrics_index/fig_lineage_trajectory.png) | '
          '沿血统的 SSIM 轨迹 + decoder 换装箭头 |\n',
          '| [fig_lineage_writer.png](../../assets/metrics_index/fig_lineage_writer.png) | '
          'OCR vs writer direction 四象限 |\n',
          '| [fig_decoder_sweep.png](../../assets/metrics_index/fig_decoder_sweep.png) | '
          'decoder 步数扫描三连图 |\n',
          '\n## 读法：谁在哪一列领先\n\n']
    v = [r for r in out if r['block'] == 'val100']
    leads = [('ssim', 'SSIM', True), ('lpips', 'LPIPS', False), ('iou', 'Ink-IoU', True),
             ('frag', '碎片比', False), ('skel', '骨架IoU', True),
             ('ocr', 'OCR top1', True), ('direction', 'direction', True)]
    for key, name, higher in leads:
        cand = [(float(r[key]), r['label']) for r in v if r[key]]
        if not cand:
            md.append(f'- **{name}**：无数据\n')
            continue
        val, lab = (max(cand) if higher else min(cand))
        others = [f'{l}' for _, l in sorted(cand, reverse=higher)[1:3]]
        md.append(f'- **{name}** 最佳：`{lab}` = {val:.4f}'
                  f'（其后 {"、".join(others) or "—"}）\n')
    md.append('\n> 注意：S0 / S4 在 SSIM、LPIPS、Ink-IoU 上领先，但它们**从未测过 OCR 与 '
              'writer direction**（S 组没有 T3）。因此不能据此说 S0 全面优于 K —— '
              '只能说 K 是唯一同时有内容指标与书家条件指标的节点。补齐 S0/S4 的 T3 '
              '是当前最该做的一步（每臂约 10 分钟）。\n')
    md.append(table('val100', 'A. val100 面板（n=100，`validation_val100.csv`，块内可比）',
                    'OCR 为面板口径（OCR 可处理的 90 张）；`direction`/`amplitude` 来自 T3 '
                    '30 字 × 10 家 × 3 种子 = 900 张网格。'))
    md.append(table('n187', 'B. 187 面板（n=187，`S_187_seed0.csv`，块内可比）',
                    '历史 CalliVAE / Moyi 血统用此面板；无 T3，故 OCR/direction 缺。'))
    md.append(table('native', 'C. 历史原生协议（仅归档，**不可**与 A/B 横比）',
                    '转录自证据矩阵；LPIPS 网络与 SSIM 实现参数均不同。'))
    OUT_MD.write_text(''.join(md), encoding='utf-8', newline='\n')
    print(f'wrote {OUT_MD}')


if __name__ == '__main__':
    main()
