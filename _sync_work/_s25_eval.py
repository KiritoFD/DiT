# -*- coding: utf-8 -*-
"""远程: s25-ids 基模 eval 趋势 + 与 s21 基模对比."""
import os, json, glob, re
os.chdir('/root/Workspace/xy/DiT')


def load_group(root, pat='eval_auto_*.json'):
    js = sorted(glob.glob(os.path.join(root, pat)),
                key=lambda p: int(re.search(r'(\d+)', os.path.basename(p)).group(1))
                if re.search(r'(\d+)', os.path.basename(p)) else 0)
    rows = []
    for p in js:
        try:
            rows.append(json.load(open(p, encoding='utf-8')))
        except Exception as e:
            pass
    return rows


def pick(d, key):
    """兼容: 平铺 'ssim' / 'base.ssim' / 嵌套 d['base']['ssim']."""
    v = d.get(key)
    if v is None:
        v = d.get('base.' + key)
    if v is None:
        g = d.get('base')
        if isinstance(g, dict):
            v = g.get(key)
    return v if isinstance(v, (int, float)) else None


def report(title, rows):
    print(f'\n{"="*56}\n{title}  (纯基模)\n{"="*56}')
    if not rows:
        print('  (无 eval)')
        return
    print(f"  {'step':>7s}{'ssim':>9s}{'iou':>9s}{'lpips':>9s}{'mse':>9s}")
    for d in rows:
        s = d.get('step', -1)
        print(f"{s:>7d}"
              f"{(pick(d,'ssim') or 0):>9.4f}"
              f"{(pick(d,'skel_iou') or 0):>9.4f}"
              f"{(pick(d,'lpips') or 0):>9.4f}"
              f"{(pick(d,'mse') or 0):>9.4f}")
    last = rows[-1]
    print(f"\n  末次: ssim={pick(last,'ssim'):.4f} iou={pick(last,'skel_iou'):.4f} "
          f"lpips={pick(last,'lpips'):.4f} mse={pick(last,'mse'):.4f}")


def last_vals(rows):
    if not rows:
        return {}
    last = rows[-1]
    return {k: pick(last, k) for k in ['ssim', 'skel_iou', 'lpips', 'mse']}


s25 = load_group('assets/results/s25_ids_pretrain/*/checkpoints')
report('s25-ids 基模 (8-31, main_ckpt for s26)', s25)
s21 = load_group('assets/results/s21_fame_flow_v2/*/checkpoints')
report('s21-fame-flow-v2 基模 (8-29, 好ctrl的基模)', s21)

v21, v25 = last_vals(s21), last_vals(s25)
if v21 and v25:
    print('\n' + '='*56)
    print('基模末次对比 (纯基模能力)')
    print('='*56)
    for k in ['ssim', 'skel_iou', 'lpips', 'mse']:
        a, b = v21.get(k), v25.get(k)
        if a is None or b is None:
            print(f'  {k}: s21无/s25无 跳过')
            continue
        up = (b > a) if k in ('ssim', 'skel_iou') else (b < a)
        print(f'  {k:>10s}: s21={a:.4f}  s25={b:.4f}  -> {"s25提升" if up else "s25变差"} '
              f'({(b-a)/a*100:+.1f}%)')
