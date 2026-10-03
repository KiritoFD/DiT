# -*- coding: utf-8 -*-
"""远程: s26 eval 真实趋势 (兼容平铺/嵌套两种 json 格式).
base = s25 基模(不训练, 恒定正常); ctrl = 加 ControlNet;  delta = ctrl - base
ssim/iou 正=变好, lpips 负=变好
"""
import os, json, glob, re
os.chdir('/root/Workspace/xy/DiT')

js = sorted(glob.glob('assets/results/s26_ctrl_gt_skel/**/eval_auto_ctrl_*.json',
                      recursive=True),
            key=lambda p: int(re.search(r'(\d+)', os.path.basename(p)).group(1)))
rows = [json.load(open(p, encoding='utf-8')) for p in js]
print('eval json 数:', len(js))


def pick(d, group, key):
    """兼容两种格式: 平铺 'base.ssim' 或 嵌套 d['base']['ssim']."""
    v = d.get(f'{group}.{key}')
    if v is None:
        g = d.get(group)
        if isinstance(g, dict):
            v = g.get(key)
    return v


def num(v):
    return v if isinstance(v, (int, float)) else None


KEYS = ['ssim', 'skel_iou', 'lpips', 'mse']
print('\n' + '=' * 100)
hdr = f"{'step':>7s}"
for k in KEYS:
    hdr += f"{'b_'+k:>11s}{'c_'+k:>11s}{'d_'+k:>11s}"
print(hdr)
print('=' * 100)
for d in rows:
    s = d.get('step', -1)
    line = f"{s:>7d}"
    for k in KEYS:
        b = num(pick(d, 'base', k))
        c = num(pick(d, 'ctrl', k))
        dk = num(d.get(f'delta_{k}'))
        if dk is None and b is not None and c is not None:
            dk = c - b
        line += (f"{b:>11.4f}" if b is not None else f"{'-':>11s}")
        line += (f"{c:>11.4f}" if c is not None else f"{'-':>11s}")
        line += (f"{dk:>+11.4f}" if dk is not None else f"{'-':>11s}")
    print(line)
print('=' * 100)

print('\n=== 判读 ===')
for k in KEYS:
    b0 = num(pick(rows[0], 'base', k))
    bN = num(pick(rows[-1], 'base', k))
    print(f'  base.{k}: 首={b0} 末={bN}  '
          f'{"恒定(正常: 基模不训练)" if b0 == bN else "有变化(!)"}')

# ctrl 趋势
for k in KEYS:
    cs = [num(pick(d, "ctrl", k)) for d in rows]
    cs = [x for x in cs if x is not None]
    if cs:
        print(f'  ctrl.{k}: 首={cs[0]:.4f} 末={cs[-1]:.4f} '
              f'{"↑改善" if cs[-1] > cs[0] else "↓下降"}'
              f' (最好={max(cs):.4f})' if k != 'lpips' and k != 'mse' else
              f'  ctrl.{k}: 首={cs[0]:.4f} 末={cs[-1]:.4f} '
              f'{"↓改善" if cs[-1] < cs[0] else "↑变差"} (最好={min(cs):.4f})')

# delta
for k in KEYS:
    ds = [num(d.get(f"delta_{k}")) for d in rows]
    ds = [x for x in ds if x is not None]
    if ds:
        good = max(ds) if k in ("ssim", "skel_iou") else min(ds)
        idx = ds.index(good)
        print(f'  delta_{k}: 首={ds[0]:+.4f} 末={ds[-1]:+.4f} '
              f'最好={good:+.4f} @step={rows[idx].get("step")}')
