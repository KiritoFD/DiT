# -*- coding: utf-8 -*-
"""远程: 报告 s26 (1px GT skel ctrl) 当前性能 — 训练 loss + 定期 eval 指标趋势."""
import os, json, glob, re
os.chdir('/root/Workspace/xy/DiT')

# --- 1. 训练 loss 趋势 ---
L = 'assets/results/s26_ctrl_gt_skel/s26_tmux.log'
if os.path.isfile(L):
    lines = open(L, errors='replace').read().splitlines()
    # 去 ANSI
    ansi = re.compile(r'\x1b\[[0-9;]*m')
    hits = []
    for ln in lines:
        ln2 = ansi.sub('', ln)
        m = re.search(r'\(step=(\d+)\)\s+loss=([\d.]+)', ln2)
        if m:
            hits.append((int(m.group(1)), float(m.group(2))))
    print(f'[train] {L}: {len(hits)} 条记录')
    if hits:
        print(f'  最新 step={hits[-1][0]}  loss={hits[-1][1]}')
        # 分段均值
        if len(hits) > 10:
            n = len(hits)
            for lab, lo, hi in (('前25%', 0, n//4), ('25-50%', n//4, n//2),
                                ('50-75%', n//2, 3*n//4), ('后25%', 3*n//4, n)):
                seg = hits[lo:hi]
                if seg:
                    print(f'  {lab:<8s} step {seg[0][0]:>6d}-{seg[-1][0]:>6d}  '
                          f'loss 均值 {sum(x[1] for x in seg)/len(seg):.4f}')
        last20 = [x[1] for x in hits[-20:]]
        print(f'  最近20点 loss: 均值 {sum(last20)/len(last20):.4f}  '
              f'min {min(last20):.4f}  max {max(last20):.4f}')

# --- 2. eval 指标趋势 ---
print('\n[eval] 定期评估指标:')
js = sorted(glob.glob('assets/results/s26_ctrl_gt_skel/**/eval_auto_ctrl_*.json',
                      recursive=True),
            key=lambda p: int(re.search(r'(\d+)', os.path.basename(p)).group(1)))
if not js:
    print('  无 eval json')
else:
    print(f'  共 {len(js)} 次评估')
    rows = []
    for p in js:
        step = int(re.search(r'(\d+)', os.path.basename(p)).group(1))
        try:
            d = json.load(open(p, encoding='utf-8'))
        except Exception as e:
            print(f'  {os.path.basename(p)} 解析失败: {e}')
            continue
        rows.append((step, d))
    # 打印最后一次的完整结构
    s_last, d_last = rows[-1]
    print(f'\n  --- 最新评估 step={s_last} 的字段 ---')
    def walk(o, pre=''):
        if isinstance(o, dict):
            for k, v in o.items():
                if isinstance(v, (dict, list)):
                    walk(v, pre + k + '.')
                else:
                    print(f'    {pre}{k} = {v}')
        elif isinstance(o, list):
            print(f'    {pre}[list len={len(o)}]')
    walk(d_last)

    # 尝试提 metric
    def find_num(d, key):
        if isinstance(d, dict):
            for k, v in d.items():
                if key.lower() in k.lower() and isinstance(v, (int, float)):
                    return v
                r = find_num(v, key)
                if r is not None:
                    return r
        return None
    print('\n  --- 趋势 (ssim / psnr / fid / lpips 若存在) ---')
    keys = ['ssim', 'psnr', 'fid', 'lpips', 'l1', 'mse']
    hdr = f'  {"step":>8s}' + ''.join(f'{k:>10s}' for k in keys)
    print(hdr)
    for step, d in rows:
        line = f'  {step:8d}'
        for k in keys:
            v = find_num(d, k)
            line += f'{v:10.4f}' if v is not None else f'{"-":>10s}'
        print(line)
