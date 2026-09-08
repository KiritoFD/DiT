#!/bin/bash
cd /root/Workspace/xy/DiT
echo "=== docs/system 编号 44+ ==="
ls docs/system/ | grep -E "^4[4-9]|^5[0-9]" | sort
echo
echo "=== stdskel_gap.json ==="
cat 5script/results/stdskel_gap.json 2>/dev/null | head -50
echo
echo "=== Sp 模型定义 ==="
grep -n "DiT-2Cond-Sp\|def DiT_2Cond_Sp" src/model/dit.py | head -5
echo
echo "=== 各实验 eval 曲线 (ssim, 每10点抽1) ==="
/opt/conda/bin/python -c "
import json, glob
for d in ['v10b_stdskel_fame3','v10b_stdskel_fame3_c41x','v10b_stdskel_fame3_sp2','v10b_stdskel_fame3_deep','v10b_stdskel_fame3_d01']:
    files = sorted(glob.glob(f'5script/results/{d}/*/checkpoints/eval_auto_*.json'),
                   key=lambda p: int(p.split('eval_auto_')[1].split('.')[0]))
    if not files: print(f'{d}: no eval'); continue
    pts = []
    for f in files:
        try:
            j = json.load(open(f))
            s = j.get('ssim', j.get('ssim_mean'))
            st = int(f.split('eval_auto_')[1].split('.')[0])
            if s is not None: pts.append((st, round(s,4)))
        except Exception: pass
    best = max(pts, key=lambda p: p[1]) if pts else None
    print(f'{d}: n={len(pts)} best={best} last3={pts[-3:]}')"
echo
echo "=== 遵循度相关 json (skel_iou / follow) ==="
/opt/conda/bin/python -c "
import json, glob
for d in ['v10b_stdskel_fame3','v10b_stdskel_fame3_c41x']:
    files = sorted(glob.glob(f'5script/results/{d}/*/checkpoints/eval_auto_*.json'))
    if not files: continue
    f = files[-1]
    j = json.load(open(f))
    keys = [k for k in j if 'iou' in k.lower() or 'follow' in k.lower() or 'lpips' in k.lower()]
    print(d, 'last:', f.split('eval_auto_')[1], {k: round(j[k],4) if isinstance(j[k], float) else j[k] for k in keys[:6]})"