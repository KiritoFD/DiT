# -*- coding: utf-8 -*-
"""远程: 验证 s26 eval 是否冻结 (不同 step 的 json 除 step 外是否相同)."""
import os, json, glob, re, hashlib
os.chdir('/root/Workspace/xy/DiT')

js = sorted(glob.glob('assets/results/s26_ctrl_gt_skel/**/eval_auto_ctrl_*.json',
                      recursive=True),
            key=lambda p: int(re.search(r'(\d+)', os.path.basename(p)).group(1)))
print('eval json 数:', len(js))

sigs = {}
for p in js:
    d = json.load(open(p, encoding='utf-8'))
    step = d.pop('step', None)
    # 去掉 step 后算内容哈希
    h = hashlib.md5(json.dumps(d, sort_keys=True).encode()).hexdigest()[:12]
    sigs.setdefault(h, []).append(step)
    print(f'  {os.path.basename(p):32s} step={step:>6d} hash={h}')
    d['step'] = step

print('\n=== 去重后不同内容数:', len(sigs), '/', len(js))
for h, steps in sigs.items():
    print(f'  {h}: steps={steps}')
if len(sigs) == 1:
    print('\n!!! eval 结果完全冻结 — 所有评估返回相同数值, 评估无效 !!!')

# base vs ctrl 对比 (最新一次)
d = json.load(open(js[-1], encoding='utf-8'))
print('\n=== 最新评估 base vs ctrl ===')
print(f"  ssim      base={d.get('base.ssim'):.4f}  ctrl={d.get('ctrl.ssim'):.4f}  "
      f"delta={d.get('delta_ssim'):+.4f}")
print(f"  lpips     base={d.get('base.lpips'):.4f} ctrl={d.get('ctrl.lpips'):.4f}  "
      f"delta={d.get('delta_lpips'):+.4f}")
print(f"  mse       base={d.get('base.mse'):.4f}  ctrl={d.get('ctrl.mse'):.4f}  "
      f"delta={d.get('delta_mse'):+.4f}")
print(f"  skel_iou  base={d.get('base.skel_iou'):.4f} ctrl={d.get('ctrl.skel_iou'):.4f} "
      f" delta={d.get('delta_skel_iou'):+.4f}")

print('\n=== eval 样本可视化目录 ===')
for p in sorted(glob.glob('assets/results/s26_ctrl_gt_skel/**/eval_samples_ctrl/**/*',
                          recursive=True))[:15]:
    print('  ', p)
