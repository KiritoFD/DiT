"""Compare eval curves: s11 (top6, 195k steps) vs s12 (3top30, overfit?).
Prints SSIM/skel_iou at each eval step for the LAST run of each series."""
import json, glob, os, sys
sys.stdout.reconfigure(encoding="utf-8")

def curve(series):
    files = sorted(glob.glob(f'/root/Workspace/xy/DiT/5script/results/{series}/*/checkpoints/eval_auto_*.json'))
    if not files:
        print(f"  {series}: no eval files")
        return
    # group by run dir, pick last run
    runs = {}
    for f in files:
        run = os.path.dirname(os.path.dirname(f))
        runs.setdefault(run, []).append(f)
    last_run = sorted(runs.keys())[-1]
    fs = sorted(runs[last_run])
    print(f"  {series}: {os.path.basename(last_run)} ({len(fs)} evals)")
    for f in fs:
        try:
            d = json.load(open(f))
            ssim = d.get('ssim', d.get('mse', -1))
            skel = d.get('skel_iou', d.get('skeleton_iou', d.get('skel', -1)))
            step = os.path.basename(f).replace('eval_auto_', '').replace('.json', '')
            print(f"    step={step}: ssim={ssim:.4f} skel={skel:.4f}")
        except Exception as e:
            print(f"    {os.path.basename(f)}: ERR {e}")

print("=== s11 top6 (10k imgs, 195k steps) ===")
curve('s11_top6_p4')
print("=== s12 3top30 (40k imgs, 600k steps) ===")
curve('s12_3top30_dino')
