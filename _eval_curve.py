import json, glob, os
os.chdir("/root/Workspace/xy/DiT")
for tag, dirname in [("s15", "s15_ws_flow"), ("s17", "s17_s_flow")]:
    fs = sorted(glob.glob(f"5script/results/{dirname}/*/checkpoints/eval_auto_*.json"))
    if not fs:
        print(f"=== {tag} ({dirname}) === no eval files")
        continue
    print(f"=== {tag} ({dirname}) === {len(fs)} evals")
    # print every 5th eval + last
    for i, f in enumerate(fs):
        step = int(os.path.basename(f).replace("eval_auto_","").replace(".json",""))
        if step % 25000 == 0 or i == len(fs)-1 or step <= 10000:
            d = json.load(open(f))
            print(f"  step={step:6d}  SSIM={d['ssim']:.4f}  LPIPS={d['lpips']:.4f}  SkelIoU={d['skel_iou']:.4f}  MSE={d['mse']:.4f}")
