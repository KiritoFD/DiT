import json, glob, os
os.chdir("/root/Workspace/xy/DiT")
tags = {"s14": "s14_ws_ddpm", "s17": "s17_s_flow", "s15": "s15_ws_flow", "s16": "s16_s_ddpm"}
print("===== 1K-STEP TEST RESULTS =====")
print(f"{'tag':<6} {'model':<16} {'diff':<6} {'MSE':>8} {'SSIM':>8} {'SkelIoU':>8} {'LPIPS':>8}")
for tag, dirname in tags.items():
    fs = glob.glob(f"5script/results/{dirname}/*/checkpoints/eval_auto_0001000.json")
    if not fs:
        print(f"{tag:<6} (not found)")
        continue
    d = json.load(open(fs[0]))
    model = "WS/2" if "ws" in dirname else "S/2"
    diff = "flow" if "flow" in dirname else "ddpm"
    print(f"{tag:<6} {model:<16} {diff:<6} {d['mse']:>8.4f} {d['ssim']:>8.4f} {d['skel_iou']:>8.4f} {d['lpips']:>8.4f}")
print("===== END =====")
