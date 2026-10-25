import json

with open("/root/Workspace/xy/DiT/scratch/v90_20l_gpu_20261017_01/assets/capacity_audit_20261017/complexity_stratification.json", encoding="utf-8") as f:
    d = json.load(f)

for m, res in d.items():
    ov = res["overall"]
    print(f"=== Model: {m} ===")
    print(f"  Overall: SSIM={ov['ssim']:.4f}, Ink-IoU={ov['ink_iou']:.4f}, Recall={ov['thin_skeleton_recall']:.4f}, Merged={ov['merged_components']:.2f}, Missing={ov['missing_components']:.2f}")
    for tier, sub in res["by_tier"].items():
        print(f"    [{tier}] SSIM={sub['ssim']:.4f}, Recall={sub['thin_skeleton_recall']:.4f}, Merged={sub['merged_components']:.2f}, Missing={sub['missing_components']:.2f}, IoU={sub['ink_iou']:.4f}")
