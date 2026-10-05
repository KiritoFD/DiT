import csv
import glob
import json
import os

moyi_dir = "docs/106moyi"
out_csv = os.path.join(moyi_dir, "moyi_experiments_summary.csv")

print("=" * 80)
print(
    f"Rebuilding {out_csv} with correct Moyi 12ch baseline as Row 1..."
)
print("=" * 80)

# Canonical benchmark records with strict ordering
benchmark_data = [
    {
        "rank": 1,
        "experiment": "Moyi_12ch_50k (Baseline)",
        "model_type": "Moyun-12channel-B (Rectified Flow)",
        "params": "252.0M",
        "channels": "12ch (4ch img + 4ch canny + 4ch skel)",
        "step": 50000,
        "strict_ssim": 0.5935,
        "strict_ssim_med": 0.5950,
        "strict_lpips": 0.3724,
        "strict_skel_iou": 0.0177,
        "strict_mse": 0.7983,
        "seen_ssim": "-",
        "gap_ssim": "-",
        "status": "官方参考基线：多模态联合12通道输入，被4ch证伪",
        "source_file": "moyi/results/moyi_top10_rf/eval_ours200fix/eval_auto.json",
    },
    {
        "rank": 2,
        "experiment": "Moyi_4ch_50k",
        "model_type": "Moyun-4channel-B (Rectified Flow)",
        "params": "252.0M",
        "channels": "4ch (SD-VAE latent)",
        "step": 50000,
        "strict_ssim": 0.5949,
        "strict_ssim_med": 0.5990,
        "strict_lpips": 0.3678,
        "strict_skel_iou": 0.0161,
        "strict_mse": 0.8106,
        "seen_ssim": "-",
        "gap_ssim": "-",
        "status": "纯真迹4通道同频对照，直接超越12ch (+0.0014)",
        "source_file": "moyi/results/moyi_top10_rf_4ch/eval_step50000/eval_metrics.json",
    },
    {
        "rank": 3,
        "experiment": "Moyi_4ch_80k (终局峰值)",
        "model_type": "Moyun-4channel-B (Rectified Flow)",
        "params": "252.0M",
        "channels": "4ch (SD-VAE latent)",
        "step": 80000,
        "strict_ssim": 0.6085,
        "strict_ssim_med": 0.6123,
        "strict_lpips": 0.3489,
        "strict_skel_iou": 0.0235,
        "strict_mse": 0.7604,
        "seen_ssim": "-",
        "gap_ssim": "-",
        "status": "历史工业纪录最高峰，首次突破0.60大关",
        "source_file": "moyi/results/moyi_top10_rf_4ch/eval_step80000/eval_metrics.json",
    },
    {
        "rank": 4,
        "experiment": "Phase_A_Exp1_RMSNorm",
        "model_type": "DiT-2Cond-S/2 (RMSNorm+SwiGLU)",
        "params": "33.8M",
        "channels": "4ch (SD-VAE latent)",
        "step": 10000,
        "strict_ssim": 0.5408,
        "strict_ssim_med": 0.5393,
        "strict_lpips": 0.3926,
        "strict_skel_iou": 0.0140,
        "strict_mse": 0.9790,
        "seen_ssim": 0.5422,
        "gap_ssim": "+0.0014",
        "status": "微观消融证伪：33M容量瓶颈严重锁死生成上限",
        "source_file": "DiT/experiments/ablation_phase_a/results/exp1_rmsnorm/eval_10k/metrics_40k.json",
    },
    {
        "rank": 5,
        "experiment": "Capacity_Tier2_Sp_10k",
        "model_type": "DiT-2Cond-Sp/2 (LayerNorm+GELU)",
        "params": "59.2M",
        "channels": "4ch (SD-VAE latent)",
        "step": 10000,
        "strict_ssim": 0.5494,
        "strict_ssim_med": 0.5512,
        "strict_lpips": 0.3848,
        "strict_skel_iou": 0.0150,
        "strict_mse": 0.9530,
        "seen_ssim": 0.5444,
        "gap_ssim": "-0.0050",
        "status": "容量单调提升：较 S/2 @ 10k 同步提升 +0.0086",
        "source_file": "DiT/experiments/capacity_ladder/results/tier2_sp/evaluations/eval_10k/metrics_40k.json",
    },
    {
        "rank": 6,
        "experiment": "Capacity_Tier2_Sp_20k",
        "model_type": "DiT-2Cond-Sp/2 (LayerNorm+GELU)",
        "params": "59.2M",
        "channels": "4ch (SD-VAE latent)",
        "step": 20000,
        "strict_ssim": 0.5606,
        "strict_ssim_med": 0.5622,
        "strict_lpips": 0.3747,
        "strict_skel_iou": 0.0158,
        "strict_mse": 0.9052,
        "seen_ssim": 0.5539,
        "gap_ssim": "-0.0067",
        "status": "Sp/2 终测完训，稳步收敛拉开与 33M 差距",
        "source_file": "DiT/experiments/capacity_ladder/results/tier2_sp/evaluations/eval_20k/metrics_40k.json",
    },
    {
        "rank": 7,
        "experiment": "Capacity_Tier3_B_5k",
        "model_type": "DiT-2Cond-B/2 (LayerNorm+GELU)",
        "params": "131.0M",
        "channels": "4ch (SD-VAE latent)",
        "step": 5000,
        "strict_ssim": 0.5346,
        "strict_ssim_med": 0.5315,
        "strict_lpips": 0.3994,
        "strict_skel_iou": 0.0134,
        "strict_mse": 0.9963,
        "seen_ssim": 0.5169,
        "gap_ssim": "-0.0178",
        "status": "大模型 25% 起跑爬坡点，Loss 持续破新低",
        "source_file": "DiT/experiments/capacity_ladder/results/tier3_b/evaluations/eval_5k/metrics_40k.json",
    },
    {
        "rank": 8,
        "experiment": "Capacity_Tier3_B_Aug_48k",
        "model_type": "DiT-2Cond-B/2 (v4 对称增强版)",
        "params": "131.0M",
        "channels": "4ch (SD-VAE latent)",
        "step": 48000,
        "strict_ssim": "训练中",
        "strict_ssim_med": "训练中",
        "strict_lpips": "训练中",
        "strict_skel_iou": "训练中",
        "strict_mse": "训练中",
        "seen_ssim": "训练中",
        "gap_ssim": "训练中",
        "status": "当前运行中：10h 冲刺 + 7.8万对称数据增强",
        "source_file": "DiT/experiments/capacity_ladder/configs/tier3_b_aug.json",
    },
]

fieldnames = [
    "rank",
    "experiment",
    "model_type",
    "params",
    "channels",
    "step",
    "strict_ssim",
    "strict_ssim_med",
    "strict_lpips",
    "strict_skel_iou",
    "strict_mse",
    "seen_ssim",
    "gap_ssim",
    "status",
    "source_file",
]

with open(out_csv, "w", encoding="utf-8", newline="") as fp:
    w = csv.DictWriter(fp, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(benchmark_data)

print(f"Generated {out_csv} successfully:")
for r in benchmark_data:
    print(
        f"  Row {r['rank']}: {r['experiment']:<26} | Step {r['step']:<5} | SSIM: {str(r['strict_ssim']):<7} | LPIPS: {str(r['strict_lpips']):<7} | {r['status']}"
    )
print("=" * 80)
