import os
import sys
import csv
import json
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torchvision.transforms as T
from skimage.metrics import structural_similarity as ssim
import lpips

sys.stdout.reconfigure(encoding="utf-8")

# 1. 载入原始 eval200_fixed.csv
csv_orig = "exp-std/csv/eval200_fixed.csv"
if not os.path.exists(csv_orig):
    csv_orig = "exp-std-csv/eval200_fixed.csv"

with open(csv_orig, "r", encoding="utf-8") as f:
    orig_rows = list(csv.DictReader(f))

print(f"原始 eval200_fixed.csv 共 {len(orig_rows)} 行")

# 2. 载入 v68 的每样本指标以确定难易度排序
df_all = pd.read_csv("assets/eval200fix_models_per_sample.csv")
df_v68 = df_all[df_all["model_name"] == "v68"].copy()
df_v68["ssim"] = df_v68["ssim"].astype(float)
df_v68["idx"] = df_v68["idx"].astype(int)
df_v68_ranked = df_v68.sort_values(by="ssim", ascending=False).reset_index(drop=True)

# 3. 三等分划分:
# Top: 62 行 (0..61)
# Mid: 63 行 (62..124)
# Worst: 62 行 (125..186)
top_indices = df_v68_ranked.iloc[0:62]["idx"].tolist()
mid_indices = df_v68_ranked.iloc[62:125]["idx"].tolist()
worst_indices = df_v68_ranked.iloc[125:187]["idx"].tolist()

print(f"三等分数量: Top={len(top_indices)}, Mid={len(mid_indices)}, Worst={len(worst_indices)}, 总和={len(top_indices)+len(mid_indices)+len(worst_indices)}")

# 4. 生成三个固定 CSV
os.makedirs("exp-std/csv", exist_ok=True)
csv_top_path = "exp-std/csv/eval200_split_top.csv"
csv_mid_path = "exp-std/csv/eval200_split_mid.csv"
csv_worst_path = "exp-std/csv/eval200_split_worst.csv"

def save_split_csv(indices, out_path):
    rows_to_save = [orig_rows[i] for i in indices]
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=orig_rows[0].keys())
        writer.writeheader()
        writer.writerows(rows_to_save)
    print(f"已保存固定切分 CSV: {out_path} ({len(rows_to_save)} 行)")

save_split_csv(top_indices, csv_top_path)
save_split_csv(mid_indices, csv_mid_path)
save_split_csv(worst_indices, csv_worst_path)

# 5. 准备计算每个模型在 187 个样本上的 SSIM, LPIPS, IoU
# 检查本地可用图片目录
print("\n=== 计算所有模型在各切分集上的精确指标 ===")
device = "cuda" if torch.cuda.is_available() else "cpu"
lpips_fn = lpips.LPIPS(net="vgg").to(device).eval()

# GT 目录
gt_dir = "assets/server_48_evals/moyi_12ch_eval200fix"

# 计算单个模型在所有 187 样本上的指标
def eval_model_dir(model_name, img_dir, prefix="g"):
    records = []
    for idx in range(187):
        pred_p = os.path.join(img_dir, f"{prefix}{idx}.png")
        if not os.path.exists(pred_p):
            pred_p = os.path.join(img_dir, f"{idx}.png")
        if not os.path.exists(pred_p):
            pred_p = os.path.join(img_dir, f"{idx:02d}.png")
        
        gt_p = os.path.join(gt_dir, f"gt{idx}.png")
        if not os.path.exists(gt_p):
            continue
            
        if not os.path.exists(pred_p):
            continue

        im_pred = Image.open(pred_p).convert("RGB")
        im_gt = Image.open(gt_p).convert("RGB")
        
        # 1. SSIM (灰度)
        g_l = np.array(im_pred.convert("L")).astype(np.float32) / 255.0
        gt_l = np.array(im_gt.convert("L")).astype(np.float32) / 255.0
        s_val = ssim(g_l, gt_l, data_range=1.0)
        
        # 2. Ink IoU (前景墨迹阈值 0.5)
        bin_pred = (g_l < 0.5).astype(np.uint8)
        bin_gt = (gt_l < 0.5).astype(np.uint8)
        inter = np.logical_and(bin_pred, bin_gt).sum()
        union = np.logical_or(bin_pred, bin_gt).sum()
        iou_val = inter / (union + 1e-6)
        
        # 3. LPIPS
        t_pred = (T.ToTensor()(im_pred).unsqueeze(0).to(device) * 2.0) - 1.0
        t_gt = (T.ToTensor()(im_gt).unsqueeze(0).to(device) * 2.0) - 1.0
        with torch.no_grad():
            lp_val = float(lpips_fn(t_pred, t_gt).squeeze().cpu().item())
            
        records.append({
            "idx": idx,
            "ssim": s_val,
            "lpips": lp_val,
            "iou": iou_val
        })
    df_res = pd.DataFrame(records)
    print(f"模型 [{model_name}]: 成功计算 {len(df_res)}/187 个样本指标")
    return df_res

# 汇总各模型目录
# 注意 moyi_4ch 在 assets/server_48_evals/eval_ours200fix 或 moyi_4ch_eval200fix
p_4ch = "assets/server_48_evals/eval_ours200fix"
if not os.path.exists(p_4ch):
    p_4ch = "assets/server_48_evals/moyi_4ch_eval200fix"

model_dirs = {
    "v68": ("assets/results/v68_aug_sp_c2ot/20261006-192021-v68-aug-sp-c2ot/eval_samples_ctrl/step0200000/eval200fix", "g"),
    "v66": ("assets/results/v66_tables_condroute2456/20261006-024102-v66-tables-condroute2456-adaLN/eval_samples_ctrl/step0150000/eval200fix", "g"),
    "v54": ("assets/v54_eval200fix", "g"),
    "moyi_12ch": ("assets/server_48_evals/moyi_12ch_eval200fix", "g"),
    "moyi_4ch": (p_4ch, "g"),
    "v70": ("assets/results/v70_aug_sp_stdskel_c2ot/20261007-141413-v70-aug-sp-stdskel-c2ot/eval_samples_ctrl/step0030000/eval200fix", "g"),
    "02_v13": ("assets/eval200_30_gathered/02_v13", ""),
    "03_v21": ("assets/eval200_30_gathered/03_v21", ""),
    "04_v23": ("assets/eval200_30_gathered/04_v23", ""),
    "gt": ("assets/server_48_evals/moyi_12ch_eval200fix", "gt"),
}

metrics_by_model = {}
for mname, (mdir, pref) in model_dirs.items():
    if os.path.exists(mdir):
        metrics_by_model[mname] = eval_model_dir(mname, mdir, pref)
    else:
        print(f"⚠ 跳过 {mname}: 目录不存在 {mdir}")

# 6. 计算每个模型在三个集合上的均值
splits = {
    "top": set(top_indices),
    "mid": set(mid_indices),
    "worst": set(worst_indices),
}

summary_table = []
for mname, df_m in metrics_by_model.items():
    row_data = {"model": mname}
    for sname, s_idxs in splits.items():
        sub_df = df_m[df_m["idx"].isin(s_idxs)]
        if len(sub_df) > 0:
            row_data[f"{sname}_ssim"] = sub_df["ssim"].mean()
            row_data[f"{sname}_lpips"] = sub_df["lpips"].mean()
            row_data[f"{sname}_iou"] = sub_df["iou"].mean()
            row_data[f"{sname}_count"] = len(sub_df)
        else:
            row_data[f"{sname}_ssim"] = 0.0
            row_data[f"{sname}_lpips"] = 0.0
            row_data[f"{sname}_iou"] = 0.0
            row_data[f"{sname}_count"] = 0
    summary_table.append(row_data)

df_splits_summary = pd.DataFrame(summary_table)
summary_csv = "assets/eval200_splits_exact_metrics.csv"
df_splits_summary.to_csv(summary_csv, index=False)
print(f"\n✓ 成功保存三等分集合上的精确统计数据: {summary_csv}")
print(df_splits_summary[["model", "top_ssim", "top_lpips", "top_iou", "mid_ssim", "mid_lpips", "mid_iou", "worst_ssim", "worst_lpips", "worst_iou"]])
