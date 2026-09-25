import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "docs/04_experiments/imgs")
os.makedirs(OUT_DIR, exist_ok=True)

with open(os.path.join(ROOT, "assets/all_runs_deep_parsed.json"), "r", encoding="utf-8") as f:
    all_runs = json.load(f)

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 1.0

# -------------------------------------------------------------
# Chart 1: Historical Timeline Milestone Curves (Strict SSIM)
# -------------------------------------------------------------
def plot_timeline_milestones():
    fig, ax = plt.subplots(figsize=(12, 6), dpi=200)
    
    milestone_keys = [
        ("v10b_stdskel_fame3_c41x_cos_e", "v10b-c41x-cos-e (360k)", "#888888", "--"),
        ("v11_struct-loss", "v11-struct-loss (490k)", "#4A90E2", "-."),
        ("v12_pretrain_S_cat_fame_kxl_tj_px60", "v12-S-cat (100k)", "#7ED321", ":"),
        ("v13_base_50k", "v13-base-50k (155k)", "#F5A623", "-"),
        ("v14_style87_s2", "v14-style87-s2 (160k)", "#BD10E0", "-"),
        ("v14_style87_s3", "v14-style87-s3 (162.5k)", "#9013FE", "-"),
        ("v17_mid_250k", "v17-mid-250k (110k)", "#50E3C2", "-"),
        ("v21_skelnet_200k", "v21-skelnet-200k (Live 25k)", "#D0021B", "-")
    ]
    
    for key, label, color, ls in milestone_keys:
        if key in all_runs:
            steps_dict = all_runs[key]["steps"]
            x = []
            y = []
            for st, sdata in sorted(steps_dict.items(), key=lambda t: int(t[0])):
                if "strict" in sdata and sdata["strict"]["ssim"] > 0:
                    x.append(int(st) / 1000.0)
                    y.append(sdata["strict"]["ssim"])
            if x and y:
                lw = 2.5 if "v21" in key or "v14" in key else 1.5
                ax.plot(x, y, label=label, color=color, linestyle=ls, linewidth=lw, marker='o' if len(x) < 15 else None, markersize=4)
                
    ax.set_title("Callig-DiT Historical Evolution: Strict SSIM Trajectory Across Eras", fontsize=14, pad=12, fontweight='bold')
    ax.set_xlabel("Training Steps (x1,000)", fontsize=11)
    ax.set_ylabel("Strict SSIM (Out-of-Distribution Generalization)", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="lower right", framealpha=0.9, fontsize=9)
    ax.set_ylim(0.48, 0.585)
    
    # Peak annotation
    ax.annotate("v14-style87 Peak: 0.5764", xy=(162.5, 0.5764), xytext=(120, 0.581),
                arrowprops=dict(facecolor='#9013FE', shrink=0.05, width=1.5, headwidth=6),
                fontsize=9, fontweight='bold', color='#9013FE')
                
    out_path = os.path.join(OUT_DIR, "curves_timeline_milestones.png")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print("Saved:", out_path)

# -------------------------------------------------------------
# Chart 2: v21 SkelNet Live Telemetry Dashboard (4 Panels)
# -------------------------------------------------------------
def plot_v21_telemetry():
    v21_data = all_runs.get("v21_skelnet_200k", {}).get("steps", {})
    if not v21_data:
        print("No v21 data found!")
        return

    steps = []
    seen_ssim = []
    strict_ssim = []
    strict_lpips = []
    strict_mse = []
    tgt_spec_seen = []
    tgt_spec_strict = []
    cal_enrich_seen = []
    cal_enrich_strict = []
    frag_strict = []

    for st, sdata in sorted(v21_data.items(), key=lambda t: int(t[0])):
        s_int = int(st)
        steps.append(s_int / 1000.0)
        seen = sdata.get("seen", {})
        strict = sdata.get("strict", {})
        
        seen_ssim.append(seen.get("ssim", 0.0))
        strict_ssim.append(strict.get("ssim", 0.0))
        strict_lpips.append(strict.get("lpips", 0.0))
        strict_mse.append(strict.get("mse", 0.0))
        
        tgt_spec_seen.append(seen.get("target_spec") or 0.0)
        tgt_spec_strict.append(strict.get("target_spec") or 0.0)
        
        cal_enrich_seen.append(seen.get("cal_enrich") or 0.0)
        cal_enrich_strict.append(strict.get("cal_enrich") or 0.0)
        
        frag_strict.append(strict.get("frag") or 0.0)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=200)

    # Panel 1: SSIM
    ax1 = axes[0, 0]
    ax1.plot(steps, seen_ssim, 'b-o', label='Seen SSIM (Reconstruction)', linewidth=2)
    ax1.plot(steps, strict_ssim, 'r-s', label='Strict SSIM (Generalization)', linewidth=2)
    ax1.set_title("Reconstruction vs Generalization SSIM", fontweight='bold')
    ax1.set_xlabel("Steps (k)")
    ax1.set_ylabel("SSIM")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend()
    ax1.set_ylim(0.515, 0.555)

    # Panel 2: Target Specificity
    ax2 = axes[0, 1]
    ax2.plot(steps, tgt_spec_seen, 'g-o', label='Seen Target Specificity', linewidth=2)
    ax2.plot(steps, tgt_spec_strict, 'm-s', label='Strict Target Specificity', linewidth=2)
    ax2.axhline(0, color='gray', linestyle=':', label='Zero baseline')
    ax2.set_title("Target Specificity (Margin over closest rival)", fontweight='bold')
    ax2.set_xlabel("Steps (k)")
    ax2.set_ylabel("Margin Delta SSIM")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend()

    # Panel 3: Calligrapher Enrichment Ratio
    ax3 = axes[1, 0]
    ax3.plot(steps, cal_enrich_seen, 'c-o', label='Seen Calligrapher Enrichment', linewidth=2)
    ax3.plot(steps, cal_enrich_strict, 'y-s', label='Strict Calligrapher Enrichment', linewidth=2)
    ax3.axhline(1.0, color='r', linestyle='--', label='1.0x Parity Line')
    ax3.set_title("Calligrapher Enrichment (>1.0x implies style affinity)", fontweight='bold')
    ax3.set_xlabel("Steps (k)")
    ax3.set_ylabel("Enrichment Multiplier")
    ax3.grid(True, linestyle="--", alpha=0.5)
    ax3.legend()

    # Panel 4: Quality & Topology (LPIPS and Frag)
    ax4 = axes[1, 1]
    color = 'tab:blue'
    ax4.set_xlabel("Steps (k)")
    ax4.set_ylabel("Strict LPIPS (Lower is better)", color=color)
    l1 = ax4.plot(steps, strict_lpips, color=color, marker='o', linewidth=2, label='Strict LPIPS')
    ax4.tick_params(axis='y', labelcolor=color)
    ax4.grid(True, linestyle="--", alpha=0.5)

    ax4_twin = ax4.twinx()
    color = 'tab:orange'
    ax4_twin.set_ylabel("Stroke Fragmentation Ratio (Lower is better)", color=color)
    l2 = ax4_twin.plot(steps, frag_strict, color=color, marker='^', linewidth=2, linestyle='--', label='Strict Fragmentation')
    ax4_twin.tick_params(axis='y', labelcolor=color)

    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax4.legend(lines, labels, loc='upper right')
    ax4.set_title("Perceptual Quality & Stroke Connectedness", fontweight='bold')

    plt.suptitle("v21-skelnet-200k Live Multi-Metric Telemetry (Step 5k - 25k)", fontsize=16, fontweight='bold', y=0.99)
    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "curves_v21_live_telemetry.png")
    plt.savefig(out_path)
    plt.close()
    print("Saved:", out_path)

# -------------------------------------------------------------
# Chart 3: Generalization Gap (Seen vs Strict across 74 runs)
# -------------------------------------------------------------
def plot_generalization_gap():
    fig, ax = plt.subplots(figsize=(10, 8), dpi=200)

    seen_vals = []
    strict_vals = []
    labels = []
    categories = []

    for k, v in all_runs.items():
        peak_seen = 0.0
        peak_strict = 0.0
        for st, sdata in v["steps"].items():
            if "seen" in sdata and sdata["seen"]["ssim"] > peak_seen:
                peak_seen = sdata["seen"]["ssim"]
            if "strict" in sdata and sdata["strict"]["ssim"] > peak_strict:
                peak_strict = sdata["strict"]["ssim"]
        if peak_seen > 0.4 and peak_strict > 0.4:
            seen_vals.append(peak_seen)
            strict_vals.append(peak_strict)
            labels.append(k)
            if "v21" in k:
                categories.append("v21 (SkelNet Decoupled)")
            elif "v14" in k or "v13" in k or "v15" in k:
                categories.append("v13-v15 (50k Landmark)")
            elif "v17" in k:
                categories.append("v17 (Injection Ablation)")
            else:
                categories.append("Early Eras (v10b-v12)")

    # Color map
    cat_colors = {
        "v21 (SkelNet Decoupled)": ("#D0021B", 120, 3.0),
        "v13-v15 (50k Landmark)": ("#4A90E2", 60, 1.0),
        "v17 (Injection Ablation)": ("#F5A623", 45, 1.0),
        "Early Eras (v10b-v12)": ("#9B9B9B", 30, 0.8)
    }

    for cat in set(categories):
        cx = [seen_vals[i] for i in range(len(categories)) if categories[i] == cat]
        cy = [strict_vals[i] for i in range(len(categories)) if categories[i] == cat]
        c, s, a = cat_colors[cat]
        ax.scatter(cx, cy, label=cat, color=c, s=s, alpha=0.85, edgecolors='black', linewidth=0.5)

    # Reference y=x line
    ax.plot([0.45, 0.8], [0.45, 0.8], 'k--', alpha=0.5, label='Zero Gap Line (Seen = Strict)')
    ax.fill_between([0.45, 0.8], [0.45, 0.8], [0.4, 0.4], color='gray', alpha=0.1, label='Memorization Overfit Zone')

    ax.set_title("Seen vs Strict SSIM: Diagnosing the Generalization Gap", fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel("Seen SSIM (In-Distribution Memorization / Reconstruction)", fontsize=11)
    ax.set_ylabel("Strict SSIM (Out-of-Distribution Generalization)", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper left", framealpha=0.9)
    ax.set_xlim(0.44, 0.81)
    ax.set_ylim(0.47, 0.59)

    out_path = os.path.join(OUT_DIR, "curves_seen_vs_strict_gap.png")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print("Saved:", out_path)

# -------------------------------------------------------------
# Chart 4: Few-Shot Adaptation (Shen Zhou)
# -------------------------------------------------------------
def plot_fewshot_comparison():
    fig, ax = plt.subplots(figsize=(10, 6), dpi=200)

    # Historical runs for Shen Zhou
    fs_runs = [
        ("row_pt_lr0.001scale_0921-040243", "row_pt (lr=1e-3, scale) - Peak 0.5568", "#D0021B", "-"),
        ("row_pt_lr0.001s1k", "row_pt (lr=1e-3, 1k)", "#F5A623", "--"),
        ("row_pt_lr0.0003s1k", "row_pt (lr=3e-4, 1k)", "#7ED321", "-."),
        ("mean_scaled_lr0.001s1k", "mean_scaled (lr=1e-3, 1k)", "#4A90E2", ":"),
        ("mean_scaled_lr0.0001s1k", "mean_scaled (lr=1e-4, 1k)", "#9B9B9B", "-.")
    ]

    for subk, label, color, ls in fs_runs:
        matched_k = None
        for k in all_runs.keys():
            if subk in k:
                matched_k = k
                break
        if matched_k:
            steps_dict = all_runs[matched_k]["steps"]
            x = []
            y = []
            for st, sdata in sorted(steps_dict.items(), key=lambda t: int(t[0])):
                val = 0.0
                for sname in ["fewshot", "strict", "seen"]:
                    if sname in sdata and sdata[sname]["ssim"] > val:
                        val = sdata[sname]["ssim"]
                if val > 0:
                    x.append(int(st) - 150000)  # relative adaptation steps
                    y.append(val)
            if x and y:
                ax.plot(x, y, label=label, color=color, linestyle=ls, linewidth=2, marker='o')

    ax.set_title("Few-Shot Adaptation Dynamics on Shen Zhou (100 samples)", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Relative Adaptation Steps (from base checkpoint)", fontsize=11)
    ax.set_ylabel("Few-Shot SSIM", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="lower right", framealpha=0.9)

    out_path = os.path.join(OUT_DIR, "curves_fewshot_comparison.png")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print("Saved:", out_path)

# -------------------------------------------------------------
# Chart 5: Falsification Registry Quantitative Comparison
# -------------------------------------------------------------
def plot_falsification_comparison():
    fig, ax = plt.subplots(figsize=(12, 6), dpi=200)

    directions = [
        "Gold Baseline\n(v13 Base 50k)",
        "Capacity Explode\n(67M Large)",
        "Capacity Drop\n(XS/2 Depth 8)",
        "12-Channel\nDiffusion",
        "Gaussian Noise\nAug (400k)",
        "Discrete\nChar ID",
        "LCA Cross-Attn\n(v17_lca_x)"
    ]

    scores = [
        0.5547,
        0.5512,
        0.5337,
        0.4993,
        0.4869,
        0.4200,
        0.0893
    ]

    colors = ['#4A90E2', '#E74C3C', '#E74C3C', '#E74C3C', '#E74C3C', '#C0392B', '#7B1FA2']

    bars = ax.bar(directions, scores, color=colors, width=0.55, edgecolor='black', linewidth=1)

    ax.axhline(0.5547, color='#4A90E2', linestyle='--', alpha=0.7, label='Baseline Strict SSIM (0.5547)')

    for bar, score in zip(bars, scores):
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.012, f"{score:.4f}", ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_title("Quantitative Comparison of Falsified Directions vs Baseline (Strict SSIM)", fontsize=13, fontweight='bold', pad=12)
    ax.set_ylabel("Strict SSIM", fontsize=11)
    ax.set_ylim(0.0, 0.65)
    ax.grid(True, axis='y', linestyle="--", alpha=0.4)
    ax.legend(loc="upper right")

    out_path = os.path.join(OUT_DIR, "chart_falsification_comparison.png")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print("Saved:", out_path)

if __name__ == "__main__":
    plot_timeline_milestones()
    plot_v21_telemetry()
    plot_generalization_gap()
    plot_fewshot_comparison()
    plot_falsification_comparison()
    print("All charts successfully generated!")
