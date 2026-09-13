#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Phase 5: Use kl-f4-specific A/ceiling ratio (not S6's f8 ratio) for s11 calibration.
Also: combined fit using all f4 experiments to get robust ratio."""
import json, numpy as np
from scipy.optimize import curve_fit

data = json.load(open("tools/all_experiments_eval.json"))
exps = {}
for r in data:
    exps.setdefault(r["exp"], []).append(r)
for e in exps: exps[e].sort(key=lambda x: x["step"])

VAE_CEIL = {"sd-vae-f8": 0.966, "kl-f4": 0.975}

# Reliable fits (A not at ceiling, curve clearly bent):
reliable = {
    "s6_top6_diffonly":     {"A":0.7510, "tau":63727,  "vae":"sd-vae-f8","model":"S","data":"top6"},
    "s5_2factor_top30":     {"A":0.7575, "tau":250666, "vae":"sd-vae-f8","model":"S","data":"top30"},
    "s5_2factor_B_latentstruct":{"A":0.7445,"tau":453240,"vae":"sd-vae-f8","model":"B","data":"top30"},
    "s7_klf4_top30":        {"A":0.7093, "tau":142560, "vae":"kl-f4","model":"S","data":"top30"},
    "s8_klf4_clean_dino":   {"A":0.7216, "tau":184760, "vae":"kl-f4","model":"S","data":"top30"},
}

print("=" * 70)
print("Reliable A/ceiling ratios (curves that clearly bent, A not at ceiling)")
print("=" * 70)
print(f"{'experiment':<28} {'VAE':<11} {'model':<6} {'data':<8} {'A':>6} {'ceil':>6} {'A/ceil':>7} {'tau':>8}")
ratios_f8, ratios_f4 = [], []
for exp, f in reliable.items():
    ceil = VAE_CEIL[f["vae"]]
    ratio = f["A"] / ceil
    print(f"{exp:<28} {f['vae']:<11} {f['model']:<6} {f['data']:<8} {f['A']:>6.3f} {ceil:>6.3f} {ratio:>7.4f} {f['tau']:>8.0f}")
    if f["vae"] == "sd-vae-f8":
        ratios_f8.append(ratio)
    else:
        ratios_f4.append(ratio)

print(f"\n  f8 ratio: {np.mean(ratios_f8):.4f} ± {np.std(ratios_f8):.4f}  (n={len(ratios_f8)})")
print(f"  f4 ratio: {np.mean(ratios_f4):.4f} ± {np.std(ratios_f4):.4f}  (n={len(ratios_f4)})")
print(f"  Δ (f4 - f8) = {np.mean(ratios_f4)-np.mean(ratios_f8):.4f}")

# ============================================================
# Two calibration approaches for s11_top6_p4:
#   A) S6-anchored: same model+data, transfer ratio from f8 (0.777)
#   B) f4-specific: use f4 ratio (0.734), same VAE
#   C) Average of A and B
# ============================================================
s11_ceil = VAE_CEIL["kl-f4"]
ratio_S6 = 0.7774  # S6's ratio
ratio_f4 = np.mean(ratios_f4)  # 0.734
ratio_avg = (ratio_S6 + ratio_f4) / 2

print("\n" + "=" * 70)
print("Three calibration estimates for s11_top6_p4 asymptote A")
print("=" * 70)
for label, ratio in [("A) S6-anchored (f8 ratio)", ratio_S6),
                      ("B) f4-specific ratio", ratio_f4),
                      ("C) Average of A+B", ratio_avg)]:
    A = ratio * s11_ceil
    print(f"  {label}: ratio={ratio:.4f} × ceil={s11_ceil:.3f} → A={A:.4f}")

# ============================================================
# Refit s11 with each A
# ============================================================
rows = exps["s11_top6_p4"]
steps = np.array([r["step"] for r in rows if r["step"] >= 5000 and r["ssim"] is not None], dtype=float)
ssims = np.array([r["ssim"] for r in rows if r["step"] >= 5000 and r["ssim"] is not None], dtype=float)

print("\n" + "=" * 70)
print("s11_top6_p4 refit with constrained A (fit B, tau)")
print("=" * 70)

for label, ratio in [("A) S6 ratio 0.777", ratio_S6),
                      ("B) f4 ratio 0.734", ratio_f4),
                      ("C) Average 0.756", ratio_avg)]:
    A_fixed = ratio * s11_ceil
    def f(x, B, tau):
        return A_fixed - B * np.exp(-x / tau)
    try:
        popt, _ = curve_fit(f, steps, ssims, p0=[0.5, 150000],
                           bounds=([0.2, 20000], [1.5, 2000000]), maxfev=20000)
        B, tau = popt
        rmse = np.sqrt(np.mean((ssims - f(steps, *popt))**2))
        print(f"\n  {label}: A={A_fixed:.4f}")
        print(f"    B={B:.4f}  tau={tau:.0f}  RMSE={rmse:.4f}")
        print(f"    SSIM = {A_fixed:.4f} - {B:.4f} * exp(-step/{tau:.0f})")
        for s in [140000, 150000, 200000, 250000, 300000, 400000, 500000, 600000]:
            pred = A_fixed - B * np.exp(-s / tau)
            mark = " ← 当前" if s == 140000 else ""
            print(f"      @{s//1000:>4}k: {pred:.4f}{mark}")
        print(f"    极限 SSIM = {A_fixed:.4f}")
        for target in [0.65, 0.70, 0.732, 0.75]:
            if A_fixed > target:
                st = -tau * np.log((A_fixed - target) / B)
                print(f"    SSIM={target:.3f} @ step ≈ {st:.0f}")
    except Exception as e:
        print(f"  {label}: failed ({e})")
