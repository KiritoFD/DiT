#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Cross-experiment calibration: build a unified mathematical model of SSIM
convergence using ALL experiments, then calibrate s11_top6_p4's fit.

Key idea: different experiments share the same fundamental learning dynamics
(SSIM grows asymptotically toward a ceiling). We fit a parametric model where:
  - The asymptote (ceiling) depends on VAE + model size + dataset size
  - The time constant (tau) depends on model size + dataset size + batch
  - The shape is shared (exponential approach: SSIM = A - B*exp(-step/tau))

We use the experiments that ran long enough to constrain the asymptote:
  - s6_top6_diffonly: S/2, sd-vae-f8, top6, 195k steps, SSIM 0.732 (best anchor)
  - s5_2factor_top30: S/2, sd-vae-f8, top30, 70k steps
  - s5_2factor_B_latentstruct: B/2, sd-vae-f8, top30, 130k steps
  - s7_klf4_top30: S/4, kl-f4, top30, 75k steps
  - s8_klf4_clean_dino: S/4, kl-f4, top30, 105k steps
  - s11_top6_p4: S/4, kl-f4, top6, 140k steps (target)
"""
import json, numpy as np
from scipy.optimize import curve_fit, minimize

data = json.load(open("tools/all_experiments_eval.json"))

# VAE ceilings (from VAE_NOISE_FLOOR.md): reconstruction SSIM
VAE_CEIL = {
    "sd-vae-f8": 0.966,   # verify_latents 100 img
    "kl-f4": 0.975,       # vae_noise_eval 79 img
}

# Group by experiment
exps = {}
for r in data:
    e = r["exp"]
    if e not in exps:
        exps[e] = []
    exps[e].append(r)

for e in exps:
    exps[e].sort(key=lambda x: x["step"])

# ============================================================
# Phase 1: Fit each long-running experiment independently
#          SSIM = A - B*exp(-step/tau)
# ============================================================
print("=" * 70)
print("Phase 1: Per-experiment exponential fit SSIM = A - B*exp(-step/tau)")
print("=" * 70)

def exp_approach(x, A, B, tau):
    return A - B * np.exp(-x / tau)

fits = {}
for exp, rows in exps.items():
    steps = np.array([r["step"] for r in rows if r["step"] >= 5000 and r["ssim"] is not None], dtype=float)
    ssims = np.array([r["ssim"] for r in rows if r["step"] >= 5000 and r["ssim"] is not None], dtype=float)
    if len(steps) < 5:
        continue
    vae = rows[0].get("vae", "sd-vae-f8")
    ceil = VAE_CEIL.get(vae, 0.97)
    try:
        popt, _ = curve_fit(exp_approach, steps, ssims,
                           p0=[ceil, ceil, 100000],
                           bounds=([0.5, 0.3, 5000], [ceil, 1.5, 2000000]),
                           maxfev=20000)
        A, B, tau = popt
        resid = ssims - exp_approach(steps, *popt)
        rmse = np.sqrt(np.mean(resid**2))
        fits[exp] = {"A": A, "B": B, "tau": tau, "rmse": rmse, "n": len(steps),
                     "step_range": f"{int(steps[0])}-{int(steps[-1])}",
                     "final_ssim": ssims[-1]}
        print(f"\n  {exp}")
        print(f"    data: {len(steps)} pts, steps {int(steps[0])}-{int(steps[-1])}, "
              f"SSIM {ssims[0]:.3f}->{ssims[-1]:.3f}")
        print(f"    A={A:.4f}  B={B:.4f}  tau={tau:.0f}  RMSE={rmse:.4f}")
        print(f"    VAE ceiling={ceil:.3f}, A/ceil={A/ceil:.3f}")
    except Exception as e:
        print(f"  {exp}: fit failed ({e})")

# ============================================================
# Phase 2: Cross-experiment analysis
#   Look for systematic relationships:
#   - A/ceil ratio: how close to VAE ceiling does DiT get?
#   - tau scaling: how does dataset/model affect convergence speed?
# ============================================================
print("\n" + "=" * 70)
print("Phase 2: Cross-experiment pattern analysis")
print("=" * 70)

print("\n--- Asymptote / VAE-ceiling ratio ---")
print(f"{'experiment':<30} {'VAE':<12} {'model':<6} {'data':<8} {'A':>6} {'ceil':>6} {'A/ceil':>7} {'tau':>8} {'RMSE':>6}")
for exp, f in sorted(fits.items(), key=lambda x: x[1]["A"], reverse=True):
    vae = exps[exp][0].get("vae", "?")
    model = exps[exp][0].get("model", "?")
    data_n = exps[exp][0].get("data", "?")
    ceil = VAE_CEIL.get(vae, 0.97)
    print(f"{exp:<30} {vae:<12} {model:<6} {data_n:<8} {f['A']:>6.3f} {ceil:>6.3f} "
          f"{f['A']/ceil:>7.3f} {f['tau']:>8.0f} {f['rmse']:>6.4f}")

# ============================================================
# Phase 3: S6 as the gold standard anchor
#   S6 ran to 195k with sd-vae-f8 and reached 0.732.
#   It's the only experiment that clearly shows the curve bending.
#   Use S6 to calibrate the "A/ceil" ratio for similar configs.
# ============================================================
print("\n" + "=" * 70)
print("Phase 3: S6-anchored calibration for s11_top6_p4")
print("=" * 70)

if "s6_top6_diffonly" in fits:
    s6 = fits["s6_top6_diffonly"]
    s6_vae_ceil = VAE_CEIL["sd-vae-f8"]
    s6_ratio = s6["A"] / s6_vae_ceil
    print(f"\nS6 (sd-vae-f8, S/2, top6):")
    print(f"  A = {s6['A']:.4f}, VAE ceiling = {s6_vae_ceil:.3f}")
    print(f"  A/ceiling = {s6_ratio:.4f}  (DiT reaches {s6_ratio*100:.1f}% of VAE ceiling)")
    print(f"  tau = {s6['tau']:.0f} steps")
    print(f"  final SSIM at {s6['step_range'].split('-')[1]} = {s6['final_ssim']:.4f}")

    # For s11: same model size (S), same dataset (top6), different VAE (kl-f4)
    # Hypothesis: A/ceiling ratio is roughly transferable for same model+data
    s11_ceil = VAE_CEIL["kl-f4"]
    s11_predicted_A = s6_ratio * s11_ceil

    print(f"\n  → Calibrated prediction for s11_top6_p4 (kl-f4, S/4, top6):")
    print(f"    VAE ceiling = {s11_ceil:.3f}")
    print(f"    predicted A = {s6_ratio:.4f} × {s11_ceil:.3f} = {s11_predicted_A:.4f}")

    # tau: s11 uses S/4 (256 tokens, same as S/2 with f8) + kl-f4 (3x latent info)
    # kl-f4 latent is 3x larger → harder to learn → tau likely larger
    # But same token count (256) and same batch (224) → compute per step similar
    # Use s11's own tau from independent fit, but constrain A
    if "s11_top6_p4" in fits:
        s11_fit = fits["s11_top6_p4"]
        print(f"\n    s11 independent fit: A={s11_fit['A']:.4f} (hit ceiling), "
              f"tau={s11_fit['tau']:.0f}")
        # Use s11's tau but calibrated A
        cal_A = s11_predicted_A
        cal_B = s11_fit["B"]
        cal_tau = s11_fit["tau"]
        print(f"    Calibrated: A={cal_A:.4f}, B={cal_B:.4f}, tau={cal_tau:.0f}")
        print(f"\n    Calibrated SSIM predictions:")
        for s in [140000, 150000, 200000, 230000, 300000, 400000, 500000, 600000]:
            pred = cal_A - cal_B * np.exp(-s / cal_tau)
            print(f"      SSIM@{s//1000:>4}k = {pred:.4f}")

# ============================================================
# Phase 4: Also fit s11 with A constrained to calibrated value
# ============================================================
print("\n" + "=" * 70)
print("Phase 4: s11_top6_p4 refit with A constrained to calibrated value")
print("=" * 70)

if "s11_top6_p4" in fits and "s6_top6_diffonly" in fits:
    rows = exps["s11_top6_p4"]
    steps = np.array([r["step"] for r in rows if r["step"] >= 5000 and r["ssim"] is not None], dtype=float)
    ssims = np.array([r["ssim"] for r in rows if r["step"] >= 5000 and r["ssim"] is not None], dtype=float)

    # Fix A = calibrated value, fit only B and tau
    A_fixed = s11_predicted_A
    def exp_approach_fixedA(x, B, tau):
        return A_fixed - B * np.exp(-x / tau)

    try:
        popt, _ = curve_fit(exp_approach_fixedA, steps, ssims,
                           p0=[0.8, 200000],
                           bounds=([0.3, 10000], [1.5, 2000000]),
                           maxfev=20000)
        B, tau = popt
        resid = ssims - exp_approach_fixedA(steps, *popt)
        rmse = np.sqrt(np.mean(resid**2))
        print(f"\n  A (fixed) = {A_fixed:.4f}  (calibrated from S6 ratio × kl-f4 ceiling)")
        print(f"  B (fit)   = {B:.4f}")
        print(f"  tau (fit) = {tau:.0f} steps")
        print(f"  RMSE      = {rmse:.4f}")
        print(f"\n  Calibrated convergence curve for s11_top6_p4:")
        print(f"  SSIM = {A_fixed:.4f} - {B:.4f} * exp(-step / {tau:.0f})")
        print()
        print(f"  {'step':>8} {'SSIM':>8}  {'current':>8}")
        for s in [140000, 150000, 200000, 250000, 300000, 400000, 500000, 600000]:
            pred = A_fixed - B * np.exp(-s / tau)
            cur = ssims[-1] if s <= steps[-1] else None
            cur_str = f"{cur:.4f}" if cur is not None else "—"
            print(f"  {s//1000:>6}k {pred:>8.4f}  {cur_str:>8}")
        print(f"\n  → 极限 SSIM = {A_fixed:.4f}")
        # when does it reach S6's 0.732?
        target = 0.732
        if A_fixed > target:
            steps_to_target = -tau * np.log((A_fixed - target) / B)
            print(f"  → 追平 S6 (SSIM=0.732) 在 step ≈ {steps_to_target:.0f}")
        target2 = 0.80
        if A_fixed > target2:
            steps_to_target2 = -tau * np.log((A_fixed - target2) / B)
            print(f"  → SSIM=0.80 在 step ≈ {steps_to_target2:.0f}")
    except Exception as e:
        print(f"  refit failed: {e}")
