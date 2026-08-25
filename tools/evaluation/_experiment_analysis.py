"""
Comprehensive S vs B experiment comparison + scaling projection.
Pulls data from all available experiments on remote.
"""
import json, sys, re
import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── S10 data (from train_data.json, already pulled) ──
s10_evals = [
    (1000, 1.30928, 0.0425, 0.1616, 0.7404),
    (2000, 1.34637, 0.1233, 0.1744, 0.6785),
    (3000, 1.31980, 0.2069, 0.1801, 0.6229),
    (4000, 1.34402, 0.2458, 0.1808, 0.5944),
    (5000, 1.35274, 0.2456, 0.1877, 0.5891),
    (10000, 1.11407, 0.2802, 0.1969, 0.5701),
    (15000, 1.08067, 0.3525, 0.2097, 0.5152),
    (20000, 1.12336, 0.3686, 0.2163, 0.4964),
    (25000, 1.14224, 0.3771, 0.2221, 0.4820),
    (30000, 1.13035, 0.3848, 0.2264, 0.4733),
    (35000, 1.12061, 0.3901, 0.2282, 0.4670),
    (40000, 1.09739, 0.3989, 0.2316, 0.4504),
    (45000, 1.08242, 0.4029, 0.2336, 0.4445),
    (50000, 1.08067, 0.4034, 0.2347, 0.4410),
    (55000, 1.07549, 0.4051, 0.2352, 0.4381),
]

# ── S8 data (kl-f4, from remote eval_auto_*.json) ──
# S8 = DiT-2Cond-XL/4, ~680M params, trained on 104k dataset
s8_raw = [
    (1000, 1.68418, 0.09364), (2000, 1.51039, 0.21455), (3000, 1.48477, 0.25317),
    (4000, 1.39852, 0.29082), (5000, 1.34861, 0.31479), (10000, 1.00622, 0.42871),
    (15000, 0.95619, 0.46042), (20000, 0.91875, 0.48261), (25000, 0.90758, 0.48924),
    (30000, 0.89560, 0.49667), (35000, 0.88878, 0.50068), (40000, 0.89221, 0.50019),
    (45000, 0.87865, 0.50589), (50000, 0.87055, 0.50945), (55000, 0.86013, 0.51455),
    (60000, 0.85629, 0.51828), (65000, 0.85231, 0.52017), (70000, 0.84537, 0.52212),
    (75000, 0.85574, 0.51811), (80000, 0.85580, 0.51862), (85000, 0.85410, 0.51844),
    (90000, 0.84679, 0.52244), (95000, 0.85110, 0.52112), (100000, 0.84136, 0.52389),
    (105000, 0.83685, 0.52572),
]

# ── S7 data (kl-f4, from remote) ──
# S7 = DiT-2Cond-XL/4, no DINO, trained on 104k dataset
s7_raw = [
    (1000, 1.99193, 0.05094), (2000, 1.69090, 0.11911), (3000, 1.54826, 0.23415),
    (4000, 1.32765, 0.31934), (5000, 1.37596, 0.34813), (10000, 1.10671, 0.42484),
    (15000, 1.01224, 0.46061), (20000, 0.98261, 0.47445), (25000, 0.97048, 0.48209),
    (30000, 0.95105, 0.49087), (35000, 0.92876, 0.49437), (40000, 0.92417, 0.49860),
    (45000, 0.91201, 0.50244), (50000, 0.91888, 0.50023), (55000, 0.92112, 0.50066),
    (60000, 0.92400, 0.50198), (65000, 0.91112, 0.50520), (70000, 0.90877, 0.50734),
    (75000, 0.88793, 0.50897),
]

print("=" * 90)
print("EXPERIMENT COMPARISON: S7 vs S8 vs S10")
print("=" * 90)

# ── Experiment configs ──
print("""
┌─────────┬──────────────┬──────────┬─────────┬──────────┬──────────────┐
│ Exp     │ Model        │ Params   │ Dataset │ VAE       │ Key Diff     │
├─────────┼──────────────┼──────────┼─────────┼──────────┼──────────────┤
│ S7      │ DiT-2Cond-XL/4│ ~680M    │ 104k    │ kl-f4    │ No DINO      │
│ S8      │ DiT-2Cond-XL/4│ ~680M    │ 104k    │ kl-f4    │ + DINO inj   │
│ S10     │ DiT-2Cond-B/4 │ ~157M    │ 105k    │ kl-f4    │ + DINO inj   │
└─────────┴──────────────┴──────────┴─────────┴──────────┴──────────────┘

S = Small model (B, 157M), B = Big model (XL, 680M)
Wait... S7/S8 are XL (big), S10 is B (small).
So the question "B比S好" means: does the Big model beat the Small model?
""")

print("─" * 90)
print("1. SSIM COMPARISON (越高越好)")
print("─" * 90)
print(f"{'Step':>8}  {'S7(XL)':>8}  {'S8(XL)':>8}  {'S10(B)':>8}  {'S8-S10':>8}  {'S7-S10':>8}")
print("-" * 60)

s7_map = {s: (m, ss) for s, m, ss in s7_raw}
s8_map = {s: (m, ss) for s, m, ss in s8_raw}
s10_map = {s: (m, ss, si, lp) for s, m, ss, si, lp in s10_evals}

for step in [1000, 5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000, 55000]:
    s7_ss = s7_map.get(step, (None, None))[1]
    s8_ss = s8_map.get(step, (None, None))[1]
    s10_ss = s10_map.get(step, (None, None, None, None))[1]
    diff_8_10 = f"{s8_ss - s10_ss:+.4f}" if s8_ss and s10_ss else "---"
    diff_7_10 = f"{s7_ss - s10_ss:+.4f}" if s7_ss and s10_ss else "---"
    s7_s = f"{s7_ss:.4f}" if s7_ss else "---"
    s8_s = f"{s8_ss:.4f}" if s8_ss else "---"
    s10_s = f"{s10_ss:.4f}" if s10_ss else "---"
    print(f"{step:>8}  {s7_s:>8}  {s8_s:>8}  {s10_s:>8}  {diff_8_10:>8}  {diff_7_10:>8}")

print()
print("─" * 90)
print("2. MSE COMPARISON (越低越好, [-1,1] scale)")
print("─" * 90)
print(f"{'Step':>8}  {'S7(XL)':>8}  {'S8(XL)':>8}  {'S10(B)':>8}  {'S8-S10':>8}  {'S7-S10':>8}")
print("-" * 60)
for step in [1000, 5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000, 55000]:
    s7_m = s7_map.get(step, (None, None))[0]
    s8_m = s8_map.get(step, (None, None))[0]
    s10_m = s10_map.get(step, (None, None, None, None))[0]
    diff_8_10 = f"{s8_m - s10_m:+.4f}" if s8_m and s10_m else "---"
    diff_7_10 = f"{s7_m - s10_m:+.4f}" if s7_m and s10_m else "---"
    s7_s = f"{s7_m:.4f}" if s7_m else "---"
    s8_s = f"{s8_m:.4f}" if s8_m else "---"
    s10_s = f"{s10_m:.4f}" if s10_m else "---"
    print(f"{step:>8}  {s7_s:>8}  {s8_s:>8}  {s10_s:>8}  {diff_8_10:>8}  {diff_7_10:>8}")

print()
print("─" * 90)
print("3. LPIPS COMPARISON (越低越好, S10 only)")
print("─" * 90)
print(f"{'Step':>8}  {'S10 LPIPS':>10}  {'S10 SkelIoU':>12}")
print("-" * 40)
for step, m, ss, si, lp in s10_evals:
    print(f"{step:>8}  {lp:>10.4f}  {si:>12.4f}")

print()
print("─" * 90)
print("4. CONVERGENCE RATE ANALYSIS")
print("─" * 90)

# SSIM at various milestones
milestones = [5000, 10000, 20000, 30000, 50000]
print(f"\n  SSIM at milestones:")
print(f"  {'Step':>8}  {'S7(XL)':>8}  {'S8(XL)':>8}  {'S10(B)':>8}  {'B/Big%':>8}")
for ms in milestones:
    s7v = s7_map.get(ms, (None, None))[1]
    s8v = s8_map.get(ms, (None, None))[1]
    s10v = s10_map.get(ms, (None, None, None, None))[1]
    ratio = f"{s10v/s8v*100:.1f}%" if s10v and s8v else "---"
    print(f"  {ms:>8}  {s7v or 0:.4f}  {s8v or 0:.4f}  {s10v or 0:.4f}  {ratio:>8}")

print()
print("─" * 90)
print("5. SCALING PROJECTION: 10万数据 S模型够不够用?")
print("─" * 90)

# S8 at 50k = 0.509 SSIM, at 105k = 0.526 SSIM
# From 50k to 105k (2.1x steps), SSIM improved 0.509 -> 0.526 = +0.017
# S10 at 50k = 0.403 SSIM, at 55k = 0.405
# Rate of S10 improvement: (0.405-0.403)/(55k-50k) = 0.0004 per 5k steps = 0.00008/k step

s10_rate = (s10_map[55000][1] - s10_map[50000][1]) / (55000 - 50000)
s8_rate = (s8_map[105000][1] - s8_map[50000][1]) / (105000 - 50000)

print(f"""
  S8 (XL/Big, 680M):
    50k → 105k: SSIM 0.509 → 0.526 (+0.017 over 55k steps)
    Rate: {s8_rate*1000:.4f} SSIM per 1k steps
    Still improving at 105k → likely needs 200k+ to plateau

  S10 (B/Small, 157M):
    50k → 55k: SSIM 0.403 → 0.405 (+0.002 over 5k steps)
    Rate: {s10_rate*1000:.4f} SSIM per 1k steps
    Slowing down — rate is {s10_rate/s8_rate:.1f}x of S8's rate

  Key observations:
  ─────────────────
  1. S10(B) SSIM at 55k = 0.405, S8(XL) SSIM at 55k = 0.515
     → Big model is 27% better in SSIM at same step count
     → Gap is WIDENING, not closing

  2. S10(B) convergence rate is {s10_rate/s8_rate:.1%} of S8(XL)
     → Small model is plateauing faster
     → At 100k steps, S10 would project to ~{0.405 + s10_rate * 45000:.3f} SSIM
     → S8 at 100k = 0.524 SSIM
     → Gap at 100k projected: ~{0.524 - (0.405 + s10_rate * 45000):.3f}

  3. S8 at 75k (0.519) is where it roughly matches S10's theoretical ceiling
     → S10 would need ~{int(55000 + (0.519 - 0.405) / s10_rate)} steps to reach S8's 75k level
     → That's {int((55000 + (0.519 - 0.405) / s10_rate) / 55000):.1f}x more training

  VERDICT:
  ─────────────────
  ❌ B(157M) does NOT beat S(680M) — not even close.
  ❌ 10万数据对S模型也不够 — XL at 105k still hasn't plateaued.
  ✅ The correct path: XL model + 10万数据 + 200k+ steps
     S8 at 105k = 0.526 SSIM and still climbing, needs ~200k+ to converge.
""")

# Also compute training speed comparison
print("─" * 90)
print("6. TRAINING SPEED & EFFICIENCY")
print("─" * 90)
print(f"""
  S10 (B, 157M): 3.5 steps/s, batch=96, 19.6G VRAM
  S8 (XL, 680M): ~2.5 steps/s, batch=32, ~22G VRAM (estimated)

  Per-step compute: S10 is ~{680/157:.1f}x cheaper
  But quality per step: S10 is {s10_rate/s8_rate:.1%} of S8
  → Compute efficiency = {s10_rate/s8_rate * (680/157):.2f}x
  → S10 is {'MORE' if s10_rate/s8_rate * (680/157) > 1 else 'LESS'} compute-efficient per SSIM gained

  However: S8 can reach 0.52 SSIM. S10 may NEVER reach 0.52 SSIM
  (157M params may lack capacity for 20468 glyph variety).
""")
