"""Analyze results.csv → insights (unified Gaussian SSIM)."""
import csv, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

rows = list(csv.DictReader(open("results.csv", encoding="utf-8")))

def flt(series, step=None, run_sub=None):
    out = []
    for r in rows:
        if r["series"] != series:
            continue
        if step is not None and int(r["step"]) != step:
            continue
        if run_sub and run_sub not in r["run"]:
            continue
        out.append(r)
    return out

def ssim_at(series, step, run_sub=None):
    for r in flt(series, step, run_sub):
        if r.get("ssim") not in (None, ""):
            return float(r["ssim"])
    return None

def best(series, run_sub=None):
    best_r, best_v = None, -1
    for r in flt(series, run_sub=run_sub):
        v = r.get("ssim") or r.get("ssim_ctrl")
        if v not in (None, "") and float(v) > best_v:
            best_v, best_r = float(v), r
    return best_r, best_v

print("=" * 78)
print("A. 主系列巅峰 SSIM (Gaussian 统一口径, eval100_top6.csv 100张)")
print("=" * 78)
print(f"{'exp':<28}{'model':<16}{'vae':<6}{'data':<20}{'dino':<6}{'lr':<9}{'best_ssim':>10}{'@step':>8}")
summ = []
for series, label, key in [
    ("s6_top6_diffonly", "s6 (f8 S/2 top6)", "ssim"),
    ("s7_klf4_top30", "s7 (f4 S/4 top30)", "ssim"),
    ("s8_klf4_clean_dino", "s8 (f4 S/4 top30C dino)", "ssim"),
    ("s10_b4_grey_clear", "s10 (f4 B/4 top30C dino)", "ssim"),
    ("s11_top6_p4", "s11 (f4 S/4 top6 dino)", "ssim"),
]:
    r, v = best(series)
    if r:
        print(f"{label:<28}{r['model']:<16}{r['vae'].split('/')[-1]:<6}{r['data_csv'].split('/')[-1]:<20}{r['dino']:<6}{r['lr']:<9}{v:>10.4f}{r['step']:>8}")
        summ.append((label, v, int(r["step"]), r.get("lpips"), r.get("skel_iou")))

print()
print("=" * 78)
print("B. s6 vs s11 同 step 对比 (f8 vs f4, 同 eval 集, Gaussian)")
print("=" * 78)
print(f"{'step':>7}{'s6_ssim':>9}{'s11_ssim':>10}{'gap':>7}")
for step in [25000, 50000, 75000, 100000, 125000, 150000]:
    a, b = ssim_at("s6_top6_diffonly", step), ssim_at("s11_top6_p4", step)
    if a and b:
        print(f"{step:7d}{a:9.4f}{b:10.4f}{a-b:7.3f}")

print()
print("=" * 78)
print("C. ControlNet 系列 (S6 195k base warm-start, skel 3px)")
print("=" * 78)
print(f"{'exp':<32}{'lr':<8}{'max_st':<8}{'base_ssim':>10}{'ctrl_best':>10}{'@step':>8}{'ΔSSIM':>8}")
for run_sub, label in [
    ("20260822-195603", "ctrl 3px v1 (lr=3e-4)"),
    ("20260822-235734", "ctrl 3px v3 (90-100k)"),
    ("20260823-085301", "ctrl top30 scratch"),
    ("20260825-115527", "ctrl s6_v2 (lr=1e-4)"),
]:
    br, bv = best("ctrl_skel", run_sub)
    if br:
        base = float(br.get("ssim_base") or 0)
        delta = float(br.get("delta_ssim") or 0)
        print(f"{label:<32}{br['lr']:<8}{br['max_steps']:<8}{base:>10.4f}{bv:>10.4f}{br['step']:>8}{delta:>8.4f}")

print()
print("=" * 78)
print("D. DINO 消融 (s7 vs s8: 同 f4 S/4 top30, 仅 data+dino 不同)")
print("=" * 78)
print(f"{'step':>7}{'s7_nodino':>11}{'s8_dino':>10}{'gap':>7}")
for step in [25000, 50000, 75000, 100000]:
    a, b = ssim_at("s7_klf4_top30", step), ssim_at("s8_klf4_clean_dino", step)
    if a and b:
        print(f"{step:7d}{a:11.4f}{b:10.4f}{b-a:7.4f}")

print()
print("=" * 78)
print("E. s11 修正前后对比 (uniform vs gaussian 影响)")
print("=" * 78)
print(f"{'step':>7}{'gaussian':>10}{'uniform':>10}{'Δ':>8}")
for step in [10000, 50000, 100000, 145000]:
    for r in flt("s11_top6_p4", step):
        g, u = r.get("ssim"), r.get("ssim_old_uniform")
        if g and u:
            print(f"{step:7d}{float(g):10.4f}{float(u):10.4f}{float(g)-float(u):8.4f}")
            break
