import csv, collections, statistics, math

rows = list(csv.DictReader(open('m432_batch.csv', encoding='utf-8')))
summ = list(csv.DictReader(open('m432_summary.csv', encoding='utf-8')))

# batch: key by idx (img_id is empty in this file)
bt = collections.defaultdict(dict)
for r in rows:
    if r['set'] == 'strict':
        bt[int(r['step'])][int(r['idx'])] = float(r['ssim'])
# summary
sm = {int(r['step']): float(r['ssim_mean']) for r in summ if r['set'] == 'strict'}

steps = sorted(bt)
print("=== batch mean vs summary ssim_mean (strict) ===")
print(f"{'step':>8} {'batch':>8} {'summary':>8} {'diff':>8}")
for s in steps[::8] + steps[-6:]:
    b = statistics.mean(bt[s].values())
    print(f"{s:>8} {b:>8.4f} {sm[s]:>8.4f} {b-sm[s]:>+8.4f}")

print("\nNOTE: if diff is a constant offset, summary uses a different ssim variant.")
print("      if diff varies, the two files track different eval runs.\n")


def paired(a, b):
    ids = sorted(set(bt[a]) & set(bt[b]))
    d = [bt[b][i] - bt[a][i] for i in ids]
    n = len(d)
    m = statistics.mean(d)
    sd = statistics.stdev(d) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n else 0.0
    return n, m, se, (m / se if se > 0 else float('inf')), sum(1 for x in d if x > 0)


print("=== paired diff vs previous eval (strict, from batch) ===")
print(f"{'step':>8} {'mean':>8} {'d_mean':>8} {'se':>7} {'t':>7} {'n_up':>5}")
prev = None
for s in steps:
    if prev is None:
        print(f"{s:>8} {statistics.mean(bt[s].values()):>8.4f}")
    else:
        n, m, se, t, up = paired(prev, s)
        print(f"{s:>8} {statistics.mean(bt[s].values()):>8.4f} {m:>+8.4f} {se:>7.4f} {t:>+7.2f} {up:>3}/{n}")
    prev = s

last = steps[-1]
peak = max(steps, key=lambda s: statistics.mean(bt[s].values()))
n, m, se, t, up = paired(peak, last)
print(f"\npeak={peak} ({statistics.mean(bt[peak].values()):.4f}) -> last={last} "
      f"({statistics.mean(bt[last].values()):.4f})")
print(f"  paired d_mean={m:+.4f} se={se:.4f} t={t:+.2f}  improved {up}/{n}")
print(f"  -> {'SIGNIFICANT' if abs(t) > 2 else 'within noise (|t|<2)'}")

# noise scale
vals = list(bt[last].values())
sd = statistics.stdev(vals)
print(f"\nper-sample sd at {last} = {sd:.4f}  -> se of a SINGLE 50-sample mean = {sd/math.sqrt(50):.4f}")
print(f"per-sample range {min(vals):.3f}..{max(vals):.3f}")

# last 40k: is there any trend?
late = [s for s in steps if s >= 110000]
print(f"\n=== strict batch mean from 110k onward ({len(late)} pts) ===")
xs = [s / 1000 for s in late]
ys = [statistics.mean(bt[s].values()) for s in late]
mx, my = statistics.mean(xs), statistics.mean(ys)
slope = sum((a - mx) * (b - my) for a, b in zip(xs, ys)) / sum((a - mx) ** 2 for a in xs)
resid = [b - (my + slope * (a - mx)) for a, b in zip(xs, ys)]
sse = sum(r * r for r in resid)
sst = sum((b - my) ** 2 for b in ys)
se_slope = math.sqrt(sse / (len(xs) - 2) / sum((a - mx) ** 2 for a in xs))
print(f"  slope = {slope:+.5f} per 1k steps  (se={se_slope:.5f}, t={slope/se_slope:+.2f})  R2={1-sse/sst:.3f}")
print(f"  => {'flat / plateau' if abs(slope/se_slope) < 2 else 'real trend'}")

# seen
bs = collections.defaultdict(dict)
for r in rows:
    if r['set'] == 'seen':
        bs[int(r['step'])][int(r['idx'])] = float(r['ssim'])
sst2 = sorted(bs)
print(f"\n=== seen batch mean (last 6) ===")
for s in sst2[-6:]:
    print(f"  {s:>7} {statistics.mean(bs[s].values()):.4f}")
sl = [s for s in sst2 if s >= 110000]
xs2 = [s / 1000 for s in sl]; ys2 = [statistics.mean(bs[s].values()) for s in sl]
mx2, my2 = statistics.mean(xs2), statistics.mean(ys2)
sl2 = sum((a - mx2) * (b - my2) for a, b in zip(xs2, ys2)) / sum((a - mx2) ** 2 for a in xs2)
print(f"  seen slope from 110k = {sl2:+.5f} per 1k steps")
