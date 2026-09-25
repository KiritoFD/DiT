"""cosine with warmup, matching src/train/train.py."""
import math

def scale(step, total, warmup, min_ratio):
    warmup = min(warmup, max(total - 1, 0))
    if warmup > 0 and step < warmup:
        return max((step + 1) / warmup, 1e-8)
    progress = (step - warmup) / max(total - warmup, 1)
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_ratio + (1.0 - min_ratio) * cosine

base, warmup, ratio = 5e-4, 3000, 0.1
print(f"{'step':>7} {'E0@100k':>10} {'mid@40k':>10} {'E0/mid':>8}")
for s in (5000, 10000, 15000, 20000, 25000, 30000, 35000, 37800, 40000):
    a = base * scale(s, 100000, warmup, ratio)
    b = base * scale(s, 40000, warmup, ratio)
    print(f"{s:7d} {a:10.3e} {b:10.3e} {a/b:8.2f}")
