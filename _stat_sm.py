"""对比 S(small=top6) vs M(mid=3top30) 的生僻字比例与稀疏度."""
import csv
from collections import Counter, defaultdict

def gb2312_chars():
    chars = set()
    for q in range(0xB0, 0xF8):
        for p in range(0xA1, 0xFF):
            try:
                chars.add(bytes([q, p]).decode("gb2312"))
            except UnicodeDecodeError:
                pass
    return chars

KEEP = gb2312_chars()

def analyze(tag, path):
    n_rows = 0
    per_char = Counter()
    combo = Counter()
    callis = set()
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            n_rows += 1
            per_char[r["character"]] += 1
            combo[(r["script"], r["character"], r["calligrapher"])] += 1
            callis.add(r["calligrapher"])
    inL1 = sum(1 for c in per_char if c in KEEP)
    rowsL1 = sum(per_char[c] for c in per_char if c in KEEP)
    # hmm: L1 vs L2 distinction needs level sets; keep it simple: common = GB2312, rare = outside
    n_rare_chars = sum(1 for c in per_char if c not in KEEP)
    n_rare_rows = sum(per_char[c] for c in per_char if c not in KEEP)
    comb_by_size = Counter(combo.values())
    char_by_size = Counter(per_char.values())
    print(f"===== {tag}: {path} =====")
    print(f"rows={n_rows} chars={len(per_char)} calligraphers={len(callis)} combos={len(combo)}")
    print(f"  常见字(GB2312): {len(per_char)-n_rare_chars} chars / {n_rows-n_rare_rows} rows "
          f"({(n_rows-n_rare_rows)/n_rows*100:.1f}%)")
    print(f"  生僻字(国标外): {n_rare_chars} chars / {n_rare_rows} rows "
          f"({n_rare_rows/n_rows*100:.1f}%)")
    print(f"  combo 大小分布(样本数): " + ", ".join(f"{k}个:{v}" for k, v in sorted(comb_by_size.items())))
    print(f"  每字符样本数分布: " + ", ".join(f"{k}:{v}" for k, v in sorted(char_by_size.items())))
    # per-combo sparsity stats
    singles = comb_by_size.get(1, 0)
    print(f"  单样本组合占比: {singles/len(combo)*100:.1f}% ({singles}/{len(combo)})")
    # mean samples per combo
    print(f"  平均每组合样本: {n_rows/len(combo):.2f}, 平均每字样本: {n_rows/len(per_char):.2f}")
    print()

for tag, path in [("S(small=top6)", "5script/train_top6.csv"),
                  ("M(mid=3top30)", "5script/train_3top30_nobeike.csv")]:
    try:
        analyze(tag, path)
    except Exception as e:
        print(f"{tag}: ERROR {e}")
        import traceback; traceback.print_exc()