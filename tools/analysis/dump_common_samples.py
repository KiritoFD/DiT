import csv

r2 = list(csv.DictReader(open('exp-std/csv/eval_top2_100.csv', encoding='utf-8-sig')))
r5 = list(csv.DictReader(open('exp-std/csv/eval_top5_fixed.csv', encoding='utf-8-sig')))
r10 = list(csv.DictReader(open('exp-std/csv/eval200_fixed.csv', encoding='utf-8-sig')))

img_to_idx_10 = {r['image_path']: i for i, r in enumerate(r10)}
img_to_idx_5 = {r['image_path']: i for i, r in enumerate(r5)}

lines = []
for i2, r in enumerate(r2):
    p = r['image_path']
    if p in img_to_idx_5 and p in img_to_idx_10:
        i5 = img_to_idx_5[p]
        i10 = img_to_idx_10[p]
        c = r['calligrapher']
        s = r['script']
        ch = r['character']
        lines.append(f"Top2: #{i2:02d} | Top5: #{i5:02d} | Top10: #{i10:03d} | {c} {s} 【{ch}】")

with open('scratch/common_samples_utf8.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))
print(f"Wrote {len(lines)} lines to scratch/common_samples_utf8.txt")
