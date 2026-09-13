import numpy as np, csv, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

embeds = np.load('tools/dino_analysis/embeddings.npy')
meta = []
with open('tools/dino_analysis/metadata.csv', encoding='utf-8') as f:
    for r in csv.DictReader(f):
        meta.append(r)
chars = [m['character'] for m in meta]
calligs = [m['calligrapher'] for m in meta]

en = embeds / (np.linalg.norm(embeds, axis=1, keepdims=True) + 1e-8)
sim = en @ en.T

# 同字 vs 不同字
same, diff = [], []
for i in range(len(en)):
    for j in range(i+1, len(en)):
        if chars[i] == chars[j] and calligs[i] != calligs[j]:
            same.append(sim[i,j])
        elif chars[i] != chars[j]:
            diff.append(sim[i,j])

print(f'same_char_diff_callig: {np.mean(same):.4f} +/- {np.std(same):.4f}  n={len(same)}')
print(f'diff_char:             {np.mean(diff):.4f} +/- {np.std(diff):.4f}  n={len(diff)}')
print(f'gap = {np.mean(same) - np.mean(diff):.4f}')

# 同书家不同字
same_callig_diff_char = []
for i in range(len(en)):
    for j in range(i+1, len(en)):
        if calligs[i] == calligs[j] and chars[i] != chars[j]:
            same_callig_diff_char.append(sim[i,j])
if same_callig_diff_char:
    print(f'same_callig_diff_char: {np.mean(same_callig_diff_char):.4f} +/- {np.std(same_callig_diff_char):.4f}  n={len(same_callig_diff_char)}')

# 每个字的组内
from collections import defaultdict
groups = defaultdict(list)
for i, c in enumerate(chars):
    groups[c].append(i)
print()
print('per-char intra-group similarity:')
for c in sorted(groups.keys()):
    idx = groups[c]
    if len(idx) < 2:
        continue
    s = []
    for ii in range(len(idx)):
        for jj in range(ii+1, len(idx)):
            s.append(sim[idx[ii], idx[jj]])
    inter = []
    for i in idx:
        for j in range(len(en)):
            if j not in idx and chars[j] != c:
                inter.append(sim[i,j])
    print(f'  {c} (U+{ord(c):05X}): intra={np.mean(s):.3f} inter={np.mean(inter):.3f} gap={np.mean(s)-np.mean(inter):.3f}')
