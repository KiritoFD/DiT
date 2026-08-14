import re, glob, os, json

ansi = re.compile(r'\x1b\[[0-9;]*m')

def parse(path):
    rows = []
    for ln in open(path, encoding='utf-8', errors='ignore'):
        ln = ansi.sub('', ln)
        m = re.search(r'\(step=(\d+)\)\s*Total:\s*([\d.]+)\s*\|\s*Diff:\s*([\d.]+)', ln)
        if m:
            rows.append({
                'step': int(m.group(1)),
                'total': float(m.group(2)),
                'diff': float(m.group(3)),
            })
    return rows

summary = []
for f in sorted(glob.glob('remote_logs/**/log.txt', recursive=True)):
    r = parse(f)
    if not r:
        continue
    rel = os.path.relpath(f, 'remote_logs')
    # 实验名 = 去掉 results[/...]/XXX-DiT-.../log.txt 取中间段
    parts = rel.split(os.sep)
    exp = parts[-2] if len(parts) >= 2 else rel
    diffs = [x['diff'] for x in r if x['diff'] == x['diff']]
    summary.append({
        'exp': exp,
        'n': len(r),
        'step_first': r[0]['step'],
        'step_last': r[-1]['step'],
        'diff_first': r[0]['diff'],
        'diff_last': r[-1]['diff'],
        'diff_min': min(diffs),
    })

print(f"{'experiment':42s} {'n':>3s} {'step':>14s} {'first':>8s} {'last':>8s} {'min':>8s}")
for s in summary:
    print(f"{s['exp']:42s} {s['n']:3d} {s['step_first']:>6}->{s['step_last']:<6} "
          f"{s['diff_first']:8.4f} {s['diff_last']:8.4f} {s['diff_min']:8.4f}")

json.dump(summary, open('all_logs_summary.json', 'w'), indent=2)
print(f"\nTotal experiments with parsed curves: {len(summary)}")
print("Saved -> all_logs_summary.json")
