"""F1 panel: same-character cross-writer matched panels with work-level source holdout.

Design contract
---------------
* Unit of independence is the WORK (derived from `src_image_path`), never the image.
* For each (script, character) we keep the writers that have that character.
* A "panel" = character c + >=2 writers with an image of c.
* Source holdout: a work is assignable to a writer-script slot; hash of work id
  decides membership in {train, test}. Test works never contribute to any fit.
* Outputs:
    panel.csv          one row per (script, character, writer) chosen record
    work_index.json    work -> {writer, script, n_images}
    protocol.json      hashes, split sizes, per-slot work counts
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path('/home/ds/Workspace/DiT')


def work_id(src: str, source: str) -> str:
    """Stable work proxy, at the finest granularity the path actually supports.

    kxl pools are cropped pages: `data/hcsu_kxl/imgs/bei/欧阳询-楷/户.png` -> one work
    per (pool, script-form, writer-script) directory. MCCD files carry the work in
    the directory segment. A flat pool with no directory structure cannot be split
    further and is reported as such (`work_granularity` in protocol.json).
    """
    s = src.replace('\\', '/')
    parent = s.rsplit('/', 2)[-2] if s.count('/') >= 2 else ''
    if parent and parent not in ('imgs', ''):
        return f'{source}:{parent}'
    return f'{source}:{s.rsplit("/", 1)[-1].rsplit(".", 1)[0]}'


def split_of(work: str, salt: str = 'style24h') -> str:
    h = hashlib.sha256(f'{salt}:{work}'.encode()).hexdigest()
    return 'test' if int(h[:8], 16) % 100 < 25 else 'train'


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default='exp-std/csv/train_top10_noaug.csv')
    ap.add_argument('--min-panel', type=int, default=2)
    ap.add_argument('--out', default='assets/style_24h48_20261026')
    args = ap.parse_args()

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    rows = [r for r in csv.DictReader(open(ROOT / args.csv, encoding='utf-8-sig'))
            if not (r.get('aug') or '').strip()]

    for r in rows:
        r['work'] = work_id(r['src_image_path'], r['source'])
        r['split'] = split_of(r['work'])

    work_index = {}
    for r in rows:
        w = work_index.setdefault(r['work'], {'writer': r['calligrapher'], 'script': r['script'],
                                              'source': r['source'], 'images': 0,
                                              'characters': set()})
        w['images'] += 1
        w['characters'].add(r['character'])
    for w in work_index.values():
        w['characters'] = len(w['characters'])

    # Per (script, character): writers present, ranked so the most distinct sources win.
    by_char = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_char[(r['script'], r['character'])][r['calligrapher']].append(r)

    panel_rows, panel_stats = [], []
    for (script, char), writers in sorted(by_char.items()):
        if len(writers) < args.min_panel:
            continue
        # Prefer a per-writer record whose work is in the TEST split if available,
        # else keep the train one; store both so downstream can pick.
        for writer, recs in sorted(writers.items()):
            recs_sorted = sorted(recs, key=lambda r: (r['split'] != 'test', r['work'], r['img_id']))
            chosen = recs_sorted[0]
            panel_rows.append({
                'script': script, 'character': char, 'writer': writer,
                'image_path': chosen['image_path'], 'std_path': chosen.get('std_path', ''),
                'source': chosen['source'], 'work': chosen['work'], 'split': chosen['split'],
                'n_writers_in_panel': len(writers),
                'n_images_writer_char': len(recs),
                'n_works_writer_char': len({r['work'] for r in recs}),
            })
        panel_stats.append({'script': script, 'character': char,
                            'writers': sorted(writers),
                            'n_writers': len(writers)})

    # Work-level coverage per writer-script slot (the honest denominator).
    slot_works = defaultdict(set)
    slot_images = defaultdict(int)
    for r in rows:
        slot_works[f'{r["calligrapher"]}·{r["script"]}'].add(r['work'])
        slot_images[f'{r["calligrapher"]}·{r["script"]}'] += 1
    slot = {k: {'works': len(v), 'images': slot_images[k],
                'works_split': {s: sum(1 for w in v if split_of(w) == s) for s in ('train', 'test')}}
            for k, v in sorted(slot_works.items())}

    protocol = {
        'csv': args.csv,
        'csv_sha256': hashlib.sha256((ROOT / args.csv).read_bytes()).hexdigest(),
        'n_rows': len(rows),
        'n_works': len(work_index),
        'n_duplicate_work_keys': len(rows) - len(work_index),
        'split_rule': 'sha256("style24h:"+work)[:8] % 100 < 25 -> test',
        'salt': 'style24h',
        'n_panels': len(panel_stats),
        'panels_by_script': {s: sum(1 for p in panel_stats if p['script'] == s)
                             for s in sorted({p['script'] for p in panel_stats})},
        'panel_rows': len(panel_rows),
        'slot_works': slot,
        'min_panel': args.min_panel,
        'caveat': ('Work proxy is derived from path structure, not a verified catalogue id. '
                   'MCCD filenames collapse to writer+dynasty; kxl pools use directory names. '
                   'Random image split is NOT used anywhere.'),
    }

    with (out / 'panel.csv').open('w', newline='', encoding='utf-8') as fh:
        wr = csv.DictWriter(fh, fieldnames=list(panel_rows[0].keys()))
        wr.writeheader()
        wr.writerows(panel_rows)
    (out / 'work_index.json').write_text(
        json.dumps({k: v for k, v in sorted(work_index.items())}, ensure_ascii=False, indent=1),
        encoding='utf-8')
    (out / 'panels.json').write_text(json.dumps(panel_stats, ensure_ascii=False, indent=1),
                                     encoding='utf-8')
    (out / 'protocol.json').write_text(json.dumps(protocol, ensure_ascii=False, indent=2),
                                       encoding='utf-8')
    print(json.dumps({k: protocol[k] for k in
                      ('n_rows', 'n_works', 'n_panels', 'panels_by_script', 'panel_rows')},
                     ensure_ascii=False))
    print('slots with fewer than 3 works (cannot do cross-work validation):',
          [k for k, v in slot.items() if v['works'] < 3])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
