# -*- coding: utf-8 -*-
"""查看 CSV：打印列名 + 按某列排序的表格预览。

用法: python _csv_view.py <csv> [sort_col] [--desc] [--top N] [--cols a,b,c]
      [--where col=val]
"""
import sys, csv


def main():
    args = [a for a in sys.argv[1:]]
    if not args:
        print(__doc__)
        return
    path = args[0]
    rest = args[1:]
    sort_col, top, cols, desc, where = None, 20, None, True, {}
    i = 0
    while i < len(rest):
        a = rest[i]
        if a == "--top":
            top = int(rest[i+1]); i += 2
        elif a == "--cols":
            cols = rest[i+1].split(","); i += 2
        elif a == "--desc":
            desc = True; i += 1
        elif a == "--asc":
            desc = False; i += 1
        elif a == "--where":
            kv = rest[i+1].split("=", 1)
            where[kv[0]] = kv[1] if len(kv) > 1 else ""
            i += 2
        else:
            sort_col = a; i += 1

    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print("(empty)")
        return
    headers = list(rows[0].keys())
    print(f"# {path}: {len(rows)} rows")
    print(f"# cols: {', '.join(headers)}\n")

    for k, v in where.items():
        rows = [r for r in rows if v in (r.get(k) or "")]

    if cols:
        cols = [c for c in cols if c in headers]
    else:
        cols = headers[:10]

    if sort_col and sort_col in headers:
        def key(r):
            v = r.get(sort_col) or ""
            try:
                return float(v)
            except ValueError:
                return float("-inf")
        rows.sort(key=key, reverse=desc)

    rows = rows[:top]
    widths = {c: max(len(c), max((len((r.get(c) or "")) for r in rows), default=0))
              for c in cols}
    widths = {c: min(w, 34) for c, w in widths.items()}
    print("  ".join(c.ljust(widths[c])[:widths[c]] for c in cols))
    print("  ".join("-" * widths[c] for c in cols))
    for r in rows:
        print("  ".join(((r.get(c) or "")[:widths[c]]).ljust(widths[c]) for c in cols))


if __name__ == "__main__":
    main()
