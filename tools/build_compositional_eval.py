"""Build deterministic evaluation strata for 3-condition composition.

The clean stratum contains unseen triples for which all three factors and all
three pairwise edges were observed in training. This isolates triple composition
from unseen-category and unseen-pair failures.
"""

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def read_rows(path):
    with open(path, encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def key(row):
    return (int(row["calligrapher_id"]), int(row["script_id"]),
            int(row["character_id"]))


def write_rows(path, rows, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def stratified_take(rows, n, seed):
    rng = random.Random(seed)
    by_script = defaultdict(list)
    for row in rows:
        by_script[int(row["script_id"])].append(row)
    for group in by_script.values():
        rng.shuffle(group)
    result = []
    while len(result) < min(n, len(rows)):
        progressed = False
        for script_id in sorted(by_script):
            if by_script[script_id] and len(result) < n:
                result.append(by_script[script_id].pop())
                progressed = True
        if not progressed:
            break
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="5script/train.csv")
    parser.add_argument("--test", default="5script/test.csv")
    parser.add_argument("--out-dir", default="5script/eval_strata")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    train = read_rows(args.train)
    test = read_rows(args.test)
    triples = {key(row) for row in train}
    cs = {(c, s) for c, s, h in triples}
    ch = {(c, h) for c, s, h in triples}
    sh = {(s, h) for c, s, h in triples}
    factors = ({c for c, s, h in triples}, {s for c, s, h in triples},
               {h for c, s, h in triples})

    clean, unseen_pair, seen = [], [], []
    for row in test:
        c, s, h = key(row)
        if (c, s, h) in triples:
            seen.append(row)
        elif c in factors[0] and s in factors[1] and h in factors[2]:
            if (c, s) in cs and (c, h) in ch and (s, h) in sh:
                clean.append(row)
            else:
                unseen_pair.append(row)

    fields = list(test[0].keys())
    out_dir = Path(args.out_dir)
    selections = {
        "clean_unseen_triple": stratified_take(clean, args.n, args.seed),
        "unseen_pair": stratified_take(unseen_pair, args.n, args.seed + 1),
        "seen_triple": stratified_take(seen, args.n, args.seed + 2),
    }
    for name, rows in selections.items():
        write_rows(out_dir / f"{name}_{len(rows)}.csv", rows, fields)

    summary = {
        "train_rows": len(train),
        "test_rows": len(test),
        "candidate_counts": {
            "clean_unseen_triple": len(clean),
            "unseen_pair": len(unseen_pair),
            "seen_triple": len(seen),
        },
        "selected_counts": {name: len(rows) for name, rows in selections.items()},
        "selected_script_counts": {
            name: dict(Counter(row["script_id"] for row in rows))
            for name, rows in selections.items()
        },
        "seed": args.seed,
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
