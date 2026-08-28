import csv
for p in ["5script/train_top6.csv", "5script/eval500_top6.csv", "5script/eval_unseen_top6.csv"]:
    with open(p, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    scripts = set(r["script"] for r in rows)
    callis = set(r["calligrapher"] for r in rows)
    chars = set(r["character"] for r in rows)
    print(f"{p}: {len(rows)} rows, {len(chars)} chars, {len(callis)} callis, {len(scripts)} scripts:{scripts}")
    prefixes = set(r["image_path"].split("/")[0] for r in rows)
    print(f"  img_prefixes: {prefixes}")
    for r in rows[:3]:
        print("  eg:", r["image_path"], "| script=", r["script"], "| calli=", r["calligrapher"], "| char=", r["character"])
