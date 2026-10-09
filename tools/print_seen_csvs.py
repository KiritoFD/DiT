import csv

def show(path):
    print("=== " + path + " ===")
    with open(path, "r", encoding="utf-8") as f:
        for i, r in enumerate(csv.DictReader(f)):
            ch = r.get("character") or r.get("char")
            cal = r.get("calligrapher")
            sc = r.get("script")
            print(f"[{i:02d}] {ch} | {cal} | {sc}")

show("assets/eval_seen_v10.csv")
print()
show("assets/eval_v13_seen.csv")
print()
show("assets/eval_top10_seen_20.csv")
