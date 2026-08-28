import csv, sys
sys.stdout.reconfigure(encoding="utf-8")

def ids(path):
    s = set()
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            s.add(int(r["image_path"].rsplit("/", 1)[1].replace(".png", "")))
    return s

def names(tag, path):
    ov = ids(path) & ids("5script/train_top6.csv")
    print(tag, "rows:", len(list(csv.DictReader(open(path, encoding="utf-8")))))
    print(tag, "overlap with train_top6:", len(ov))

names("eval500_top6", "5script/eval500_top6.csv")
names("eval100_top6", "5script/eval100_top6.csv")
names("eval_unseen_top6", "5script/eval_unseen_top6.csv")

# check (script,calli) and (script,char) coverage in train for eval500_top6
train_tuples_ca = set()
train_tuples_ch = set()
with open("5script/train_top6.csv", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        train_tuples_ca.add((r["script"], r["calligrapher"]))
        train_tuples_ch.add((r["script"], r["character"]))
for tag, path in [("eval500_top6", "5script/eval500_top6.csv")]:
    bad_ca = bad_ch = 0
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if (r["script"], r["calligrapher"]) not in train_tuples_ca:
                bad_ca += 1
            if (r["script"], r["character"]) not in train_tuples_ch:
                bad_ch += 1
    print(tag, "calli-not-in-train:", bad_ca, "char-not-in-train:", bad_ch)
# scale: scripts dist
import collections
with open("5script/eval500_top6.csv", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
print("eval500_top6 scripts:", dict(collections.Counter(r["script"] for r in rows)))
print("eval500_top6 callis:", sorted(set(r["calligrapher"] for r in rows)))