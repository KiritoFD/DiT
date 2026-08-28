import csv, json, sys
from collections import Counter, defaultdict
sys.stdout.reconfigure(encoding="utf-8")

man = json.load(open("archive/final_manifest.json", encoding="utf-8"))
id2e = {e["img_id"]: e for e in man}

# train_top6 id + name tuples
train_ids = set()
train_ca = set()   # (script, calli)
train_ch = set()   # (script, char)
train_ca_id = {}   # (script, calli) -> calligrapher_id
train_ch_id = {}   # (script, char) -> character_id
with open("5script/train_top6.csv", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        iid = int(r["image_path"].rsplit("/", 1)[1].replace(".png", ""))
        train_ids.add(iid)
        sk = (r["script"], r["calligrapher"])
        gk = (r["script"], r["character"])
        train_ca.add(sk); train_ch.add(gk)
        if sk not in train_ca_id: train_ca_id[sk] = int(r["calligrapher_id"])
        if gk not in train_ch_id: train_ch_id[gk] = int(r["character_id"])
print("train_top6 imgs:", len(train_ids))
print("train (script,calli) pairs:", len(train_ca), " (script,char) pairs:", len(train_ch))

SCRIPT_NAME_TO_ID = {"楷": 0, "隶": 4}
NUM_CHARACTERS = 7026

# candidates: orig_script in {楷,隶}, img not in train, (script,char) in train, (script,calli) in train
cand = []
by_script = defaultdict(list)
for iid, e in id2e.items():
    if iid in train_ids:
        continue
    sname = e.get("orig_script", "")
    if sname not in SCRIPT_NAME_TO_ID:
        continue
    sk = (sname, e.get("orig_char", ""))
    ck = (sname, e.get("orig_calli", ""))
    if sk not in train_ch or ck not in train_ca:
        continue
    cand.append((iid, e, sk, ck))
    by_script[sname].append((iid, e, sk, ck))
print("candidates:", len(cand), dict(Counter(e["orig_script"] for _, e, _, _ in cand)))
# existing eval100_top6 overlap with train (strictness check)
e100_ids = set()
with open("5script/eval100_top6.csv", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        e100_ids.add(int(r["image_path"].rsplit("/", 1)[1].replace(".png", "")))
print("eval100_top6 rows:", len(e100_ids), "overlap train:", len(e100_ids & train_ids))
e500_ids = set()
with open("5script/eval500_top6.csv", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        e500_ids.add(int(r["image_path"].rsplit("/", 1)[1].replace(".png", "")))
print("eval500_top6 rows:", len(e500_ids), "overlap train:", len(e500_ids & train_ids))
# candidate glyph/calli diversity
for sname in ["楷", "隶"]:
    sub = by_script[sname]
    print(f"  {sname}: {len(sub)} cand, {len(set(e['orig_char'] for _,e,_,_ in sub))} unique chars, "
          f"{len(set(e['orig_calli'] for _,e,_,_ in sub))} unique callis")