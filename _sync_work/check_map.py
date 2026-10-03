import csv, json, collections, os
os.chdir("/root/Workspace/xy/DiT")

base = json.load(open("assets/callig_id_map_base.json", encoding="utf-8"))
new = json.load(open("assets/callig_id_map_hcsu.json", encoding="utf-8"))
bm, nm = base["id_map"], new["id_map"]
print(f"base num={base['num_calligraphers']} len={len(bm)}")
print(f"new  num={new['num_calligraphers']} len={len(nm)}")
print(f"新增槽位: {sorted(set(nm) - set(bm))}")

old = list(csv.DictReader(open("assets/train_fame-kxl-tj-px60.csv", encoding="utf-8")))
fin = list(csv.DictReader(open("assets/train_hcsu_kxl_final.csv", encoding="utf-8")))
oc = {r["calligrapher"] for r in old}
nc = {r["calligrapher"] for r in fin}
newc = sorted(nc - oc)
print(f"\n新 CSV 里新书家 {len(newc)}: {newc}")

# 每个新书家的 id 与是否拿到槽位
cid = {}
for r in fin:
    cid[r["calligrapher"]] = r["calligrapher_id"]
cnt = collections.Counter(r["calligrapher"] for r in fin)
ch = collections.defaultdict(set)
for r in fin:
    ch[r["calligrapher"]].add(r["character"])

print(f"\n{'书家':<12}{'id':>7}{'张数':>7}{'字数':>7}  槽位")
for c in newc:
    i = cid[c]
    print(f"{c:<12}{i:>7}{cnt[c]:>7}{len(ch[c]):>7}  {nm.get(str(i), '**无槽位**')}")
