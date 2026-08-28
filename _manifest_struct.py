import json, sys
sys.stdout.reconfigure(encoding="utf-8")
man = json.load(open("archive/final_manifest.json", encoding="utf-8"))
print("type:", type(man).__name__, "len:", len(man))
e0 = man[0]
print("sample keys:", list(e0.keys())[:20])
print("sample entry:", {k: e0[k] for k in list(e0.keys())[:10]})
import collections
snames = collections.Counter(e.get("orig_script", "") for e in man)
print("orig_script dist:", dict(snames))
# id range and format of img_id
print("first ids:", [e["img_id"] for e in man[:5]])
# check orig_calli presence
callis = collections.Counter(e.get("orig_calli", "") for e in man)
print("unique orig_calli:", len(callis), "top:", callis.most_common(15))