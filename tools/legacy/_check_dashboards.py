import json, re
f = "s10_b4_grey_clear"
t = open("G:/GitHub/DiT/tools/dashboards/" + f + ".html", encoding="utf-8").read()
m = re.search(r"const __DATA__ = (\{.*?\});\nconst COLORS", t, re.S)
d = json.loads(m.group(1))
src = d.get("source", "")[:70]
n = len(d.get("rows", []))
last = d["rows"][-1].get("step") if d.get("rows") else "?"
ds = d.get("dataset_size")
bn = d.get("batch")
pm = re.search(r"li\.src='([^']+)'", t)
poster = pm.group(1) if pm else "?"
print(f + ": " + str(n) + " rows | step " + str(last) + " | ds=" + str(ds) + " batch=" + str(bn) + " | poster=" + poster + " | " + src)
