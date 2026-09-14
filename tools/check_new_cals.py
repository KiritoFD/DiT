#!/opt/conda/envs/cu121/bin/python
import os, json
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_TOKEN"] = "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"
from huggingface_hub import hf_hub_download
from collections import Counter, defaultdict

bei = json.load(open(hf_hub_download("Tongji209/HCSU", "bei_annotations.json", repo_type="dataset")))
tie = json.load(open(hf_hub_download("Tongji209/HCSU", "tie_annotations.json", repo_type="dataset")))

# Our calligraphers
import csv
rows = list(csv.DictReader(open("assets/train_base_noaug.csv", encoding="utf-8")))
our_cals = set(r["calligrapher"] for r in rows)

# New calligraphers (in HCSU but not in us)
all_hcsu = bei + tie
new_cals = set(i["calligrapher"] for i in all_hcsu) - our_cals

# Count images & unique chars for new calligraphers
cal_data = defaultdict(lambda: {"images": 0, "chars": set(), "scripts": set(), "quality": Counter()})
for item in all_hcsu:
    if item["calligrapher"] in new_cals:
        cal = item["calligrapher"]
        cal_data[cal]["images"] += 1
        cal_data[cal]["chars"].add(item["char"])
        cal_data[cal]["scripts"].add(item["script_type"])
        cal_data[cal]["quality"][item.get("quality","?")] += 1

print(f"New calligraphers: {len(new_cals)}")
print(f"{'name':<10} {'imgs':>5} {'chars':>5} {'scripts':>8} {'clear':>5} {'blurry':>6}")
print("-" * 50)
for cal in sorted(cal_data.keys()):
    d = cal_data[cal]
    print(f"{cal:<10} {d['images']:>5} {len(d['chars']):>5} {','.join(sorted(d['scripts'])):>8} {d['quality'].get('clear',0):>5} {d['quality'].get('blurry',0):>6}")
print("-" * 50)
total_imgs = sum(d["images"] for d in cal_data.values())
total_clear = sum(d["quality"].get("clear",0) for d in cal_data.values())
total_blurry = sum(d["quality"].get("blurry",0) for d in cal_data.values())
print(f"{'TOTAL':<10} {total_imgs:>5} {'':>5} {'':>8} {total_clear:>5} {total_blurry:>6}")
