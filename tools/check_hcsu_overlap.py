#!/opt/conda/envs/cu121/bin/python
import csv, json, os
from collections import Counter

# Our base calligraphers
rows = list(csv.DictReader(open("assets/train_base_noaug.csv", encoding="utf-8")))
our_cals = Counter(r["calligrapher"] for r in rows)
print(f"Our base: {len(our_cals)} calligraphers, {len(rows)} images")
print(f"  {sorted(our_cals.keys())}")

# HCSU
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_TOKEN"] = "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"
from huggingface_hub import hf_hub_download
bei = json.load(open(hf_hub_download("Tongji209/HCSU", "bei_annotations.json", repo_type="dataset")))
tie = json.load(open(hf_hub_download("Tongji209/HCSU", "tie_annotations.json", repo_type="dataset")))
hcsu_cals = set(i["calligrapher"] for i in bei + tie)

overlap = set(our_cals.keys()) & hcsu_cals
only_ours = set(our_cals.keys()) - hcsu_cals
only_hcsu = hcsu_cals - set(our_cals.keys())

print(f"\nOverlap: {len(overlap)}/{len(our_cals)} of ours = {sorted(overlap)}")
print(f"Only in ours (not HCSU): {sorted(only_ours)}")
print(f"Only in HCSU (not ours): {sorted(only_hcsu)}")
print(f"\nHCSU has {len(hcsu_cals)} unique calligraphers")
print(f"  Total images: {len(bei)+len(tie)} (bei={len(bei)}, tie={len(tie)})")
print(f"  Quality clear: {sum(1 for i in bei+tie if i.get('quality')=='clear')}")
print(f"  Quality blurry: {sum(1 for i in bei+tie if i.get('quality')=='blurry')}")
