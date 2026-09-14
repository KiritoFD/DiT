#!/opt/conda/envs/cu121/bin/python
import os, json, csv
from collections import defaultdict

BASE = "/root/Workspace/xy/HCSU"

# Our calligraphers
rows = list(csv.DictReader(open("/root/Workspace/xy/DiT/assets/train_base_noaug.csv", encoding="utf-8")))
our_cals = set(r["calligrapher"] for r in rows)

# HCSU annotations
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_TOKEN"] = "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"
from huggingface_hub import hf_hub_download
bei_ann = json.load(open(hf_hub_download("Tongji209/HCSU", "bei_annotations.json", repo_type="dataset")))
tie_ann = json.load(open(hf_hub_download("Tongji209/HCSU", "tie_annotations.json", repo_type="dataset")))

# Build ann lookup: filename -> annotation
ann_by_name = {}
for item in bei_ann + tie_ann:
    ann_by_name[item["filename"]] = item

# Walk extracted directories and match by filename
for dataset, ann_list in [("bei", bei_ann), ("tie", tie_ann)]:
    d = os.path.join(BASE, dataset)
    if not os.path.isdir(d):
        continue
    
    # Build ann index by calligrapher
    cal_ann = defaultdict(lambda: {"chars": set(), "scripts": set(), "clear": 0, "blurry": 0, "total": 0})
    for item in ann_list:
        cal = item["calligrapher"]
        cal_ann[cal]["chars"].add(item["char"])
        cal_ann[cal]["scripts"].add(item["script_type"])
        cal_ann[cal]["total"] += 1
        if item.get("quality") == "clear":
            cal_ann[cal]["clear"] += 1
        elif item.get("quality") == "blurry":
            cal_ann[cal]["blurry"] += 1
    
    print(f"\n{'='*65}")
    print(f"{dataset}: {sum(d2['total'] for d2 in cal_ann.values())} annotations")
    print(f"{'='*65}")
    print(f"{'Calligrapher':<14} {'Anns':>5} {'Chars':>6} {'Scripts':>12} {'Clear':>6} {'Blurry':>7} {'In ours?':>8}")
    print("-" * 70)
    for cal in sorted(cal_ann.keys()):
        d2 = cal_ann[cal]
        in_ours = "YES" if cal in our_cals else ""
        print(f"{cal:<14} {d2['total']:>5} {len(d2['chars']):>6} {','.join(sorted(d2['scripts'])):>12} {d2['clear']:>6} {d2['blurry']:>7} {in_ours:>8}")
    
    hcsu_cals = set(cal_ann.keys())
    overlap = hcsu_cals & our_cals
    new_cals = sorted(hcsu_cals - our_cals)
    print(f"\nOverlap with us: {len(overlap)}/{len(hcsu_cals)}")
    print(f"New calligraphers ({len(new_cals)}): {new_cals}")
    
    # Summary for new calligraphers
    print(f"\n--- New calligrapher details ---")
    for cal in new_cals:
        d2 = cal_ann[cal]
        print(f"  {cal}: {d2['total']} imgs, {len(d2['chars'])} chars, clear={d2['clear']}, blurry={d2['blurry']}, scripts={sorted(d2['scripts'])}")
