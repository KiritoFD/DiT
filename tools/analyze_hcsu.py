#!/opt/conda/envs/cu121/bin/python
import os, json
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_TOKEN"] = "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"
from huggingface_hub import HfApi
api = HfApi()
files = list(api.list_repo_tree("Tongji209/HCSU", repo_type="dataset", recursive=True))

# Separate into top-level categories
pngs = {}
jsons = {}
zips = {}
others = {}
for f in files:
    p = f.path
    if p.endswith(".png"):
        parts = p.split("/")
        if len(parts) >= 3:
            category = parts[0]  # bei_samples / tie_samples
            cal_script = parts[1]  # 于右任-楷
            char_name = parts[2].replace(".png", "")
            if category not in pngs:
                pngs[category] = {}
            if cal_script not in pngs[category]:
                pngs[category][cal_script] = []
            pngs[category][cal_script].append(char_name)
    elif p.endswith(".json"):
        jsons[p] = f.size if hasattr(f, 'size') else 0
    elif p.endswith(".zip") or p.endswith(".001"):
        zips[p] = f.size if hasattr(f, 'size') else 0

print("=" * 60)
print("HCSU Dataset Analysis")
print("=" * 60)

total_chars = 0
all_cals = set()
all_scripts = set()

for cat, cals in sorted(pngs.items()):
    print(f"\n--- {cat} ---")
    n_cals = len(cals)
    n_images = sum(len(v) for v in cals.values())
    print(f"  Calligrapher-Script combos: {n_cals}")
    print(f"  Total images: {n_images}")
    
    cal_set = set()
    script_set = set()
    for cs in cals:
        if "-" in cs:
            cal, script = cs.rsplit("-", 1)
            cal_set.add(cal)
            script_set.add(script)
            all_cals.add(cal)
            all_scripts.add(script)
    
    print(f"  Unique calligraphers: {len(cal_set)}")
    print(f"  Scripts: {sorted(script_set)}")
    print(f"  Images per calligrapher-script (min/max/avg):", end=" ")
    counts = [len(v) for v in cals.values()]
    print(f"{min(counts)}/{max(counts)}/{sum(counts)/len(counts):.0f}")
    
    # Per-calligrapher breakdown
    for cs in sorted(cals.keys(), key=lambda x: -len(cals[x]))[:5]:
        print(f"    {cs}: {len(cals[cs])} chars")
    if len(cals) > 5:
        print(f"    ... and {len(cals)-5} more")
    
    total_chars += n_images

print(f"\n{'=' * 60}")
print(f"TOTAL: {total_chars} images")
print(f"Unique calligraphers: {len(all_cals)}")
print(f"Unique scripts: {sorted(all_scripts)}")
print(f"Calligraphers: {sorted(all_cals)}")
print(f"\nJSON files: {jsons}")
print(f"ZIP/part files: {zips}")
