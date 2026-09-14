#!/opt/conda/envs/cu121/bin/python
import os, json
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_TOKEN"] = "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"
from huggingface_hub import hf_hub_download
from collections import Counter

bei = json.load(open(hf_hub_download("Tongji209/HCSU", "bei_annotations.json", repo_type="dataset")))
tie = json.load(open(hf_hub_download("Tongji209/HCSU", "tie_annotations.json", repo_type="dataset")))

print(f"bei: {len(bei)} annotations, {len(set(i['calligrapher'] for i in bei))} calligraphers")
print(f"tie: {len(tie)} annotations, {len(set(i['calligrapher'] for i in tie))} calligraphers")
print(f"Total: {len(bei)+len(tie)} annotations")

bei_q = Counter(i.get("quality","?") for i in bei)
tie_q = Counter(i.get("quality","?") for i in tie)
print(f"bei quality: {dict(bei_q)}")
print(f"tie quality: {dict(tie_q)}")

ours = ["赵孟頫","颜真卿","柳公权","欧阳询","虞世南","褚遂良","王羲之","王献之","黄庭坚","米芾","苏轼","董其昌","怀素","张旭","智永","赵之谦","邓石如","何绍基","伊秉绶","吴昌硕","傅山"]
bei_cals = set(i["calligrapher"] for i in bei)
tie_cals = set(i["calligrapher"] for i in tie)
overlap_bei = bei_cals & set(ours)
overlap_tie = tie_cals & set(ours)
print(f"\nOverlap with our data:")
print(f"  bei: {len(overlap_bei)}/{len(ours)} of ours ({sorted(overlap_bei)})")
print(f"  tie: {len(overlap_tie)}/{len(ours)} of ours ({sorted(overlap_tie)})")
print(f"  New in bei (not in ours): {sorted(bei_cals - set(ours))}")
print(f"  New in tie (not in ours): {sorted(tie_cals - set(ours))}")
