import csv

with open("/root/Workspace/xy/DiT/assets/train_50k_v2.csv", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for i, r in enumerate(reader):
        if i < 5:
            print(f"Row {i}: img_id={r.get('img_id')}, char={r.get('char') or r.get('character')}, callig={r.get('calligrapher')}, script={r.get('script')}, path={r.get('image_path') or r.get('path')}")
            for k in r.keys():
                if "skel" in k or "std" in k:
                    print(f"   {k} -> {r[k]}")
        else:
            break
