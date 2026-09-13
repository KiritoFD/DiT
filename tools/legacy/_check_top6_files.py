"""Verify: (1) all PNGs in the 4 top6 CSVs exist under final_imgs_256;
(2) max glyph_id / script_id fit in num_characters=35130 vocab."""
import os, sys, csv, re
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.chdir("/root/Workspace/xy/DiT")

for csv_path in ["5script/train_top6.csv", "5script/eval100_top6.csv",
                 "5script/show2_top6.csv", "5script/seen2_top6.csv"]:
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    n = len(rows)
    missing = []
    max_gid = -1
    max_sid = -1
    max_cid = -1
    for r in rows:
        m = re.search(r"(\d+)\.png$", r["image_path"])
        if not m:
            continue
        fn = f"{m.group(1)}.png"
        full = os.path.join("final_imgs_256", fn)
        if not os.path.exists(full):
            missing.append(r["image_path"])
        max_gid = max(max_gid, int(r.get("glyph_id", 0)))
        max_sid = max(max_sid, int(r.get("script_id", 0)))
        max_cid = max(max_cid, int(r.get("character_id", 0)))
    print(f"{os.path.basename(csv_path)}: n={n} missing_png={len(missing)} "
          f"max_glyph_id={max_gid} max_script_id={max_sid} max_char_id={max_cid}")
    if missing[:3]:
        print(f"    e.g. {missing[:3]}")
    if max_gid + 1 > 35130:
        print(f"    !! glyph_id exceeds vocab 35130 (need {max_gid + 2} embed rows incl uncond)")