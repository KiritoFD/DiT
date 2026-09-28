# -*- coding: utf-8 -*-
"""Build DG-Font data: baseline/data/dgfont/train/{domain}/{char}.png

24 domains = 23 style slots + content_deng. Files named by the character itself
so cross-domain char alignment is explicit. Rows without a content-font render
are skipped (same 5-char exclusion as other baselines).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from top10_common import DIT_ROOT, load_train_rows, content_png, renderable_chars  # noqa: E402

OUT = os.path.join(DIT_ROOT, "baseline", "data", "dgfont", "train")


def link(src, dst):
    if os.path.exists(dst):
        return
    try:
        os.symlink(os.path.abspath(src), dst)
    except OSError:
        os.link(os.path.abspath(src), dst)


def main():
    rows = load_train_rows()
    ok = renderable_chars()
    import json
    import shutil
    n = 0
    skipped = 0

    # DG-Font's ImageFolerRemap sorts class dirs by int(name[3:]) -> use id_XX
    slots = sorted({r["slot_name"] for r in rows},
                   key=lambda s: min(int(r["pair_id"]) for r in rows if r["slot_name"] == s))
    domain_map = {f"id_{i:02d}": s for i, s in enumerate(slots)}
    domain_map[f"id_{len(slots):02d}"] = "content_deng"

    root = OUT
    os.makedirs(root, exist_ok=True)
    for d in os.listdir(root):          # cleanup legacy slot-named dirs
        if not (d.startswith("id_") and d[3:].isdigit()):
            shutil.rmtree(os.path.join(root, d), ignore_errors=True)

    for domain, slot in domain_map.items():
        d = os.path.join(root, domain)
        os.makedirs(d, exist_ok=True)
        if slot == "content_deng":
            for ch in ok:
                link(content_png(ch), os.path.join(d, f"{ch}.png"))
            continue
        for r in rows:
            if r["slot_name"] != slot:
                continue
            ch = r["character"]
            if ch not in ok:
                skipped += 1
                continue
            link(os.path.join(DIT_ROOT, r["image_path"]), os.path.join(d, f"{ch}.png"))
            n += 1

    with open(os.path.join(root, "domain_map.json"), "w", encoding="utf-8") as f:
        json.dump(domain_map, f, ensure_ascii=False, indent=1)
    print(f"linked {n} style imgs across {len(domain_map)} domains "
          f"(content={len(ok)} imgs, skipped {skipped} rows)")


if __name__ == "__main__":
    main()
