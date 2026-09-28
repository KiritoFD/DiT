# -*- coding: utf-8 -*-
"""Shared helpers for top10_style23 baseline adaptation.

Data contract (v24 same-protocol, full parity):
  - Training rows: ALL 38,583 rows of assets/train_top10_style23.csv
  - Eval A: assets/eval_top10_strict_subset84.csv (images from data/50k/imgs, char-level 79/84 seen)
  - Eval B: assets/eval_top10_seen_20.csv   (images ARE top10 training images)
  - Content font: Deng.ttf (5,850/5,855 coverage); chars it misses are Ext-B/I
    and none appear in any eval set.
"""
import csv
import hashlib
import os
import random

DIT_ROOT = "/root/Workspace/xy/DiT"
ASSETS = os.path.join(DIT_ROOT, "assets")
TRAIN_CSV = os.path.join(ASSETS, "train_top10_style23.csv")
STRICT84_CSV = os.path.join(ASSETS, "eval_top10_strict_subset84.csv")
SEEN20_CSV = os.path.join(ASSETS, "eval_top10_seen_20.csv")
CONTENT_FONT_DIR = os.path.join(DIT_ROOT, "baseline", "data", "content_font", "deng")

IMG_SIZE = 256


def load_train_rows():
    with open(TRAIN_CSV, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_eval_rows(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def slots_in_order(rows):
    """23 slots ordered by pair_id, each with its training img filenames (basename)."""
    by_slot = {}
    for r in rows:
        by_slot.setdefault((int(r["pair_id"]), r["slot_name"]), []).append(
            os.path.basename(r["image_path"]))
    return [k[1] for k in sorted(by_slot.keys())], {k[1]: v for k, v in sorted(by_slot.items())}


def renderable_chars():
    """Chars with a rendered content-font PNG available."""
    return {fn[:-4] for fn in os.listdir(CONTENT_FONT_DIR) if fn.endswith(".png")}


def content_png(char):
    return os.path.join(CONTENT_FONT_DIR, f"{char}.png")


def _stable_rng(*keys):
    h = hashlib.md5("||".join(keys).encode("utf-8")).hexdigest()
    return random.Random(int(h[:16], 16))


def slot_char_pool(rows):
    """{slot: sorted unique char list} — one glyph per (slot,char), matching
    the dedup'd layouts of all three baselines."""
    pool = {}
    for r in rows:
        pool.setdefault(r["slot_name"], set()).add(r["character"])
    return {s: sorted(c) for s, c in pool.items()}


def build_refs(refs_json, k=3):
    """Deterministic per-(slot,char) reference protocol shared by ALL baselines.

    refs[slot][char] -> [k ref CHARS] drawn from that slot's training chars,
    target char excluded (faithful to LF-Font's FixedRefDataset protocol:
    `avail_unis = ref_unis - {trg_uni}`). Each ref char maps to the dedup'd
    first training image of that (slot, char) in every baseline's layout.
    Deterministic: md5(slot||char)-seeded RNG.
    """
    if os.path.exists(refs_json):
        import json
        with open(refs_json, encoding="utf-8") as f:
            return json.load(f)

    train_rows = load_train_rows()
    pool = slot_char_pool(train_rows)

    items = []  # (slot, char)
    for r in load_eval_rows(STRICT84_CSV):
        items.append((r["slot_name"], r["character"]))
    for r in load_eval_rows(SEEN20_CSV):
        items.append((r["slot_name"], r["character"]))

    refs = {}
    for slot, char in items:
        cand = [c for c in pool[slot] if c != char]
        if len(cand) < k:
            raise RuntimeError(f"slot {slot} has only {len(cand)} ref chars for {char}")
        rng = _stable_rng(slot, char)
        refs.setdefault(slot, {})[char] = sorted(rng.sample(cand, k))

    os.makedirs(os.path.dirname(refs_json), exist_ok=True)
    import json
    with open(refs_json, "w", encoding="utf-8") as f:
        json.dump(refs, f, ensure_ascii=False, indent=1)
    print(f"refs for {sum(len(v) for v in refs.values())} (slot,char) items -> {refs_json}")
    return refs
