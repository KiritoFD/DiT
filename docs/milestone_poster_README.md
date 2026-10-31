# Milestone poster pipeline

`tools/milestone_poster.py` stitches **20 exact sample IDs** from a manifest. Example manifest: [milestone_poster.example.json](milestone_poster.example.json). It **does not generate images or recalculate metrics**, and fails on any missing PNG, duplicate idx, or incompatible protocol ID.

## Run (on the machine containing PNGs)

```bash
python tools/milestone_poster.py \
  --manifest tools/milestone_poster.example.json \
  --out assets/milestone_gallery/posters/historical_20.png \
  --font /usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc
```

**Important:** This JSON is a **layout example**, not a ready-made unified benchmark. The supplied STD pattern may not match the actual file naming; adapt it after checking the local asset. The historical v68 score is from `docs/107`; its evaluation implementation has to be checked before adding other rows to the same comparison. Change `protocol_id` to the actual, verified protocol and use a dedicated panel for every different evaluation contract. The sample IDs 0–19 are only a placeholder; replace with 20 frozen indices covering writers, scripts, complexity and failures.

Expected layout: **one checkpoint per row** (left: large checkpoint name, method and metrics); 20 columns (one indexed character per column); optionally add STD and GT rows. The poster writes `*.selection.json` containing the exact file mapping and sample identities. Run on local copies of just 20 PNGs per checkpoint; do not transfer GB checkpoints.

## Reuse / safety

- Historical inputs: `tools/setup_milestones_dir.py`, `tools/gather_eval_for_download.py`, `tools/make_poster.py`, `docs/1028/posters/*.indices.json`.
- **Do not** use filenames' lexical order, auto-pair a foreign CSV, hide missing images, or treat a mismatched Writer remap as valid.
- `tools/infer_milestones_eval200.py` is a historical helper; its `strict=False` model loading is **not safe for scientific reruns**. For GPU backfill, load checkpoint/model/decoder/VAE with exact contracts and verify generated image provenance before reporting metrics.
- Until current OCR and E-G are backfilled and compared under matching panels, write `pending` in Milestones.md instead of filling in guessed numbers.
