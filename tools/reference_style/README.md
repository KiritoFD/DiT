# Reference-conditioned FM (DINOv2, not DINOv3)

## Fixed data scope, 2026-10-23

Only the already protected K/X/L pool is allowed. No other scripts, no additional
Top10 images and no target-GT feature as generator condition.

4090 data directory: `/root/Workspace/xy/DiT/assets/reference_kxl_20261023`.

| Split | Images |
|---|---:|
| Original protected training pool | 100263 |
| Training with strict cross-character references | 100095 |
| K/X/L training | 35935 / 35057 / 29103 |
| Held-out validation | 1046 |
| Excluded: no different-character reference in same writer/script group | 168 |

Labels are restored from the raw metadata by exact original image filename,
checking character and script equality. Existing generic `UNKNOWN/-1` labels
remain untouched; supervision lives in new `style_writer_*` columns. Source-scoped
writer keys avoid merging unrelated IDs across datasets: 636 eligible groups.
These are not 636 independently verified historical writers. Directory source
is not a verified artwork identity. Existing split/duplicate protection is reused;
this pass checks exact train/validation pixel disjointness, not a new near-duplicate audit.

```bash
python -m tools.reference_style.prepare \
  --source assets/generic48_kxl_top10_20261022 \
  --raw /root/Workspace/xy/UNIFIED_RAW/meta/train_clean.csv \
  --out assets/reference_kxl_20261023
```

## Implemented building blocks

- Frozen DINOv2-B final global CLS: 768D; feature extraction must be offline.
- Trainable `768 -> 256 -> 128` head, normalize each reference then masked mean,
  normalize aggregate. No codebook or hand-crafted brush attributes.
- `reference_features=[B,K,768]`, mask `[B,K]` are explicit U-Net kwargs.
  Writer mode must be `none`. Missing references fail instead of falling back to IDs.
- Reference bank contains only actual training rows, samples 1..K distinct other
  characters of the same source-scoped writer and same script. No target-character fallback.
- Half the batch uses same-script 2x2 quartets for hard negatives; half uses
  same-script writer-balanced cross-character pairs so non-shared characters are
  also exposed. Quartet selection is uniform over eligible writer pairs, not
  exactly uniform over writers; this changes exposure and is recorded in the manifest.
- Full-batch SupCon, default temperature .1, hard-negative denominator weight 2,
  coefficient .05. Same-writer/same-character duplicates and cross-script pairs excluded.
- Shared condition-transform + joint objective hooks; one backward/optimizer/EMA cycle.
  Projection parameters are in model checkpoints/EMA and the ordinary optimizer.

## Launch and evaluation

User explicitly requested immediate launch on 4090. The frozen cache completed for
all 100095 training images (~1064 images/s). It is bound to the training CSV SHA256;
runtime rejects a different CSV. `run.py` then launches the existing shared trainer.
Reference tensors are explicit extra sampler kwargs, including correct tail padding.
The same training-only pool provides fixed references to validation; 2 of the original
1046 targets have no valid cross-character reference, leaving **1044** evaluated rows.
Do not compare this subset directly with an unfiltered 1046-row historical mean.

```bash
python -m tools.reference_style.cache --csv assets/reference_kxl_20261023/train.csv \
  --out assets/reference_kxl_20261023/features.npz
python -m tools.reference_style.run --steps 30000 --batch 256
```

Model: channels192/384/576, 4 blocks per scale, frozen DINOv2 plus trainable128D head,
no writer conditioning, LR6e-5, warmup500, 30k cosine horizon; checkpoint/evaluation
every5k. Existing VAE/STD bbox cache/FM/EMA are retained. No manual attributes, REPA,
new VAE, writer table learning or target-feature conditioning. First-step throughput,
actual peak memory and generated quality must be read from live logs, not inferred
from the cache extraction speed. Failure status is persisted; no silent retry loop.
Initial batch512 failed with a first-forward CUDA OOM; it is not a successful run.
The actual launch was reduced to batch256 without changing data, LR or step budget.

## Top10 fine-tuning, 2026-10-24

```bash
python -m tools.reference_style.finetune
```

Output: `assets/reference_top10_20261024`; initialize **all** model tensors from
pretraining **25k EMA**, strict keys/shapes and tensor equality, no newly added branch.
Optimizer, LR schedule, EMA and local step clock restart. Train the whole U-Net and
128D head; DINO remains frozen and offline. No writer ID conditioning or Writer adapter.

Only the existing protected Top10 train23485 / validation100 / test200 are used.
Reuse the audited remapped image IDs and VAE latent cache from the previous curriculum
bundle. Reference features come exclusively from Top10 training images, same writer
and script, strictly different characters. Named Top10 writer labels supervise SupCon
and sampling only. No generic pretraining replay is mixed into this first fine-tune.

10k additional steps, batch256, LR2e-5, warmup200, cosine to .1 peakLR, checkpoint and
validation every5k. Keep FM+.05 full-batch SupCon and the same mixed quartet sampler.
Run validation before the first update to provide a same-protocol downstream baseline;
never compare its100-row mean directly to pretraining's1044-row validation.
The test200 set is preserved, not used as references or training examples.

### Continue Top10 to total30k

```bash
python -m tools.reference_style.finetune --out assets/reference_top10_30k_20261024 \
  --steps 30000 --resume assets/reference_top10_20261024/runs/20261024-010707-Unet-flow/checkpoints/0010000.pt
```

Full raw/EMA/Adam/RNG resume at local10k, with total horizon30k and the same training
CSV/cache/reference/validation protocol. No re-encoding, model transfer or EMA reset.
New directory preserves the original0/5k/10k artifacts. Save/evaluate15/20/25/30k.
BaseLR2e-5 is unchanged; extending the cosine horizon rejoins it at approximately
1.56e-5 at10k instead of remaining at the old terminal2e-6. This schedule change is
explicit, not claimed to be equivalent to an uninterrupted original10k trajectory.

Feature-bank schema for the current building blocks: `ids:int64[N]`,
`features:float32[N,768]`, `writers:int64[N]` (style supervision IDs),
`chars:int64[N]` (Unicode scalar), `scripts:int64[N]` (consistent dense script IDs).
Only `train.csv` rows may populate this bank; targets for SupCon also come from
training rows. Freeze DINO, not the projection. Generator inference needs explicit
reference features, STD and script, never a writer table lookup.
