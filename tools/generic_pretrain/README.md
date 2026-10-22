# 48M generic pretraining → Top10 Writer adaptation

Active 2026-10-22 experiment: `assets/generic48_kxl_top10_20261022` on 4090.
Run from repository root: `python -u -m tools.generic_pretrain.full`.
The persistent runner writes `status.json`, stage logs, configs, CSVs and a manifest.
It exits on preparation/training failure; it never silently skips a failed stage.

## Data and contracts

Use only 楷/行/隶 from `UNIFIED_RAW/meta/train_clean.csv`: 36,591 / 35,622 / 29,465,
101,678 candidates before necessary gates. No new aesthetic filtering.
Exclude unreadable/unsupported characters and exact/conservative near duplicates of frozen
Top20 validation/test and legacy evaluation. This is not proof of artwork-level disjointness.
Generic validation holds out 1,000 exact-pixel groups: train100,263 / validation1,046.
369 protected near duplicates excluded; total accepted101,309. Counts are in `manifest.json`.
No Writer lookup or projection during generic forward. Script aliases are mapped by name,
not by incompatible unified numeric IDs: 楷0/行3/隶4 → dense0/1/2.

STD uses the frozen per-script fonts, actual ink bbox fitted proportionally into256,
BOX-downsampled to64. No blur, no shape augmentation. VAE is unchanged SD EMA,
posterior mode, scale0.18215, shift0. Unique explicit IDs and shard names are checked.
Downstream Top10 uses frozen train23,485 / validation100 / test200, not Top20.
Its 10×32 writer table is selected by raw-ID mapping from the independently contrastively
trained Top20 table, then frozen throughout both stages.

## Serial budgets

| Stage | Setup | Budget |
| --- | --- | --- |
| Generic | channels96/192/256, 8 blocks/scale, B512, compiled, LR7e-5 | 50k, unchanged50k LR horizon; save/evaluate5k |
| Top10 | new zero-initialized rank16 Writer-only FiLM in middle/decoder; EMA model-only transfer | 10k; save/evaluate5k |

Top10 steps0–999 train Writer branches only. From1k jointly train Decoder/Middle/Up/Output
at LR1e-5 while adapter peak LR is5e-5, warmup200 and10k cosine horizon.
Use10% generic replay by explicit sampler probability with hard-null Writer masking;
GT is only the FM target, never an inference condition. Optimizer, LR clock and EMA restart
at transfer; every old tensor must load and only Writer FiLM tensors may be new.
Refresh frozen-parameter EMA bookkeeping when unfreezing. Encoder and writer table remain frozen.

`full.py` is the accepted minimally gated runner; `contracts.py` contains its small CSV,
unknown-label and heldout-near-duplicate helpers. Earlier optional diversity-selection
utilities are not dependencies of this experiment.
No REPA, teacher or new VAE is introduced. Completion is not a quality claim: inspect unseen
and seen metrics, Writer and actual generated images before drawing conclusions.

## Live budget revision at ~23:35

The already-running trainer read its original30k stop at startup. Do not discard its updates:
`python -u -m tools.generic_pretrain.full --extend-running` replaces only the orchestration
parent and leaves the active trainer alive. At clean30k completion it resumes raw weights,
optimizer, scheduler, RNG and EMA to50k using `pretrain_50k_config.json`; the50k LR horizon
is unchanged. DataLoader replay is not claimed bitwise identical. Then
`top10_50k_config.json` takes the50k EMA into10k adaptation with5k saves/evaluations.
Original launch configs/manifests are preserved as historical identities.

The [condition-path audit](../../ANALYSIS/generic48_top10_condition_contract_20261022.md)
explains STD/Script/Writer semantics, zero-equivalent transfer, null masking and CPU checks.
