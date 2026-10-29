# eval_unpaired_v1

Frozen evaluators, separate Quality / Style / Content / Diversity axes. No composite score.
No generator updates. Existing artifact screening and formal new generation are separate protocols.

Run with an isolated environment, from the repository root:

```sh
python -m tools.eval_unpaired.run --machine 4090 --reference-csv assets/u100_p1p2_20261025/p2_train.csv --out assets/eval_unpaired_v1/4090
```

The command takes the global GPU lock, checks ownership, strictly loads a trained global
Writer network, loads MUSIQ-AVA / MUSIQ-KONIQ / TOPIQ-NR, calibrates real/blur/thick/broken/blank,
then scores all discoverable completed generation artifacts. Failure is preserved; restart uses caches.
Use a new output version if the set of completed artifacts changes.

Reference images are deduplicated and sampled across available source/character groups,
up to160 per writer-script. Retrieval excludes the query character and exact pixel identity,
uses K32/64/128 references and the mean of the top5 similarities within each candidate writer.
Insufficient K is explicitly invalid. Competition is restricted to the same script.
Cross-dataset-source exclusion is reported separately; dataset labels are not artwork provenance.

Style MMD is an unbiased multi-bandwidth RBF estimate, using a real-only median bandwidth;
negative values are retained. It excludes query characters from the reference slot. This is
**Train-reference MMD**, not proof of generalization; character-composition confounding remains.
Real-split MMD is a small-sample diagnostic, not a corrected score.

Optional `--local-checkpoint` accepts only a strictly loaded trained PixelWriterV2 model.
Missing local weights, OCR, formal Clean-FID/KID and human calibration are recorded as missing.
No crop-based pseudo-local encoder or randomly initialized OCR score is substituted.

IQA is always labelled exploratory until TTF and human calligraphy preference calibration passes.
The perturbation screen alone does not establish calligraphic aesthetics.
Intervals use writer-character clusters; paired model differences and formal multi-seed
inference require identical sample identities and a later hierarchical analysis.

Formal panel B must use the intersection of characters excluded from every compared training
CSV, fixed legal writer-script slots and three shared noise seeds. Do not silently claim
200 unseen characters per writer when the intersection or STD support is insufficient.

References: [IQA-PyTorch model cards](https://github.com/chaofengc/IQA-PyTorch/blob/main/docs/ModelCard.md),
[Clean-FID](https://github.com/GaParmar/clean-fid).

`panels/history.py` maps historical PNGs to labels by exact saved-GT RGB pixels, retaining
native sampler identities; it refuses incomplete mappings. `panels/clean.py` reports pooled
small-sample distribution diagnostics with writer-script-balanced, equal-N real controls.
Its low-rank FID is algebraically identical to full empirical-covariance FID, not a PCA variant.
`panels/rescore.py` versions scalar statistics without re-running networks. MMD macro aggregation
is first within writer across scripts, then across writers; the slot mean is retained separately.
`execution/report.py` merges compact gzip exports and gives fixed-writer character-cluster
paired intervals. These are exploratory single-seed intervals, without multiple-test adjustment.

`execution/wait_s11.py` and `execution/finish.py` are one-shot evaluation jobs for the already
running experiment; they do not resume any architecture queue. Missing OCR weights fail closed
in `src/eval/eval_metrics.py`. After new calibration fails, formal panel B is withheld.

On 48, official release downloads required curl HTTP/1.1; both nodes downloaded their own
weights, with no checkpoint/image transfer. The isolated virtual environments inherit the
existing Torch installation; training environments were not upgraded.
