# BC FM Top10

114M CNN U-Net + five continuous attributes; downstream is Top10, not Top20.
The existing shared FM trainer, EMA, checkpoint and evaluation infrastructure is retained.

## Active pipeline

4090 root: `assets/brush114_top10_20261023`.

1. `python -m tools.brush_fm.prepare --source assets/generic48_kxl_top10_20261022 --out assets/brush114_top10_20261023/attributes`
2. `python -m tools.brush_fm.run`
3. After inspecting control images, only if justified, create `P1_CONTENT_REVIEW_ACCEPTED` in the experiment directory and run `python -m tools.brush_fm.run --accept-p2`.

P0 measures fixed SD VAE **decoded cached posterior-mode** GT latents, not online arbitrary encodings.
WidthP20/P50/P80 are twice EDT on skeleton, divided by ink-bbox height. Junctions and their2pixel
neighborhoods, endpoints and small/uncertain measurements are excluded. Invalid measurements
become Brush-null without dropping the image FM objective. Ink density is mean blackness inside
the0.5mask. Edge softness is0.1–0.9transition area/perimeter/bbox-height: an approximate transition
width, not an exact normal-to-contour estimate. Crossings, degraded images and saturated ink remain limitations.

P0 reports original/reconstruction comparisons on the first1024 jointly valid cache rows, explicitly
not a random source-balanced audit. Initial gate: >=90% valid and Spearman>=.7 on three width
features. These engineering gates do not prove complete attribute reliability or style disentanglement.

Only 楷/行/隶 are supported initially. The full unified393,486pool contains101,678 such candidates,
of which101,309 passed the prior heldout exact/near guard. No claim of20万/40万 training is made.
Reuse those cached data plus Top10 train23,485. Remove exact generic duplicates and generic
overlap with Top10 pixels; preserve frozen Top10 train/validation/test. Take characters in Top10
test but absent from Top10 train as strict character holdout, excluding them from generic training
as well. All normalization uses this final training subset only; cache rows not in the subset
are excluded from statistics/banks. This does not establish artwork/source-disjoint generalization.

## Architecture and conditioning

Channels192/384/576, four ResBlocks/scale; STD bboxfit256→64 and Script are unchanged.
Brush MLP5→128→256, independent zero-initialized256→2C projections in Middle+Decoder.
The pretraining shared condition remains Time+Script, and Writer lookup/projection is bypassed.
Brush receives0.075standard-deviation noise and10% whole-vector dropout. Null masks are applied
**after** biased projections. Default inference uses train-only Script median, not evaluation GT.

Writer is added by strict same-Brush-architecture EMA transfer; only Writer FiLM keys may be new.
It is zero-equivalent at identical explicit Brush attributes. Writer-specific **default attributes**
can differ, so do not compare differing defaults and claim a failed transfer.
In Writer CFG both passes share identical Brush attributes, STD and Script.

## Training and gates

P1: B512, peakLR6e-5, warmup1500, decay horizon60k, stop5k, save/evaluate5k, FM only.
Explicit mixture75% generic /25% Top10, Writer disabled on both. VAE caches are not rebuilt.
Capacity probe is three timed optimizer steps plus warmup; no silent batch fallback.
P1 controls:16 strict heldout characters, three fixed seeds, low/mid/high controls for widthP50,
ink density and edge softness. WidthP50 is clamped between P20/P80; density and edge values
are physically clamped. These are independently specified controls, never target-GT attributes.
Results preserve16×3×3×3=432 original PNGs, CSV, summary and representative poster.
Statistic gate requires >=60% nondecreasing triplets and >=75% positive endpoints **per attribute**;
correlated seeds and content/topology quality still need manual image review. Failure stops P2.
Standard validation uses independently chosen train-only defaults, not Oracle conditions.

P2 after accepted P1: resume5k→60k with same horizon, then **Top10** full-network fine-tuning30k.
Writer-only peakLR8e-5, all other trainable weights1e-5, warmup200; save/evaluate5k including10/20/30k.
The independently contrastively trained Writer table stays frozen, consistent with the project contract;
only the added Writer FiLM learns. This deliberately does not jointly retrain the table.
Fine-tuning labels sample an empirical attribute vector from another **training character** of the
same Writer+Script, excluding every sample with the target character. No valid donor → Brush-null.
10% generic replay remains training-only GT-conditioned. Default inference uses a shrunk train-bank
Writer mean, n/(n+20) toward Script median; `sample_writer_attributes(cache,writer,script,seed)`
provides empirical training-bank sampling with no test access. Multivariate attributes are sampled
together, preserving correlations; no independent-component Gaussian assumption is made.

No no-Brush or from-scratch Top10 controls are silently launched. Existing48M results are not a
same-size114M baseline, and short114M no-Brush run is not a matched60k comparison. Claims about
positive transfer require those matched controls, independent Writer metrics and image review.
