# DiT Calligraphy Generation — Experiment Log

This document is a chronological record of the major experiment series (V1/V2
through S7) for the calligraphy diffusion-transformer (DiT) project. Each entry
summarizes the setup, key findings, and how the result shaped the next decision.
A consolidated summary table and a list of the standing design decisions derived
from this history appear at the end.

The log is maintained as the single high-level index for the project; detailed
design notes and per-run reports live in the linked subdocuments under
`docs/design/`, `docs/experiments/`, and `docs/s6_report/`.

---

## V1 / V2 — Early exploration (2026-08-10 → 2026-08-14)

### Setup

- **Model**: original DiT-S/2 backbone with a single ImageNet-style class
  condition, later extended to a 3-factor joint-condition MLP.
- **Conditions**: calligrapher × script × character (3 factors), fused by a
  legacy joint nonlinear MLP (single embedding lookup over the triple).
- **Data**: full dataset, from-scratch training, no structural losses, no EMA.

### Findings

- **The three factors do not compose.** Script is heavily confounded with
  calligrapher:
  - Mutual information between calligrapher and script **I = 1.527 bits**.
  - **79.28%** of calligraphers appear in only one script.
- The effective configuration space is enormous but barely covered:
  - **81.98%** of observed triples have only a single image.
  - The training set covers only **0.264%** of the configured triple space.
- With script confounded into calligrapher and almost no per-triple support, the
  joint MLP degenerates to memorization of a sparse set of triples rather than
  learning compositional factors. Generalization to unseen triples is not
  achievable this way.

### Conclusion

The 3-factor joint approach is fundamentally limited. The path forward is a
**compositional** design: merge the confounded factors into a single glyph
identity and use factorized conditioning so that unseen factor combinations can
be generated from individually-observed factors. This directly motivates V3-A.

---

## V3-A — Factorized 2-condition + glyph_id (2026-08-15)

### Setup

- **Condition refactor**: merge `script × character → glyph_id`
  (35,130 classes total, 21,495 active in the dataset).
- **Fusion**: `factorized_add` — calligrapher embedding (128-d) and glyph
  embedding (256-d) are independently projected and **added** into the model's
  conditioning stream, rather than being concatenated and pushed through a joint
  MLP.
- **CFG training**: 4-way classifier-free guidance dropout —
  full 60% / calligrapher-only 15% / glyph-only 15% / unconditional 10%.
- **Model**: `DiT-2Cond-S/2`, trained from scratch.
- **Data**: top6 calligraphers.

### Findings

- `factorized_add` **enables compositional generalization**: the model can
  produce reasonable outputs for (calligrapher, glyph) pairs that were never
  observed together, as long as each factor was seen individually. This is the
  core capability the 3-factor joint MLP could not deliver.
- The 4-way CFG mask gives usable guidance control across both axes
  independently.

### Conclusion

Two independently-embedded factors with additive fusion is the right
conditioning structure. This becomes the project default from V3 onward.

- Detailed design:
  `docs/design/2026-08-15-sparse-compositional-calligraphy-dit.md`
- Experiment notes:
  `docs/experiments/2026-08-15-factorized-3cond.md`,
  `docs/experiments/2026-08-15-v3a-2factor-glyph.md`

---

## V3-B / V3-C — XL variants with glyph conditioning (2026-08-16)

### Setup

- **Model**: `DiT-2Cond-XL/2` variants.
- **Glyph conditioning (spatial)**: a standard glyph latent is added as a
  spatial condition. The glyph latent library is built by `sd-vae`-encoding font
  renderings (kai / lishu).
- **glyph_embedder**: `Conv2d → token-add` with a **learnable scale** initialized
  to 0.4, so the glyph condition is injected gradually during training.
- **Data**: `kailishu` subset.

### Findings

- Spatial glyph conditioning **helps** — providing the model with an explicit
  target-glyph shape signal improves character-fidelity generation.
- However, the **XL backbone is overkill** at the current data scale. The extra
  capacity is not justified by the available data and does not pay for itself in
  quality.

### Conclusion

Keep glyph conditioning as an idea worth revisiting, but do not pursue the XL
backbone. Match model capacity to data scale — stay on the S-sized backbone.

---

## S5 Series — top30, structural-loss exploration (2026-08-17 → 2026-08-20)

### Setup

- **Data**: top30 calligraphers, **128,842 images**, with latent-cached
  training.
- **Model**: `DiT-2Cond-S/2`.
- **Variants**: multiple structural-conditioning variants tested in parallel —
  pixel-space structure, latent-space structure, canny edges, skeleton.

### Findings

- **Pixel-space structural losses (canny / skeleton) drag down generation
  quality.**
  - Training develops **colored noise artifacts**.
  - The x0 prediction drifts off the VAE manifold: with structural losses the
    raw `X0Lat` sits around **36–39**, whereas the diff-only baseline's raw
    X0Lat is only **1–2.5**. That is roughly an order of magnitude of manifold
    drift.
  - Pixel-space structure losses push x0 prediction away from the learned VAE
    manifold, and the decoder amplifies that drift into visible artifacts.
- **top30 diff-only at 70k**: MSE = 0.841, SSIM = 0.520 — and still improving
  when the run was stopped.

### Analysis

- Latent-vs-pixel struct comparison:
  `docs/experiments/2026-08-17-latent-vs-pixel-struct.md`

### Conclusion

Structural losses in pixel space are actively harmful. The next step is a
controlled, head-to-head comparison of diff-only versus struct on identical
data to quantify the gap — this becomes the S6 series.

---

## S6 Series — Controlled diff-only vs struct comparison (2026-08-20 → 2026-08-22)

### Setup

- **Controlled comparison**: diff-only versus struct on the **same data**
  (top6, **10,866 images**) so that the only variable is the loss composition.
- **Model**: `DiT-2Cond-S/2`, `sd-vae` f8, identical evaluation set for both
  runs.

### Findings — diff-only is better

- **diff-only @ 195k**: MSE = **0.432**, SSIM = **0.732** — clean, sharp
  calligraphy.
- **struct @ 120k**: MSE = **0.788**, SSIM = **0.403** — noise artifacts, visibly
  worse on both metrics despite fewer steps.

The struct variant is worse on both MSE and SSIM, and the qualitative output
shows the same colored-noise artifacts seen in S5. At equal loss weights,
structural losses (canny + skeleton) are **counterproductive**.

### Report

- Full report: `docs/s6_report/REPORT.md`

### Conclusion

Structural losses at equal weights are counterproductive. Abandon them and move
to **pure diff-only** training. This directly motivates S7.

---

## S7 Ramp — Attempted structural-loss ramp-up (2026-08-22)

### Setup

- **Attempt**: starting from the diff-only base @ 195k, ramp the structural loss
  weight up from 0 over 20k steps, on the theory that a gradual introduction
  might let the model absorb the structure signal without the manifold drift
  seen in S5/S6.

### Findings

- Even with the ramp, **structural loss did not help**. Quality did not improve
  over the diff-only base, and the manifold-drift / artifact tendency reappeared
  as the struct weight rose.

### Conclusion

Abandon structural losses entirely. The project pivots to **pure diff-only**
training plus a **better VAE** as the lever for further quality gains.

---

## S7 kl-f4 — Current run (2026-08-23 →)

### Setup

- **New VAE**: `kl-f4` (f4 downsampling, 3 channels, **55.3M params**), replacing
  `sd-vae-ft-ema` (f8, 4 channels, **83.7M params**).
- **Rationale**:
  - **2× lower floor noise**: kl-f4 floor-noise MSE = **0.0019** versus
    sd-vae MSE = **0.0037**. A cleaner reconstruction ceiling means the diffusion
    model is not spending capacity chasing VAE distortion.
  - **3× more latent information**: kl-f4 latent is **3 × 64 × 64 = 12,288**
    elements versus sd-vae's **4 × 32 × 32 = 4,096**. More spatial resolution is
    preserved into the latent the model has to predict.
  - **Fewer params**: 55.3M vs 83.7M, and f4 keeps more spatial detail than f8.
- **Model**: `DiT-2Cond-S/4` (patch size 4). With the larger latent, patch 4
  yields the **same 256 tokens** and effectively the same compute budget as the
  old S/2 on the f8 latent — i.e. more latent information for free.
- **Training**: from scratch, **batch = 224**, **bf16**, **EMA on**, **max_steps
  = 600k**.
- **Data**: top30, **128,842 images**, **26 latent shards** (latent-cached).

### Status

- Training in progress.
- Throughput: **3.51 steps/s**.
- VRAM: **19.74 GB**.

### Why this configuration

This run consolidates every lesson from the log above:

- **2Cond factorized_add** (from V3-A) for compositional generalization.
- **Pure diff-only** (from S6 / S7 ramp) — no structural losses.
- **kl-f4 VAE** (new) for a lower noise floor and 3× more latent information.
- **DiT-S/4** to keep token count and compute flat while absorbing the larger
  latent.
- **From scratch** (no ImageNet warm-start) for full parameter control.
- **EMA on** for stable evaluation and reliable early stopping.

---

## Summary table

| Series     | Period      | Model              | Data    | VAE | Key Result                     | Lesson                                   |
|------------|-------------|--------------------|---------|-----|--------------------------------|------------------------------------------|
| V1/V2      | 08-10→08-14 | DiT-S/2 3Cond      | full    | f8  | Failed                         | 3 factors don't work (script confounded) |
| V3-A       | 08-15       | DiT-2Cond-S/2      | top6    | f8  | factorized_add works           | Compositional > joint                    |
| V3-B/C     | 08-16       | DiT-2Cond-XL/2     | kailishu| f8  | glyph cond helps               | XL overkill for data scale               |
| S5         | 08-17→08-20 | DiT-2Cond-S/2      | top30   | f8  | Struct hurts                   | Pixel struct → VAE manifold drift        |
| S6         | 08-20→08-22 | DiT-2Cond-S/2      | top6    | f8  | diff-only best @0.432/0.732    | Pure diff > struct losses                |
| S7 ramp    | 08-22       | DiT-2Cond-S/2      | top6    | f8  | Ramp didn't help               | Abandon struct losses                    |
| **S7 kl-f4**| **08-23→** | **DiT-2Cond-S/4**  | **top30**| **f4**| **In progress**             | **Better VAE + pure diff-only**          |

---

## Key decisions & rationale

The configuration of the current S7 kl-f4 run is the product of the entire
history above. Each standing decision is listed with the experiment that
established it.

1. **2Cond, not 3Cond.** Script is confounded with calligrapher
   (I = 1.527 bits; 79.28% of calligraphers appear in only one script), so
   script is merged into `glyph_id`. *(V1/V2)*
2. **`factorized_add`, not legacy joint MLP.** Independent factor embeddings
   added together enable compositional generalization to unseen
   (calligrapher, glyph) pairs. *(V3-A)*
3. **Diff-only, not struct.** Structural losses push x0 off the VAE manifold
   (raw X0Lat ≈ 36–39 vs diff-only 1–2.5) and create colored-noise artifacts;
   even a ramp from 0 does not help. *(S5, S6, S7 ramp)*
4. **kl-f4, not sd-vae.** 2× lower floor-noise MSE (0.0019 vs 0.0037) and 3×
   more latent information (12,288 vs 4,096 elements), with fewer params.
   *(S7 kl-f4)*
5. **DiT-S/4, not S/2.** Patch 4 on the f4 latent keeps the token count at 256
   and compute flat versus S/2 on f8, while carrying 3× more latent
   information. *(S7 kl-f4)*
6. **From scratch, not warm-start.** No ImageNet bias; full control over every
   parameter. *(V3-A onward)*
7. **EMA on.** ~0.8% VRAM cost buys better evaluation stability and more
   reliable early stopping. *(S7 kl-f4)*

---

## References

- Design: `docs/design/2026-08-15-sparse-compositional-calligraphy-dit.md`
- Experiments:
  - `docs/experiments/2026-08-15-factorized-3cond.md`
  - `docs/experiments/2026-08-15-v3a-2factor-glyph.md`
  - `docs/experiments/2026-08-17-latent-vs-pixel-struct.md`
- S6 report: `docs/s6_report/REPORT.md`
- Early handover: `docs/HANDOVER_2026-08-15.md`
