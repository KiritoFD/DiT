# Official-weight HWD: square calligraphy panels

CPU-only runner; two threads, batch1, no interaction with active GPU ownership.
Uses official FontSquare VGG16-10400, strict full checkpoint load, frozen features.
Matches upstream HWDScore non-streaming mechanism at commit
`eabb5b5ede5899ac7f8b716429cbd1a1c97c7485`: white pad, nearest height32,
RGB [0,1], no ImageNet/feature normalization, author mean feature L2, author macro mean.
Square tiles produce exactly one final VGG token each. Non-square tiles fail closed.
The upstream streaming counter has a mixed-author counting concern; this port uses
the unambiguous non-streaming per-author formula, not that code path.

```
python -m tools.eval_unpaired.hwd.run --roots <completed-eval-roots> \
  --weights <local-official-VGG16_class_10400.pth> --out <new-output>
```

Scores compare generation and saved GT with identical character composition.
This adapts HWD to Chinese single characters; FontSquare pretraining does not
establish calligraphic aesthetics or Chinese writer sensitivity. Results must be
reported beside E-G/E-L, not used as a new unquestionable ranking. Different
100/187-character panels are not directly comparable. Cache inputs include exact
PNG and CSV identities. No model, image, or latent transfer is needed.

`verify.py` adds same-script, different-character, K64 cosine verification in
unit-normalized HWD features, with GT controls. Original non-square reference
crops receive aspect-preserving square padding; this is a labelled adaptation,
not the official HWD distance. Input identities and cached reference features
are retained. On this study GT AUC is only51–54%; do not use it as a trustworthy
Chinese writer evaluator or interpret its disagreement as model style failure.

`aesthetic.py` is an exploratory frozen-HWD-feature ridge regressor, **not** the
CHAED ANN or TabNet paper reproduction. The official public ZIP lacks all10
GB3632 images; the available subset has990 images/99 characters. Exact exclusion
is recorded, labels are never remapped. Nested character-grouped CV fits scaling
and selects ridge alpha inside each training fold. A transfer perturbation screen
remains mandatory. This baseline failed and is not promoted to a primary metric.
