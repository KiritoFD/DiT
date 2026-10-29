# Writing-domain diagnostics on frozen features

`run.py` reuses completed E-G/E-L image feature caches. It reports pooled
verification ROC-AUC/EER and per-query ROC statistics, writer-macro retrieval
mAP, equal-writer/script-count Style-FID/KID, centroid distance and PRDC k5
density/coverage with an equal-N real-vs-real control. It trains no model.

The reference is explicitly generator-training data with queried characters
excluded. Unit-L2 learned features determine the Style-FID/KID scale; scores
cannot be compared numerically to Inception-FID/KID. Negative unbiased KID
estimates are retained. Pooled distribution matching does not guarantee that
the requested writer is correct. Dataset-source labels do not establish
artwork holdout. Small-sample point estimates are exploratory.

`paired.py` uses saved per-query statistics, checks identical writer/script/char
identities, fixes the10 writers and resamples characters within each writer.
Its confidence intervals describe writer-macro averages of per-query ROC
statistics, **not pooled ROC statistics**. EER/TAR calculated separately on
each query is a descriptive statistic, not a deployable common threshold.
The2,000 bootstrap samples do not represent independent training/noise seeds;
multiple comparisons are not adjusted.

Official HWD and CHAED/TabNet are not implemented by renaming the centroid
distance or these features. Their missing results remain missing.
