"""P0a: variance decomposition and shared-term permutation tests.

Balanced panel: characters covered by ALL dense writers with >=min images.
Model per feature dim:  f[w,c] = m[c] + a[w] + i[w,c] + eps
  m[c]  char content (what STD+Script can express)
  a[w]  writer main effect  -> everything a discrete Writer ID can express
  i[w,c] writer-by-char interaction -> the char-dependent style the reference
        route must transfer; transferable only if it is structured, not noise

Reported:
 1. energy shares of m / a / i (of total, after removing per-(w,c) means)
 2. interaction structure: singular spectrum of the interaction matrix vs a
    char-label permutation null (does i live in few directions or is it noise?)
 3. shared-writer residual correlation: observed corr(r_AB, r_AC) vs
       (a) independence baseline 0.5 (analytic, equal-variance model)
       (b) permutation null: permute char alignment of partner sequences
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bank', default='assets/u100_pretrain_20261024/style_bank_top10.npz')
    parser.add_argument('--writers', type=int, default=6)
    parser.add_argument('--min-imgs', type=int, default=2)
    parser.add_argument('--perms', type=int, default=200)
    parser.add_argument('--out', default='assets/u100_pretrain_20261024/exp_a2_variance.json')
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists(): raise FileExistsError(out)
    rng = np.random.default_rng(90)
    d = np.load(args.bank)
    feats, writers, chars = d['features'], d['writers'], d['chars']
    W = args.writers
    wc = defaultdict(list)
    for i in range(len(chars)):
        if int(writers[i]) < W:
            wc[(int(writers[i]), int(chars[i]))].append(feats[i])
    wc_mean = {k: np.mean(v, 0) for k, v in wc.items() if len(v) >= args.min_imgs}
    by_char = defaultdict(set)
    for (w, c) in wc_mean: by_char[c].add(w)
    panel = sorted(c for c, ws in by_char.items() if len(ws) == W)
    if len(panel) < 30: raise ValueError(f'balanced panel too small: {len(panel)}')
    F = np.stack([np.stack([wc_mean[(w, c)] for w in range(W)]) for c in panel])  # [C,W,D]
    D = F.shape[2]

    m = F.mean(1)                     # [C,D] char content
    g = F - m[:, None, :]             # writer deviation from char mean
    a = g.mean(0)                     # [W,D] writer main effect
    I = g - a[None, :, :]             # [C,W,D] interaction (+ eps residue)

    e_m = float((m ** 2).mean())
    e_a = float((a ** 2).mean())
    e_i = float((I ** 2).mean())
    total = e_m + e_a + e_i

    # interaction structure: singular spectrum vs char-permutation null
    I_mat = I.transpose(0, 1, 2).reshape(len(panel), W * D)   # [C, W*D]
    sv_obs = np.linalg.svd(I_mat, compute_uv=False)
    sv_null = np.zeros((args.perms, min(len(panel), W * D)))
    for p in range(args.perms):
        perm = np.empty_like(F)
        for w in range(W):
            perm[:, w, :] = F[rng.permutation(len(panel)), w, :]  # independent char shuffle per writer
        mp = perm.mean(1)[:, None, :]
        ap = (perm - mp).mean(0)[None, :, :]
        Ip = perm - mp - ap
        sv_null[p] = np.linalg.svd(Ip.reshape(len(panel), W * D), compute_uv=False)
    sv_obs_n = sv_obs / sv_obs.sum()
    sv_null_n = sv_null / sv_null.sum(axis=1, keepdims=True)
    top_frac_obs = float(sv_obs_n[:10].sum())
    top_frac_null = sv_null_n[:, :10].sum(1)
    p_top = float((top_frac_null >= top_frac_obs).mean())

    # shared-writer residual correlation with permutation null
    def pair_resid_corr(permute=False):
        cors = []
        for A in range(W):
            partners = [b for b in range(W) if b != A]
            for i in range(len(partners)):
                for j in range(i + 1, len(partners)):
                    B, C = partners[i], partners[j]
                    rA = g[:, A, :] - g[:, A, :].mean(0)
                    rB = g[:, B, :] - g[:, B, :].mean(0)
                    rC = g[:, C, :] - g[:, C, :].mean(0)
                    if permute:
                        orderB = rng.permutation(len(panel)); orderC = rng.permutation(len(panel))
                        rB = rB[orderB]; rC = rC[orderC]
                    x = (rA - rB).flatten(); y = (rA - rC).flatten()
                    cors.append(float(np.corrcoef(x, y)[0, 1]))
        return cors
    obs = pair_resid_corr(False)
    null = [pair_resid_corr(True) for _ in range(args.perms)]
    null_mean = float(np.mean([c for n in null for c in n]))
    obs_mean = float(np.mean(obs))
    p_shared = float(np.mean([np.mean(n) >= obs_mean for n in null]))

    summary = dict(
        panel_chars=len(panel), writers=W, dims=D,
        energy_share=dict(char_content_m=round(e_m / total, 4),
                          writer_main_a=round(e_a / total, 4),
                          interaction_i=round(e_i / total, 4)),
        writer_main_effect_norms={str(w): round(float(np.linalg.norm(a[w])), 3) for w in range(W)},
        interaction_spectrum=dict(top10_frac_observed=round(top_frac_obs, 4),
                                  top10_frac_null_mean=round(float(top_frac_null.mean()), 4),
                                  top10_frac_null_p95=round(float(np.percentile(top_frac_null, 95)), 4),
                                  p_value=round(p_top, 4)),
        shared_writer_residual_corr=dict(observed_mean=round(obs_mean, 4),
                                         independence_baseline=0.5,
                                         permutation_null_mean=round(null_mean, 4),
                                         p_value=round(p_shared, 4),
                                         excess_over_baseline=round(obs_mean - 0.5, 4)),
        note='energy shares: a[w] is everything a discrete Writer ID can express; '
             'i[w,c] must be transferred by any reference mechanism; permutation shuffles '
             'char alignment per writer, destroying char-level consistency while keeping marginals')
    out.write_text(json.dumps(dict(summary=summary), indent=1))
    print('EXP_A2_DONE', json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
