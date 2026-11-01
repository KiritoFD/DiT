# J2-vs-J3 comprehensive metrics

Mean with paired-sample bootstrap 95% confidence intervals.

| Arm | SSIM | LPIPS | Ink-IoU | Frag | SoftInk abs err | Severe SSIM | Ink-ratio OOR |
|---|---:|---:|---:|---:|---:|---:|---:|
| J2 | 0.6198 [0.6013,0.6385] | 0.3150 [0.2998,0.3292] | 0.3447 [0.3150,0.3744] | 2.236 [1.531,3.274] | 0.3218 [0.2702,0.3907] | 11.8% | 46.5% |
| J3 | 0.6193 [0.6012,0.6382] | 0.3147 [0.2994,0.3287] | 0.3433 [0.3143,0.3731] | 2.186 [1.494,3.177] | 0.3229 [0.2710,0.3925] | 11.8% | 47.1% |

## Paired improvement over J2

Positive means better after orienting lower-is-better metrics.

| Arm | dSSIM | dLPIPS | dInk-IoU | dFrag | dInk calibration |
|---|---:|---:|---:|---:|---:|
| J3 | -0.0004 [-0.0015,+0.0004] | +0.0003 [-0.0007,+0.0011] | -0.0014 [-0.0040,+0.0005] | +0.0508 [-0.0136,+0.1157] | -0.0011 [-0.0046,+0.0021] |
