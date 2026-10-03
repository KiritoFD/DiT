# _archive/ 实验记录（删除前存档）

共 67 个实验，来自 `assets/results/_archive/`。

> 该目录已于 2026-09-23 删除 ckpt；本表保留其配置与最终指标。

| 实验 | 模型 | fusion | 参数量 | LR | 步数 | Diff首→末 | ssim最佳(steps) | Steps/s |
|---|---|---|---|---|---|---|---|---|
| 20260906-054549-v10a-skel-cond-pretrain | DiT-2Cond-S/2 | factorized_add | 46,425,491 | 0.00015 0.02 | 22550→140600 | 0.2054→0.1191 | 0.8422 @95000 | 2.56 |
| 20260906-175850-v10b-skel-only-pretrain | DiT-2Cond-S/2 | factorized_add | 32,638,738 | 0.00015 0.02 | 50→86500 | 1.0414→0.1642 | 0.8397 @82500 | 2.8 |
| 20260906-010643-v10a-skel-cond-pretrain | DiT-2Cond-S/2 | factorized_add | 46,425,491 | 0.00015 0.02 | 50→23750 | 1.0074→0.2038 | 0.6137 @10000 | 2.81 |
| 20260907-211427-v10b-stdskel-fame3-d01 | DiT-2Cond-S/2 | factorized_add | 32,638,738 | 0.00015 0.02 | 50→24150 | 1.151→0.3126 | 0.5799 @5000 | 2.63 |
| 20260908-003035-v10b-stdskel-fame3-deep | DiT-2Cond-S/2 | factorized_add | 35,292,946 | 0.00015 0.02 | 50→33000 | 1.09→0.3005 | 0.5744 @5000 | 2.7 |
| 20260907-144350-v10b-stdskel-fame3 | DiT-2Cond-S/2 | factorized_add | 32,638,738 | 0.00015 0.02 | 50→58700 | 1.1519→0.2932 | 0.5712 @7500 | 2.6 |
| 20260907-141032-v10b-stdskel-pretrain | DiT-2Cond-S/2 | factorized_add | 32,638,738 | 0.00015 0.02 | 50→4400 | 1.154→0.3744 | 0.5404 @1000 | 2.76 |
| 20260912-123352-v11-pretrain-S2-ref12ch | DiT-2Cond-S/2 | factorized_add | 38,736,050 | 0.0001 0.02 | 50→1300 | 2.3126→0.4404 | - | 5.78 |
| 20260912-121045-v11-pretrain-S2 | DiT-2Cond-S/2 | factorized_add | 38,711,442 | 0.0001 0.02 | 50→1750 | 2.2268→0.4633 | - | 3.18 |
| 20260912-122236-v11-pretrain-S2 | DiT-2Cond-S/2 | factorized_add | 38,711,442 | 0.0001 0.02 | 50→250 | 2.2268→1.1186 | - | 3.13 |
| 20260912-030448-v11_struct-loss | DiT-2Cond-Sp/2 | factorized_add | 75,834,258 | 5e-05 0.02 | 390050→407050 | 0.1505→0.138 | - | 4.09 |
| 20260910-150038-c41x-sty16 | DiT-2Cond-Sp/2 | factorized_add | 76,899,474 | 0.00015 0.02 | 50→24550 | 1.9205→0.3063 | - | 4.04 |
| 20260909-175311-lrtest-5e-6 | DiT-2Cond-Sp/2 | factorized_add | - | - | -→- | - | - | - |
| 20260909-175319-lrtest-1.5e-5 | DiT-2Cond-Sp/2 | factorized_add | - | - | -→- | - | - | - |
| 20260909-175327-lrtest-5e-5 | DiT-2Cond-Sp/2 | factorized_add | - | - | -→- | - | - | - |
| 20260909-175335-lrtest-1.5e-4 | DiT-2Cond-Sp/2 | factorized_add | - | - | -→- | - | - | - |
| 20260909-175529-lrtest-5e-6 | DiT-2Cond-Sp/2 | factorized_add | 344,665,234 | - | -→- | - | - | - |
| 20260909-175541-lrtest-1.5e-5 | DiT-2Cond-Sp/2 | factorized_add | 344,665,234 | - | -→- | - | - | - |
| 20260909-175553-lrtest-5e-5 | DiT-2Cond-Sp/2 | factorized_add | 344,665,234 | - | -→- | - | - | - |
| 20260909-175604-lrtest-1.5e-4 | DiT-2Cond-Sp/2 | factorized_add | 344,665,234 | - | -→- | - | - | - |
| 20260909-175836-lrtest-5e-6 | DiT-2Cond-Sp/2 | factorized_add | 344,665,234 | 5e-06 0.02 | -→- | - | - | - |
| 20260909-180124-lrtest-1.5e-5 | DiT-2Cond-Sp/2 | factorized_add | 344,665,234 | 1.5e-05 0.02 | -→- | - | - | - |
| 20260909-180411-lrtest-5e-5 | DiT-2Cond-Sp/2 | factorized_add | 344,665,234 | 5e-05 0.02 | -→- | - | - | - |
| 20260909-180627-lrtest-5e-6 | DiT-2Cond-Sp/2 | factorized_add | 84,231,378 | 5e-06 0.02 | 175050→177000 | 0.1921→0.1877 | - | 4.19 |
| 20260909-181653-lrtest-1.5e-5 | DiT-2Cond-Sp/2 | factorized_add | 84,231,378 | 1.5e-05 0.02 | 175050→177000 | 0.1924→0.1836 | - | 4.21 |
| 20260909-182726-lrtest-5e-5 | DiT-2Cond-Sp/2 | factorized_add | 84,231,378 | 5e-05 0.02 | 175050→177000 | 0.1949→0.183 | - | 4.21 |
| 20260909-183752-lrtest-1.5e-4 | DiT-2Cond-Sp/2 | factorized_add | 84,231,378 | 0.00015 0.02 | 175050→177000 | 0.2085→0.204 | - | 4.21 |
| 20260910-144545-smoke-c41x-sty32 | DiT-2Cond-Sp/2 | factorized_add | 77,964,434 | 0.00015 0.02 | 50→1000 | 1.4651→0.4274 | - | 4.26 |
| 20260906-054343-v10a-skel-cond-pretrain | DiT-2Cond-S/2 | factorized_add | 46,425,491 | 0.00015 0.02 | 22550→22600 | 1.0125→0.5029 | - | 2.83 |
| 20260907-033105-v10a-dino-skel-cond-pretrain | DiT-2Cond-S/2 | factorized_add | 32,639,891 | 0.00015 0.02 | 50→80150 | 1.0717→0.1614 | - | 2.79 |
| 20260908-171835-v10b-stdskel-fame3-c41 | DiT-2Cond-Sp/2 | factorized_add | 65,303,442 | 0.00015 0.02 | -→- | - | - | - |
| 20260908-172252-v10b-stdskel-fame3-c41 | DiT-2Cond-Sp/2 | factorized_add | 65,303,442 | 0.00015 0.02 | 50→3950 | 1.1096→0.3537 | - | 3.43 |
| 20260908-174401-v10b-stdskel-fame3-c41d01 | DiT-2Cond-Sp/2 | factorized_add | 65,303,442 | 0.00015 0.02 | 50→16800 | 1.0907→0.3115 | - | 3.31 |
| 20260910-120052-v10b-stdskel-fame3-c41x-cos-cl | DiT-2Cond-Sp/2 | factorized_add | 75,834,258 | 5e-05 0.02 | 172550→175450 | 0.2486→0.1831 | - | 4.25 |
| 20260912-010838-v10b-stdskel-fame3-c41x-cos-e- | DiT-2Cond-Sp/2 | factorized_add | 75,834,258 | 5e-05 0.02 | 390050→391700 | 0.1468→0.1295 | - | 4.24 |
| 20260912-011838-v10b-stdskel-fame3-c41x-cos-e- | DiT-2Cond-Sp/2 | factorized_add | 75,834,258 | 5e-05 0.02 | 390050→411350 | 0.2108→0.2789 | - | 3.35 |
| 20260912-025853-v10b-stdskel-fame3-c41x-cos-e- | DiT-2Cond-Sp/2 | factorized_add | 75,834,258 | 5e-05 0.02 | 390050→390350 | 0.1505→0.1432 | - | 4.1 |
| 20260910-133702-v10b-stdskel-fame3-c41x-scratc | DiT-2Cond-Sp/2 | factorized_add | 77,415,570 | 5e-05 0.02 | 50→13400 | 2.1538→0.3325 | - | 4.02 |
| 20260910-172525-v10b-stdskel-fame3-c41x-scratc | DiT-2Cond-Sp/2 | factorized_add | 77,415,570 | 5e-05 0.02 | 12550→13900 | 0.3338→0.3321 | - | 4.2 |
| 20260908-040914-v10b-stdskel-fame3-deep | DiT-2Cond-S/2 | factorized_add | 35,292,946 | 0.00015 0.02 | -→- | - | - | - |
| 20260907-234643-v10b-stdskel-fame3-mid | DiT-2Cond-S/2 | factorized_add | 32,638,738 | 0.00015 0.02 | 50→450 | 1.151→0.4738 | - | 2.65 |
| 20260907-235343-v10b-stdskel-fame3-mid | DiT-2Cond-S/2 | factorized_add | 32,638,738 | 0.00015 0.02 | 50→5700 | 1.152→0.3577 | - | 2.8 |
| 20260910-173813-v10b-stdskel-fame3-scratch-sty | DiT-2Cond-Sp/2 | factorized_add | 77,964,434 | 5e-05 0.02 | 50→48650 | 2.21→0.2973 | - | 4.2 |
| 20260910-205828-v10b-stdskel-fame3-scratch-sty | DiT-2Cond-Sp/2 | factorized_add | 77,964,434 | 5e-05 0.02 | 47550→73550 | 0.2961→0.2923 | - | 4.1 |
| 20260910-224716-v10b-stdskel-fame3-scratch-sty | DiT-2Cond-Sp/2 | factorized_add | 77,964,434 | 5e-05 0.02 | 72550→204050 | 0.2963→0.2293 | - | 4.1 |
| 20260911-075000-v10b-stdskel-fame3-scratch-sty | DiT-2Cond-S/2 | factorized_add | 43,875,730 | 5e-05 0.02 | 50→113100 | 2.1677→0.2831 | - | 2.42 |
| 20260911-141832-v10b-stdskel-fame3-scratch-sty | DiT-2Cond-S/2 | factorized_add | 43,875,730 | 5e-05 0.02 | 50→1600 | 2.1677→0.5492 | - | 5.54 |
| 20260911-142555-v10b-stdskel-fame3-scratch-sty | DiT-2Cond-S/2 | factorized_add | 43,900,338 | 5e-05 0.02 | 50→10950 | 2.3343→0.2548 | - | 5.45 |
| 20260911-150233-v10b-stdskel-fame3-scratch-sty | DiT-2Cond-S/2 | factorized_add | 43,900,338 | 5e-05 0.02 | 50→1000 | 2.333→0.5215 | - | 3.73 |
| 20260911-151113-v10b-stdskel-fame3-scratch-sty | DiT-2Cond-S/2 | factorized_add | 43,900,338 | 5e-05 0.02 | 50→38100 | 2.333→0.2135 | - | 3.76 |
| 20260911-180759-v10b-stdskel-fame3-scratch-sty | DiT-2Cond-S/2 | factorized_add | 39,157,170 | 5e-05 0.02 | 50→13300 | 2.3241→0.2514 | - | 4.14 |
| 20260911-193749-v10b-stdskel-fame3-scratch-sty | DiT-2Cond-S/2 | factorized_add | 36,370,610 | 5e-05 0.02 | 50→850 | 2.3292→0.5915 | - | 4.41 |
| 20260911-194343-v10b-stdskel-fame3-scratch-sty | DiT-2Cond-S/2 | factorized_add | 36,370,610 | 5e-05 0.02 | 50→650 | 2.3276→0.6936 | - | 3.0 |
| 20260911-195016-v10b-stdskel-fame3-scratch-sty | DiT-2Cond-S/2 | factorized_add | 36,370,610 | 5e-05 0.02 | 50→1300 | 2.3279→0.4857 | - | 3.35 |
| 20260911-201236-v10b-stdskel-fame3-scratch-sty | DiT-2Cond-S/2 | factorized_add | 36,370,610 | 5e-05 0.02 | 50→35650 | 2.3279→0.215 | - | 3.33 |
| 20260908-051518-v10b-stdskel-fame3-sp | DiT-2Cond-Sp/2 | factorized_add | 65,433,106 | 0.00015 0.02 | -→- | - | - | - |
| 20260908-052149-v10b-stdskel-fame3-sp | DiT-2Cond-Sp/2 | factorized_add | 65,433,106 | 0.00015 0.02 | 50→80000 | 1.1151→0.2129 | - | 3.25 |
| 20260908-135937-v10b-stdskel-fame3-sp2 | DiT-2Cond-Sp/2 | factorized_add | 65,433,106 | 0.00015 0.02 | 50→29200 | 1.1218→0.2984 | - | 3.41 |
| 20260907-140753-v10b-stdskel-pretrain | DiT-2Cond-S/2 | factorized_add | 32,638,738 | 0.00015 0.02 | -→- | - | - | - |
| 20260907-125716-v10b-repa-strong | DiT-2Cond-S/2 | factorized_add | 32,638,738 | 0.00015 0.02 | 50→7750 | 1.0508→0.2454 | - | 2.76 |
| 20260902-122709-v8a-s30-base | DiT-2Cond-S/2 | factorized_add | 32,929,427 | - | 20→2060 | 2.3524→0.419 | - | 8.51 |
| 20260902-123403-v8a-s30-base | DiT-2Cond-S/2 | factorized_add | 32,929,427 | - | 20→1600 | 2.3483→0.4334 | - | 4.17 |
| 20260902-125615-v8a-s30-base | DiT-2Cond-S/2 | factorized_add | 32,929,427 | - | 20→133740 | 2.359→0.2342 | - | 3.99 |
| 20260904-232227-v9a-repa-pretrain | DiT-2Cond-S/2 | factorized_add | 32,929,427 | 0.00015 0.02 | 20→1000 | 2.4221→0.5211 | - | 3.78 |
| 20260904-233358-v9a-repa-pretrain | DiT-2Cond-S/2 | factorized_add | 32,929,427 | 0.00015 0.02 | -→- | - | - | - |
| 20260905-003217-v9a-repa-pretrain | DiT-2Cond-S/2 | factorized_add | 32,929,427 | 0.00015 0.02 | 20→200 | 2.4214→1.1656 | - | 2.36 |
| 20260905-004106-v9a-repa-pretrain | DiT-2Cond-S/2 | factorized_add | 32,929,427 | 0.00015 0.02 | 20→131240 | 2.4223→0.2673 | - | 2.81 |
