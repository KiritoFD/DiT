#!/bin/bash
cd /root/Workspace/xy/DiT
tar czf /tmp/all_posters.tgz assets/results/v11_pretrain_S2_ref12ch/posters assets/results/v11_pretrain_S2_v8_aux02/posters assets/results/v11_pretrain_M432_v8_sk3/posters assets/results/v11_pretrain_M432_adaln4_sym/posters assets/results/v11_pretrain_M432_adaln4_sym_noise400k/posters assets/results/v11_pretrain_Sp2_base/posters
ls -lh /tmp/all_posters.tgz
