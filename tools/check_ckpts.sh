#!/bin/bash
for d in v11_pretrain_S2_ref12ch v11_pretrain_S2_v8_aux02 v11_pretrain_M432_v8_sk3 v11_pretrain_M432_adaln4_sym v11_pretrain_M432_adaln4_sym_noise400k; do
  echo "== $d"
  ls -t /root/Workspace/xy/DiT/assets/results/$d/*/checkpoints/[0-9]*.pt 2>/dev/null | head -2
done
