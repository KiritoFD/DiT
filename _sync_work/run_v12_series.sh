#!/bin/bash
# v12 ???????: ?? tmux ?????? (?? 100k ?, ???????)
#   v12_d8    : depth 12->8  (XS/2, ?)
#   v12_w320  : width 384->320 (S320/2, ?)
#   v12_12ch  : +12ch aux ???? (canny 0.3 / skel3 0.8)
#   v12_xattn : glyph ?? adaln -> xattn (4?)
# base=v12 (S/2 + factorized_cat + adaln + glyph_vec), ???????????
set -u
cd /root/Workspace/xy/DiT || exit 1
export PYTHONPATH=/root/Workspace/xy/DiT
export TORCHINDUCTOR_CACHE_DIR=/root/.cache/torch/inductor
PY=/opt/conda/envs/cu121/bin/python
CFG=src/train/configs
for NAME in v12_d8 v12_w320 v12_12ch v12_xattn; do
    LOGD=/root/Workspace/xy/DiT/logs/v12_series/$NAME
    mkdir -p $LOGD
    mkdir -p /root/Workspace/xy/DiT/assets/registry/configs
    cp $CFG/${NAME}_pretrain.json /root/Workspace/xy/DiT/assets/registry/configs/ 2>/dev/null || true
    TS=$(date +%Y%m%d-%H%M%S)
    echo "[series] ===== START $NAME  $(date) ====="
    $PY -u src/train/train.py --config $CFG/${NAME}_pretrain.json 2>&1 | tee $LOGD/train_$TS.log
    RC=$?
    echo "[series] ===== END $NAME  rc=$RC  $(date) ====="
done
echo "[series] ALL DONE $(date)"