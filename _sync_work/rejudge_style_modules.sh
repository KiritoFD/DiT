#!/bin/bash
# 用 correct/shuffled 诊断重判历史风格模块（当年被 strict ssim 判死）
cd /root/Workspace/xy/DiT || exit 1
PY=/opt/conda/envs/cu121/bin/python
run() {
  local TAG="$1"; local CK="$2"
  echo ""
  echo "########## $TAG ##########"
  echo "ckpt=$CK"
  if [ ! -f "$CK" ]; then echo "  (ckpt 不存在, 跳过)"; return; fi
  $PY -u tools/diag_callig_cond.py --ckpt "$CK" --n 100 2>&1 | grep -av Warning \
    | sed -n '/判据 ①/,$p'
}
run "lowrank_spatial_r32 @35k (空间风格)" \
    "$(ls assets/results/v17_lowrank_spatial_r32/*/checkpoints/0035000.pt 2>/dev/null | head -1)"
run "E1 style_ln @40k" \
    "$(ls assets/results/v17_s2z_ada1_ln/*/checkpoints/0040000.pt 2>/dev/null | head -1)"
run "multistyle_k4 @150k (风格 token)" \
    "$(ls assets/results/v15a_multistyle_k4/*/checkpoints/0150000.pt 2>/dev/null | head -1)"
run "styletok @195k" \
    "$(ls assets/results/v13_styletok/*/checkpoints/0195000.pt 2>/dev/null | head -1)"
echo ""
echo "=== 对照: E0 (主线, 无风格模块) correct-shuffled = +0.0111 ==="
