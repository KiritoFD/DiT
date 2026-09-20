#!/usr/bin/env bash
# ============================================================================
# v15 多模态风格 K=4 —— 从头训练矩阵 (v15a / v15b / v15c, 150k 步收尾)
#
# 设计(docs/919 §3 + 用户裁定 2026-09-19):
#   注入方式变了 resume 不公平(styletok 前车之鉴 + adaLN 条件几何断层) -> 三个
#   变体全部从头训。共享: 87 pair × K=4 可训表(DINO K-Means 质心初始化,
#   λ=0.01 mean 锚定) + v14_s2/v13_base 配方(REPA 0.03, wd 0.1, batch 360,
#   lr 1e-4, warmup 3000) + glyph_drop 0.1(双轴 CFG 前提)。
#     v15a  仅 mean pooling -> adaLN          (纯 token 容量, 消融基线)
#     v15b  a + CalligStyleCrossAttn 书家化骨架 (用户原设计核心)
#     v15c  a + 每 xattn 层 context 可见风格    (batch 260 / lr 7.2e-5, xattn 显存)
#   判据: strict/seen 轨迹 vs v13_base(0.5703@155k) / v14_s2(0.5703@160k)
#   + 最终 ratio_style(基线 1.18-1.42)。历史教训: 旧 SupCon 表的 DINO 锚定
#   因 extract 漏 /255(+repeat 维数 bug)从未生效, 特征已重提, 本矩阵的表是真 DINO。
#
# 用法(远端 /root/Workspace/xy/DiT):
#   bash scripts/ops/run_v15_multistyle.sh prep    # 重提 DINO 特征 + K-Means 聚类
#   bash scripts/ops/run_v15_multistyle.sh a|b|c   # 单独起一个变体 (tmux v15a/b/c)
#   bash scripts/ops/run_v15_multistyle.sh serial  # a -> b -> c 串行一条龙 (推荐)
#   bash scripts/ops/run_v15_multistyle.sh ratio [ckpt]  # ratio_style 判据
#
# ⚠ Windows 侧编辑过 -> 本文件必须保持 LF (本地生成即 LF, 远端严禁 sed s/\r//g)
# ============================================================================
set -euo pipefail
cd /root/Workspace/xy/DiT
export PYTHONPATH=/root/Workspace/xy/DiT
export TORCHINDUCTOR_CACHE_DIR=/root/.cache/torch/inductor

PY=/opt/conda/envs/cu121/bin/python
CSV=assets/train_50k_v2.csv
MAP=assets/callig_script_id_map.json
DINO=assets/dino_cls_50k.npz
MSEMB=assets/multistyle_k4_pretrained.pt
CFGA=src/train/configs/v15a_multistyle_k4_pool.json
CFGB=src/train/configs/v15b_multistyle_k4_ca.json
CFGC=src/train/configs/v15c_multistyle_k4_ctx.json
LOGDIR=logs/v15_series
mkdir -p "$LOGDIR"
STAGE="${1:-serial}"

gpu_busy_guard() {
  USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  if [ "$USED" -gt 500 ]; then
    echo "[gpu-guard] ✗ GPU 已被占用 (${USED}MiB) —— 先停掉在跑实验再启动"; exit 1
  fi
}

start_variant() {  # $1=tmux 会话名  $2=config 路径
  TS=$(date +%Y%m%d-%H%M%S)
  # ⚠ tmux 服务器已存在时, 新会话继承**服务器**环境而非本 shell -> 环境变量必须内联
  tmux new-session -d -s "$1" \
    "export PYTHONPATH=/root/Workspace/xy/DiT TORCHINDUCTOR_CACHE_DIR=/root/.cache/torch/inductor; \
     $PY -u src/train/train.py --config $2 2>&1 | tee $LOGDIR/${1}_train_$TS.log"
  echo "[$1] 已起 (tmux $1)。日志: tail -f $LOGDIR/${1}_train_$TS.log"
}

case "$STAGE" in
  prep)
    echo "===== Stage 0a: 重提 50k DINO CLS 特征 [GPU, ~1min] ====="
    # 旧 npz 曾因 load() 的 repeat 维数 bug(被 except 吞掉 -> 全零图) + 漏 /255
    # 而**全行恒定**; 两个 bug 已修, extractor 自带唯一值自检。
    $PY tools/extract_dino_cls_50k.py --csv "$CSV" --out "$DINO"
    echo "===== Stage 0b: 87 pair × K-Means(K=4) 聚类 [CPU, ~几分钟] ====="
    $PY tools/build_multistyle_k4.py \
        --npz "$DINO" --map "$MAP" --out "$MSEMB" --k 4
    echo "[prep] 完成: $MSEMB (工具自带垃圾特征守卫 + 体检报告)"
    ;;

  a) gpu_busy_guard; start_variant v15a "$CFGA" ;;
  b) gpu_busy_guard; start_variant v15b "$CFGB" ;;
  c) gpu_busy_guard; start_variant v15c "$CFGC" ;;

  serial)
    # 三个变体在**一个 tmux 会话**里前台串行 (a -> b -> c, 各 150k 步, ~26h)
    gpu_busy_guard
    TS=$(date +%Y%m%d-%H%M%S)
    tmux new-session -d -s v15 \
      "export PYTHONPATH=/root/Workspace/xy/DiT TORCHINDUCTOR_CACHE_DIR=/root/.cache/torch/inductor; \
       $PY -u src/train/train.py --config $CFGA 2>&1 | tee $LOGDIR/v15a_train_$TS.log; \
       $PY -u src/train/train.py --config $CFGB 2>&1 | tee $LOGDIR/v15b_train_$TS.log; \
       $PY -u src/train/train.py --config $CFGC 2>&1 | tee $LOGDIR/v15c_train_$TS.log"
    echo "[serial] v15a -> v15b -> v15c 已排队 (tmux v15, ~26h)。"
    echo "[serial] 日志: tail -f $LOGDIR/v15a_train_$TS.log"
    echo "[serial] 每个变体启动后确认: [callig-emb] 加载预训练多模态风格表 (87, 1536);"
    echo "         [style-anchor] λ=0.01 mode=mean (87, 384); 第一个 eval 点 2.5k 出数。"
    ;;

  ratio)
    CKPT="${2:-$(ls -t assets/results/v15*/*/checkpoints/*.pt 2>/dev/null | head -1)}"
    if [ -z "$CKPT" ]; then echo "[ratio] ✗ 找不到 ckpt, 传第二个参数指定"; exit 1; fi
    echo "[ratio] ckpt=$CKPT (判据: ratio_style, 基线 1.18 -> 目标显著上升并向 5~10 靠)"
    $PY tools/eval_diversity.py --ckpt "$CKPT" \
        --inter-csv "$CSV" \
        --inter-skel-shards data/50k/shards_std \
        --callig-script-map "$MAP" \
        --skip-intra --inter-char 8 --inter-callig 12 --cfg 1.0 \
        --device "${DEVICE:-cpu}" \
        2>&1 | tee "$LOGDIR/v15_ratio_$(date +%Y%m%d-%H%M%S).log"
    ;;

  *)
    echo "用法: $0 {prep|a|b|c|serial|ratio [ckpt]}"; exit 1;;
esac
