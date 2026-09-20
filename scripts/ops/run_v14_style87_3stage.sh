#!/usr/bin/env bash
# ============================================================================
# v14 (书家×书体)=87 类联合风格表 —— 三阶段训练编排
#
# 设计依据(见 src/utils/callig_script_map.py 与 docs/system/73):
#   单书家向量装不下多书体风格(ratio_style=1.18, r=-0.62) -> 把风格类别细化为
#   (书家,书体) 对(45->87)。三阶段(用户裁定, 不端到端):
#     Stage 1  对比学习(层级 SupCon)预训练 87 表 -> 冻结
#     Stage 2  冻结表, 训主干
#     Stage 3  冻结主干, 只微调表 + 锚定正则(防塌缩/漂移), wd=0.1 全程保留
#
# 用法(远端 /root/Workspace/xy/DiT):
#   bash scripts/ops/run_v14_style87_3stage.sh prep     # 建词表+提DINO+预训练表(前台,~几分钟)
#   bash scripts/ops/run_v14_style87_3stage.sh stage2   # 冻表训主干(tmux v14s2)
#   bash scripts/ops/run_v14_style87_3stage.sh stage3   # 冻主干微调表(tmux v14s3, 自动接 stage2 最新 ckpt)
#   bash scripts/ops/run_v14_style87_3stage.sh ratio    # 训练后测 ratio_style(判据, 非 strict ssim)
#
# ⚠ Windows 侧编辑过 -> 跑前确认 LF: sed -i 's/\r$//' 本文件
# ============================================================================
set -euo pipefail
cd /root/Workspace/xy/DiT
# 与 v13 系列 launcher 一致: train.py 用 `from src...` 绝对导入, 需 repo root 在 sys.path;
# inductor 缓存目录固定 -> 多阶段/重启复用编译 kernel。
export PYTHONPATH=/root/Workspace/xy/DiT
export TORCHINDUCTOR_CACHE_DIR=/root/.cache/torch/inductor

PY=/opt/conda/envs/cu121/bin/python
CSV=assets/train_50k_v2.csv
MAP=assets/callig_script_id_map.json
DINO=assets/dino_cls_50k.npz
EMB=assets/callig_script_emb_pretrained.pt
S2DIR=assets/results/v14_style87_s2
S3DIR=assets/results/v14_style87_s3
LOGDIR=logs/v14_series
mkdir -p "$LOGDIR"
STAGE="${1:-prep}"

case "$STAGE" in
  prep)
    echo "===== Stage 1a: 建 (书家×书体) 词表 ====="
    $PY -m src.utils.callig_script_map --csv "$CSV" --out "$MAP" --min-samples 50

    echo "===== Stage 1b: 提取 50k DINO CLS 特征(带 script) [GPU, ~1min] ====="
    # ⚠ 占用 GPU; 若训练在跑, 建议错峰或接受短暂争用(只前向, 不申请大显存)
    $PY tools/extract_dino_cls_50k.py --csv "$CSV" --out "$DINO"

    echo "===== Stage 1c: 层级 SupCon 预训练 87 表 [GPU, ~1min] ====="
    $PY tools/pretrain_callig_script_emb.py \
        --dino-npz "$DINO" --map "$MAP" --out "$EMB" \
        --w-sibling 0.3 --w-anchor 0.5 --steps 3000
    echo "[prep] 完成: $MAP / $DINO / $EMB"
    echo "[prep] 体检: 上面 pretrain 末尾的 'pairwise cos' 应明显 < 0.323(塌缩基线),"
    echo "        '同书家异书体 cos' 应为中等(共享身份但已分开)。"
    ;;

  stage2)
    TS=$(date +%Y%m%d-%H%M%S)
    echo "[stage2] tmux v14s2: 冻结 87 表, 训主干 (wd=0.1)"
    tmux new-session -d -s v14s2 \
      "$PY -u src/train/train.py --config src/train/configs/v14_style87_stage2.json \
       2>&1 | tee $LOGDIR/v14_s2_train_$TS.log"
    echo "[stage2] 已起。看日志: tail -f $LOGDIR/v14_s2_train_$TS.log"
    echo "[stage2] 启动后确认日志里有: [callig-script-map] num_calligraphers -> 87"
    echo "         以及 [callig-emb] 加载预训练书家表 ...: torch.Size([87, 128])"
    ;;

  stage3)
    CKPT=$(ls -t "$S2DIR"/*/checkpoints/*.pt 2>/dev/null | head -1 || true)
    if [ -z "$CKPT" ]; then echo "[stage3] ✗ 找不到 stage2 ckpt($S2DIR), 先跑完 stage2"; exit 1; fi
    TS=$(date +%Y%m%d-%H%M%S)
    echo "[stage3] 承接 ckpt: $CKPT"
    echo "[stage3] tmux v14s3: 冻结主干, 只微调表 + 锚定 λ=0.005 (wd=0.1)"
    tmux new-session -d -s v14s3 \
      "$PY -u src/train/train.py --config src/train/configs/v14_style87_stage3.json \
       --resume-full $CKPT --fresh-scheduler \
       2>&1 | tee $LOGDIR/v14_s3_train_$TS.log"
    echo "[stage3] 已起。确认日志: [train-only-callig-table] 冻结主干, 只训书家风格表: N 参数可训"
    echo "         以及 [style-anchor] λ=0.005 锚到预训练表 (87, 128)"
    echo "[stage3] ⚠ 早停判据 = ratio_style(见 'ratio' 分支), 不是 strict ssim; ratio 见顶回落即停。"
    ;;

  ratio)
    # 判据: 固定 书体+字+噪声, 只换书家 -> ratio_style = inter_callig/intra (cfg=1.0)
    # 用已修好的 tools/eval_diversity.py(doc68 §2.2 修过 D1~D3)。CKPT 传 stage3 的。
    CKPT="${2:-$(ls -t "$S3DIR"/*/checkpoints/*.pt 2>/dev/null | head -1)}"
    if [ -z "$CKPT" ]; then echo "[ratio] ✗ 找不到 ckpt, 传第二个参数指定"; exit 1; fi
    echo "[ratio] ckpt=$CKPT"
    # inter 阶段用训练集 csv(可控性测试, 非泛化); shards 必须与 50k 配套, 否则 g=ZERO
    $PY tools/eval_diversity.py --ckpt "$CKPT" \
        --inter-csv assets/train_50k_v2.csv \
        --inter-skel-shards data/50k/shards_std \
        --callig-script-map "$MAP" \
        --skip-intra --inter-char 8 --inter-callig 12 --cfg 1.0 \
        --device "${DEVICE:-cpu}" \
        2>&1 | tee "$LOGDIR/v14_ratio_$(date +%Y%m%d-%H%M%S).log"
    echo "[ratio] 看输出 'ratio_style': 基线(v13 单书家向量)=1.18, 目标显著 >1.18 并向 5~10 靠。"
    ;;

  all)
    # 串行一条龙: prep -> stage2(训到 max_steps) -> 取 stage2 最新 ckpt -> stage3。
    # 在**一个 tmux 会话前台**跑(本分支不再嵌套 tmux)。启动:
    #   tmux new-session -d -s v14 "bash scripts/ops/run_v14_style87_3stage.sh all \
    #       2>&1 | tee logs/v14_series/v14_all_$(date +%Y%m%d-%H%M%S).log"
    echo "[all] ===== Stage 1 (prep) ====="
    "$0" prep
    echo "[all] ===== Stage 2: 冻 87 表训主干 (前台) ====="
    $PY -u src/train/train.py --config src/train/configs/v14_style87_stage2.json
    CK2=$(ls -t "$S2DIR"/*/checkpoints/*.pt 2>/dev/null | head -1 || true)
    if [ -z "$CK2" ]; then echo "[all] ✗ stage2 无 ckpt, 终止"; exit 1; fi
    echo "[all] ===== Stage 3: 冻主干微调表, 承接 $CK2 ====="
    $PY -u src/train/train.py --config src/train/configs/v14_style87_stage3.json \
        --resume-full "$CK2" --fresh-scheduler
    echo "[all] 全部完成。ratio 判据: bash $0 ratio"
    ;;

  *)
    echo "用法: $0 {prep|stage2|stage3|ratio [ckpt]|all}"; exit 1;;
esac
