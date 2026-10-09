#!/usr/bin/env bash
# launch_v72.sh — v72 (冻结新 Calli-VAE 预编码 latent + DiT) 启动器
#
# 设计原则 (2026-10-09, 防混淆):
#   1. **预检即契约**: 配置/制品/约定/显存/重复进程 全部先验后启;
#   2. **清单落盘**: 把 VAE MD5、shards 的 PROVENANCE.ok、data_csv 行数、git commit
#      写进 logs/v72_launch_manifest.json —— 任何一次 v72 run 都可溯源;
#   3. **命名不可混**: 旧 calli latent (`shards_img_aug_calli`, 裸 sample+非标准 decoder)
#      与新 latent (`shards_img_aug_calli_kl1e6_s12500_rsample`, sample*sf+标准 decoder)
#      名称不同且各有 PROVENANCE; 严禁互指。
#
# 用法:
#   bash tools/launch_v72.sh            # 预检 + 启动 (tmux 会话 v72)
#   bash tools/launch_v72.sh --check    # 只预检不启动
#   CORPUS=aug bash tools/launch_v72.sh # 用 3x 增强语料 (默认 noaug 原版)
set -euo pipefail

ROOT="${ROOT:-/root/Workspace/xy/DiT}"
PY="${PY:-/opt/conda/envs/cu121/bin/python}"
SES="${SES:-v72}"
CORPUS="${CORPUS:-noaug}"          # noaug | aug
CFG="$ROOT/src/train/configs/v72_callivae_kl1e6_s12500_frozen_sp_c2ot.json"
SHARDS="$ROOT/exp-std/data/shards_img_aug_calli_kl1e6_s12500_rsample"
VAE="$ROOT/experiments/vae_frozen/calli_vae_kl1e6_s12500_stdconv"
CSV_NOAUG="$ROOT/exp-std/csv/train_top10_noaug.csv"
CSV_AUG="$ROOT/exp-std/csv/train_top10_aug_sym.csv"
LOGDIR="$ROOT/logs"

cd "$ROOT"
fail=0
say() { printf '  %-46s %s\n' "$1" "$2"; }

echo "===== v72 预检 ($CORPUS 语料) ====="
# 1. 配置
[ -f "$CFG" ] && say "config" "OK $(basename "$CFG")" || { say "config" "缺失 ❌"; fail=1; }
# 2. VAE 冻结制品 + MD5
if [ -d "$VAE" ]; then
  ( cd "$VAE" && md5sum -c MD5SUMS.txt >/dev/null 2>&1 ) \
    && say "VAE 制品+MD5" "OK $(basename "$VAE")" || { say "VAE MD5" "不匹配 ❌"; fail=1; }
else say "VAE 制品" "缺失 ❌ $VAE"; fail=1; fi
# 3. shards + PROVENANCE
if [ -f "$SHARDS/PROVENANCE.json" ]; then
  # 注意: 系统 python3 可能是 3.6 (ASCII 默认编码), 读含中文的 json 会崩 -> 用 $PY + 显式 utf-8
  ok=$("$PY" -c "import json;print(json.load(open('$SHARDS/PROVENANCE.json', encoding='utf-8'))['ok'])" 2>/dev/null || echo False)
  n=$(ls "$SHARDS"/shard_*.npz 2>/dev/null | wc -l)
  std=$("$PY" -c "import json;print(json.load(open('$SHARDS/PROVENANCE.json', encoding='utf-8'))['latent_std_measured'])" 2>/dev/null || echo '?')
  l1=$("$PY" -c "import json;print(json.load(open('$SHARDS/PROVENANCE.json', encoding='utf-8'))['selfcheck_decode_l1'])" 2>/dev/null || echo '?')
  say "shards" "$n shards | 自检 ok=$ok | std=$std | decode L1=$l1"
  [ "$ok" = "True" ] || fail=1
else say "shards PROVENANCE" "缺失 ❌"; fail=1; fi
# 4. 语料 CSV
CSV="$CSV_NOAUG"; [ "$CORPUS" = "aug" ] && CSV="$CSV_AUG"
if [ -f "$CSV" ]; then say "data_csv ($CORPUS)" "OK $(basename "$CSV") $(($(wc -l < "$CSV")-1)) 行"
else say "data_csv ($CORPUS)" "缺失 ❌ $CSV (先跑 tools/derive_noaug_csv.py)"; fail=1; fi
# 4b. ★ 一致性断言: config 里的 data_csv 必须与 CORPUS 选择一致
#     (曾经踩过: 预检查的是 noaug, 但 config 里写的是 aug -> 实际跑了aug)
CFG_CSV=$("$PY" -c "import json;print(json.load(open('$CFG', encoding='utf-8')).get('data_csv',''))" 2>/dev/null || echo '')
say "config data_csv" "$CFG_CSV"
if [ "$CFG_CSV" = "${CSV#$ROOT/}" ]; then say "语料一致性" "OK (config == 预期)"
else say "语料一致性" "config=$CFG_CSV ≠ 预期=${CSV#$ROOT/} ❌ (改 config 或换 CORPUS)"; fail=1; fi
# 5. 显存 + 重复进程
free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
say "GPU free" "${free} MiB"
[ "${free:-0}" -gt 20000 ] || { say "GPU" "不足 (<20G), 先确认没有别的任务 ❌"; fail=1; }
if pgrep -f 'v72_callivae_kl1e6_s12500_froze[n]' >/dev/null; then say "重复进程" "已有 v72 在跑 ❌"; fail=1;
else say "重复进程" "无"; fi

if [ "${1:-}" = "--check" ]; then
  echo "===== 仅预检: $([ $fail -eq 0 ] && echo PASS || echo FAIL) ====="
  exit $fail
fi
[ $fail -eq 0 ] || { echo "预检失败, 不启动"; exit 1; }

# ---- 清单 ----
mkdir -p "$LOGDIR"
GITC=$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)
cat > "$LOGDIR/v72_launch_manifest.json" <<EOF
{
  "config": "src/train/configs/$(basename "$CFG")",
  "corpus": "$CORPUS",
  "data_csv": "${CSV#$ROOT/}",
  "latent_shards_dir": "${SHARDS#$ROOT/}",
  "latent_provenance": $(cat "$SHARDS/PROVENANCE.json"),
  "vae_dir": "${VAE#$ROOT/}",
  "vae_md5": "$(grep safetensors "$VAE/MD5SUMS.txt" | awk '{print $1}')",
  "git_commit": "$GITC",
  "launched": "$(date '+%F %T')"
}
EOF
echo "  清单: logs/v72_launch_manifest.json"

# ---- 启动 (tmux 会话, 日志 tee) ----
LOG="$LOGDIR/v72_train_$(date +%Y%m%d-%H%M%S).log"
if tmux has-session -t "$SES" 2>/dev/null; then
  echo "tmux 会话 $SES 已存在, 请先 tmux kill-session -t $SES"; exit 1
fi
tmux new-session -d -s "$SES" \
  "cd $ROOT && PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 $PY -u -m src.train.train --config $CFG 2>&1 | tee $LOG"
echo "===== 已启动: tmux $SES | 日志 $LOG ====="
sleep 20
tmux capture-pane -t "$SES" -p 2>/dev/null | grep -v '^$' | tail -8 || true
