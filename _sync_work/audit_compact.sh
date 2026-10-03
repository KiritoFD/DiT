#!/usr/bin/env bash
# 紧凑审计: 每项只输出 裁决 + 一行证据
set -u
cd /root/Workspace/xy/DiT || exit 1
V() { printf '%-58s %s\n' "$1" "$2"; }
E() { echo "        ↳ ${1:0:150}"; }

echo "======== 审计裁决 (远端实况取证) ========"

# --- P0-2 指标齐备性 ---
A=$(grep -c 'ink_iou\|ink_iou_mean' src/eval/in_mem_eval.py)
B=$(grep -c 'cal_enrich' src/eval/in_mem_eval.py)
C=$(grep -ci 'white\|blank_baseline\|白纸' src/eval/in_mem_eval.py)
D=$(grep -c 'frag_ratio\|hole_pred' src/eval/in_mem_eval.py)
V "P0-2a ink_iou/ink_ssim/frag/hole 已实现" "$([ "$A" -gt 0 ] && [ "$D" -gt 0 ] && echo '成立(已实现)' || echo '不成立')"
V "P0-2b 书家分类探针(cal_enrich/tgt_spec) 已实现" "$([ "$B" -gt 0 ] && echo '成立(已实现)' || echo '不成立')"
V "P0-2c '白纸'基线 已实现" "$([ "$C" -gt 0 ] && echo '成立' || echo '★不成立(缺)')"
grep -n 'cal_enrich = \|tgt_spec = \|def ' src/eval/in_mem_eval.py | grep -i 'enrich\|spec\|nn' | head -2 | sed 's/^/        ↳ /'

# --- P0-3 min_delta 量程 ---
V "P0-3 早停 min_delta 相对 skel_iou 量程过大" "$(grep -aq 'skel_iou.*0.005' src/train/early_stop.py src/train/cli.py 2>/dev/null && echo '★成立' || echo '待定')"
grep -n 'min_delta\|0.005\|0.002' src/train/early_stop.py | head -4 | sed 's/^/        ↳ /'
grep -a 'early-stop. metric' exp-std/logs_AB/A_*.log 2>/dev/null | tail -1 | cut -c1-170 | sed 's/^/        ↳ 实测: /'

# --- P1-1 latent_dataset np.empty / 缺失 id ---
V "P1-1a latents 用 np.empty (未初始化内存)" "$(grep -q 'np\.empty' src/utils/latent_dataset.py && echo '★成立' || echo '不成立(用的是 torch.empty)')"
V "P1-1b 缺失 id 直接报错" "$(grep -q 'KeyError.*latent not found' src/utils/latent_dataset.py && echo '已满足(报错)' || echo '不成立')"
grep -n 'np\.empty' src/utils/latent_dataset.py | head -3 | sed 's/^/        ↳ /'

# --- P1-2 fresh-scheduler ---
V "P1-2 fresh_scheduler 默认路径已关闭" "$(grep -q 'fresh_scheduler.*default=False' src/train/cli.py && echo '已满足(默认 False)' || echo '待查')"
V "P1-2b 绝对步数校验存在" "$(grep -q 'early_stop_from_step' src/train/train.py && echo '已满足' || echo '★不成立')"

# --- P1-3 阶段启动护栏 ---
V "P1-3a 级联脚本 kill 旧进程" "$(grep -q 'pkill' _sync_work/run_stdmix_cascade.sh && echo '已满足' || echo '★不成立(缺)')"
V "P1-3b 级联脚本校验 GPU 空闲" "$(grep -q 'nvidia-smi' _sync_work/run_stdmix_cascade.sh && echo '已满足' || echo '★不成立(缺)')"
V "P1-3c 我的 launch_AB 有 kill/GPU 校验" "$(grep -q 'nvidia-smi' _sync_work/launch_AB_100k.sh && echo '已满足' || echo '★不成立(缺)')"

# --- P1-4 数据侧 ---
V "P1-4a 存在繁简折叠修复工具" "$(ls tools/ 2>/dev/null | grep -qi 'fold\|simplif\|繁简' && echo '成立' || echo '★不成立(无)')"
V "P1-4b 存在 std shard x eval csv 全量 id 核对工具" "$(ls tools/ tools/*.py 2>/dev/null | grep -qi 'cross.?check\|coverage\|audit_id' && echo '成立' || echo '★不成立(无)')"
echo "        ↳ 已有实测证据: eval200 img 21968 真迹='復' 而 std 条件='复' (我们亲手修过)"

# --- P2-1 ---
V "P2-1a summary 有 stage/tag 列" "$(grep -q '"stage"' src/eval/in_mem_eval.py && echo '成立' || echo '★不成立(表头无 stage/tag)')"
V "P2-1b 训练集几何增强" "$(grep -qi 'rotate\|affine\|geometric' src/utils/latent_dataset.py && echo '成立' || echo '★不成立(无)')"
V "P2-1c _pred 缺失时回退 GT" "$(grep -qi 'fallback\|回退' src/eval/in_mem_eval.py && echo '待人工判读' || echo '不成立(未发现回退)')"

# --- P2-2 ---
V "P2-2a 有 _archive 目录" "$(ls -d _archive >/dev/null 2>&1 && echo '成立' || echo '★不成立')"
V "P2-2b 脚本里有 rsync 外备" "$(grep -qr 'rsync' _sync_work/ tools/ 2>/dev/null && echo '成立' || echo '★不成立(无备份机制)')"
echo "        ↳ git 未跟踪/未提交文件数: $(git status --porcelain 2>/dev/null | wc -l)"
