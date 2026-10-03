#!/usr/bin/env bash
# 对用户列出的审计项逐条取证 (远端实况)
set -u
cd /root/Workspace/xy/DiT || exit 1

hr() { echo; echo "════════ $* ════════"; }

hr "P0-1  in_mem_eval: 表头/句柄/自检"
echo "--- summary 打开与表头检查 (795~845) ---"
sed -n '795,845p' src/eval/in_mem_eval.py
echo "--- 写入与 flush (1020~1065) ---"
sed -n '1020,1065p' src/eval/in_mem_eval.py
echo "--- 是否有 '[eval] summary written' 或行数自检 ---"
grep -n 'summary written\|written.*rows\|rows.*written\|自检' src/eval/in_mem_eval.py | head -5 || echo "  (无)"
echo "--- 实况: 备份文件数 (每备份 1 次 = 曲线被重置 1 次) ---"
ls -1 exp-std/runs_AB/*bak_oldcols* 2>/dev/null | wc -l
ls -la exp-std/runs_AB/eval_stdskel_summary.csv 2>/dev/null || echo "  ⚠ 主 summary 文件不存在!"

hr "P0-2  主指标: 是否已有 ink_iou/ink_lpips/分类探针/三基线"
grep -n 'def .*metric\|ink_iou\|ink_lpips\|frag\|hole' src/eval/in_mem_eval.py | head -20
echo "--- 书家分类探针 / 同槽异字 / 白纸 基线 ---"
grep -rn 'cal_enrich\|tgt_spec\|samechar_nn\|white\|blank_baseline\|基线' src/eval/in_mem_eval.py | head -12
echo "--- 选模/早停用的指标 ---"
grep -n 'early_stop_metric\|iou_lpips\|best_metric' src/train/train.py | head -10

hr "P0-3  早停 min_delta 量程"
grep -n 'min_delta' src/train/early_stop.py | head -20
echo "--- 实况: 日志里的 min_delta 与 skel_iou 量程 ---"
grep -a 'early-stop. metric' exp-std/logs_AB/A_*.log 2>/dev/null | tail -1
grep -a 'skel_iou=' exp-std/logs_AB/A_*.log 2>/dev/null | tail -3

hr "P1-1  latent_dataset: np.empty / 缺失 id"
grep -n 'np.empty\|torch.empty' src/utils/latent_dataset.py | head -20
echo "--- 缺失 id 是报错还是 print ---"
grep -n 'print\|raise\|warn\|missing\|缺失' src/utils/latent_dataset.py | head -20

hr "P1-2  early_stop 语义 / fresh-scheduler"
grep -n 'fresh_scheduler\|fresh-scheduler' src/train/train.py src/train/cli.py | head -10
grep -n 'early_stop_from_step\|from_step' src/train/train.py | head -8

hr "P1-3  每阶段启动前 kill 旧进程 + 校验 GPU 空闲"
grep -n 'pkill\|nvidia-smi\|GPU 空闲\|CUDA_VISIBLE' _sync_work/run_stdmix_cascade.sh | head -10
grep -n 'pkill\|nvidia-smi' _sync_work/launch_AB_100k.sh | head -5 || echo "  launch_AB_100k.sh: 无 (我写的, 确实没有)"

hr "P1-4  数据侧 繁简折叠 / id 全量核对"
grep -rn '繁简\|fold\|simplified\|traditional' tools/*.py src/utils/*.py 2>/dev/null | head -10 || echo "  (未找到繁简折叠处理)"
ls tools/ | grep -i 'check\|verify\|audit' | head -10
echo "--- 实况证据: eval200 里 '复' 的真迹是 '復' (我们已实测) ---"
grep -ac '复' exp-std/csv/eval200.csv 2>/dev/null

hr "P2-1  exp 列 stage/tag / 几何增强 / _pred 回退 GT"
grep -n 'stage\|tag' src/train/train.py | grep -i 'append\|row\|col' | head -8 || echo "  (csv 列里无 stage/tag)"
grep -rn 'geometric\|随机旋转\|rotate\|affine\|augment' src/utils/latent_dataset.py | head -8 || echo "  (latent_dataset 无几何增强)"
grep -n '_pred\|pred_dir\|回退\|fallback.*gt\|gt.*fallback' src/eval/in_mem_eval.py | head -10

hr "P2-2  归档/备份机制"
ls -d _archive 2>/dev/null && ls _archive | head -5
grep -rn 'rsync' _sync_work/*.sh tools/*.sh 2>/dev/null | head -5 || echo "  (脚本里无 rsync)"
git -C . status --porcelain 2>/dev/null | wc -l
echo "  ↑ 未提交文件数 (含我们本轮所有新脚本)"
