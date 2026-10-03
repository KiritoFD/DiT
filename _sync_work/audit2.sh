#!/usr/bin/env bash
# 第二轮审计取证 (用户给的 8 条)
set -u
cd /root/Workspace/xy/DiT || exit 1
V() { printf '%-56s %s\n' "$1" "$2"; }

echo "════ #1 版本控制是否真的断了 ════"
V "git ls-files src 数量" "$(git ls-files src 2>/dev/null | wc -l)"
V "git status 未跟踪(??)/修改(M)/删除(D)" "$(git status --porcelain 2>/dev/null | awk '{print $1}' | sort | uniq -c | tr '\n' ' ')"
V ".gitignore 是否存在" "$([ -f .gitignore ] && echo '存在' || echo '★不存在')"
echo "  --- 关键文件是否被跟踪 ---"
for f in src/model/dit.py src/eval/in_mem_eval.py src/utils/latent_dataset.py src/train/ckpt.py src/utils/deform_aug.py src/model/train.py; do
  t=$(git ls-files --error-unmatch "$f" >/dev/null 2>&1 && echo '已跟踪' || echo '★未跟踪')
  printf '    %-40s %s\n' "$f" "$t"
done
V "git 里最后一条提交" "$(git log -1 --format='%h %ad %s' --date=short 2>/dev/null | cut -c1-70)"
echo "  --- 会被误提交的大件 ---"
du -sh src/utils/std_glyph_latent_v2 2>/dev/null | sed 's/^/    /'
find . -name '__pycache__' -not -path './.git/*' 2>/dev/null | wc -l | sed 's/^/    __pycache__ 目录数: /'

echo
echo "════ #3 v50 配置 vs 证伪注册表 ════"
REG=docs/04_experiments/03_falsification_registry.md
V "注册表文件存在" "$([ -f "$REG" ] && echo '存在' || echo '不存在')"
if [ -f "$REG" ]; then
  grep -nE '^#|^\|' "$REG" | head -30 | sed 's/^/    /'
fi
echo "  --- v50 实际配置里被注册表点名过的键 ---"
grep -nE 'glyph_inject_mode|glyph_inject_layers|early_stop|min_delta|xattn' \
  src/train/configs/v50_A_space_xattn_style_adaLN_top10.json | sed 's/^/    /'
V "early_stop_min_delta_iou 是否在 v50 里" "$(grep -c 'min_delta_iou' src/train/configs/v50_A_space_xattn_style_adaLN_top10.json)"

echo
echo "════ #4 eval200_fixed 是否被真迹污染 (我自己那笔账) ════"
V "fix 脚本是否用真迹选条件字形" "$(grep -c 'corr(.*gt\|corr(new, gt)\|corr(_im, gt)' tools/fix_eval200_abs.py)"
echo "  --- 实际影响面 ---"
D=exp-std/runs_AB/20261003-212654-v50-A-space-xattn-style-adaln-top10
/opt/conda/envs/cu121/bin/python - <<'PY'
import csv, os
p = "exp-std/csv/eval200_fixed.csv"
if os.path.exists(p):
    rows = list(csv.DictReader(open(p, encoding="utf-8")))
    n = len(rows)
    diff = [r for r in rows if r.get("char_used") and r["char_used"] != r.get("character")]
    print(f"    {p}: {n} 行; char_used != character 的 = {len(diff)} 行 ({100*len(diff)/max(n,1):.2f}%)")
    for r in diff[:6]:
        print(f"      img_id={r.get('img_id')} 标注 '{r.get('character')}' -> 按 '{r.get('char_used')}' 渲染")
else:
    print("    (文件不存在)")
PY
echo "  --- 早停是否依赖 summary (承认过的静默失效) ---"
sed -n '110,120p' src/train/early_stop.py

echo
echo "════ #5 np.empty / 静默缺 id ════"
sed -n '360,372p' src/utils/latent_dataset.py
grep -n 'np.empty' src/utils/latent_dataset.py | sed 's/^/    /'
grep -n '会是全 0\|全 0\|_miss' src/utils/latent_dataset.py | head -6 | sed 's/^/    /'

echo
echo "════ #8 衰老代码清单 + 是否有引用者 (归档可行性) ════"
for f in src/model/train.py src/train/train_repa.py src/train/train_controlnet.py; do
  if [ -f "$f" ]; then
    n=$(wc -l < "$f")
    ref=$(grep -rl "$(basename $f .py)" --include=*.py --include=*.sh src _sync_work tools 2>/dev/null | grep -v "^$f$" | head -3 | tr '\n' ' ')
    printf '    %-34s %5s 行  引用者: %s\n' "$f" "$n" "${ref:-无}"
  fi
done
echo "    legacy 目录与文件数:"
for d in src/model/legacy src/train/legacy src/eval/legacy src/train/configs.bak_922; do
  [ -d "$d" ] && printf '      %-30s %s 个文件, %s\n' "$d" "$(find $d -type f | wc -l)" "$(du -sh $d | cut -f1)"
done
echo "    eval driver 数量 (src/eval/*.py + tools/*eval*.py):"
ls src/eval/*.py tools/*eval*.py 2>/dev/null | wc -l

echo
echo "════ (附) 早停指标 3σ 量程 (用于重设 min_delta) ════"
/opt/conda/envs/cu121/bin/python - <<'PY'
import glob, json, os, statistics as st
vals = {}
for f in glob.glob("exp-std/runs_AB/**/eval_auto_*.json", recursive=True):
    d = json.load(open(f, encoding="utf-8"))
    for k in ("ssim", "lpips", "skel_iou", "ink_iou", "frag", "mse"):
        if k in d:
            vals.setdefault(k, []).append(d[k])
print(f"    样本数={len(next(iter(vals.values()))) if vals else 0} (n<3 不足以估 σ)")
for k, v in vals.items():
    if len(v) >= 2:
        sd = st.pstdev(v)
        print(f"      {k:<10} 值={['%.4f'%x for x in v]}  σ={sd:.5f}  3σ={3*sd:.5f}  极差={max(v)-min(v):.5f}")
    else:
        print(f"      {k:<10} 值={['%.4f'%x for x in v]}  (点太少)")
PY
