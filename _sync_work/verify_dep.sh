#!/usr/bin/env bash
# 依赖解开的验证 (纯 CPU, 不抢 GPU)
set -u
cd /root/Workspace/xy/DiT || exit 1
export PYTHONPATH=/root/Workspace/xy/DiT
export CUDA_VISIBLE_DEVICES=
PY=/opt/conda/envs/cu121/bin/python

echo "########## [1] 应用搬家补丁 ##########"
$PY -u _sync_work/fix_dep_patch.py 2>&1 | tail -22

echo
echo "########## [2] 残留引用应为空 ##########"
grep -rn 'legacy' --include=*.py src tools 2>/dev/null | grep -v '_archive' | head -5 || echo "  ✓ 全仓已无 legacy 引用"

echo
echo "########## [3] 关键: 搬家是否逐字未改 (新文件 vs 归档件) ##########"
$PY - <<'PY'
import io, re
new = io.open("src/model/injections.py", encoding="utf-8").read()
old = io.open("_archive/20261003_twostage/src_model_legacy/controlnet.py", encoding="utf-8").read()

def grab(src, name, endnames):
    i = src.find(f"class {name}" if name[0].isupper() else f"def {name}")
    if i < 0:
        return None
    j = len(src)
    for e in endnames:
        k = src.find(e, i + 5)
        if k > 0:
            j = min(j, k)
    return src[i:j].strip()

a = grab(old, "zero_init_linear", ["class "])
b = grab(new, "zero_init_linear", ["class "])
print("  zero_init_linear 逐字一致:", "✓" if a == b else f"✗\n    旧:{a!r}\n    新:{b!r}")

a = grab(old, "ZeroAdaLNInjection", ["class ControlConditionEncoder", "def _strip_compile"])
b = grab(new, "ZeroAdaLNInjection", ["# ---", "\nclass ", "\ndef "])
if b is None:
    b = new[new.find("class ZeroAdaLNInjection"):].strip()
if a and b:
    # 去掉两边的注释块再比 (新文件多了"搬家说明"注释, 实现应完全一致)
    na = re.sub(r"#.*", "", a); nb = re.sub(r"#.*", "", b)
    na = re.sub(r"\s+", " ", na).strip(); nb = re.sub(r"\s+", " ", nb).strip()
    print("  ZeroAdaLNInjection 实现逐字一致:", "✓" if na == nb else "✗ (有差异!)")
    if na != nb:
        print("    旧:", na[:200]); print("    新:", nb[:200])
PY

echo
echo "########## [4] 决定性: 用真 adaLN ckpt 验证键名未变 ##########"
$PY - <<'PY'
import glob
import torch
from src.model.injections import ZeroAdaLNInjection

m = ZeroAdaLNInjection(384)
print("  新建 ZeroAdaLNInjection(384) 的键:", list(m.state_dict().keys()))

ck = sorted(glob.glob("exp-std/runs_purestd/*v46-std-adaln4*/checkpoints/*.pt"))
print(f"  adaLN 历史 ckpt: {len(ck)} 个, 取 {ck[-1] if ck else '<无>'}")
if ck:
    d = torch.load(ck[-1], map_location="cpu", weights_only=False)
    sd = d.get("ema") or d.get("model") or d.get("delta") or d
    inj = sorted(k for k in sd if "glyph_injections" in k)
    print(f"  ckpt 里 glyph_injections 键数: {len(inj)}")
    for k in inj[:4]:
        print("     ", k)
    ok = inj and all(k.endswith("proj.weight") or k.endswith("proj.bias") for k in inj)
    print("  全部是 proj.weight/proj.bias:", "✓ 键名未变 (搬家安全)" if ok else "✗ 键名变了!")
PY

echo
echo "########## [5] preflight 两路配置仍可解析 ##########"
for cfg in v50_A_space_xattn_style_adaLN_top10 v51_B_joint_kv_k4_top10; do
  $PY tools/preflight_stage_args.py -u src/train/train.py \
      --config "src/train/configs/$cfg.json" --experiment-name chk \
      --results-dir exp-std/runs_smoke --global-batch-size 128 \
      --skel-latent-shards-weights 1.0,0.0 --max-steps 100 --lr 5e-5 2>&1 | tail -1
done

echo
echo "########## [6] 训练未受惊动 ##########"
A=$(ls -t exp-std/logs_AB/A_*.log 2>/dev/null | head -1)
grep -a 'Steps/Sec' "$A" 2>/dev/null | tail -1 | sed 's/\x1b\[[0-9;]*m//g'
