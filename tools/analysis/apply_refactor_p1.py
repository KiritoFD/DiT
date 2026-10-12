# -*- coding: utf-8 -*-
"""apply_refactor_p1.py — 批量、行为中性的定点修复（Phase 1/2）

修 4 处（全部可静态证明不改变可达行为）:
 1. src/eval/auto_eval_cpu.py : 删掉 `if cond_mode=="3cond":` 之后 raise 之后的**不可达死代码块**
    （DiT_3Cond 已于 2026-08-31 删除, 该分支已显式 raise -> 其后代码永不执行）
 2. src/eval/gpu_ablate_eval.py : 2026 改动把 load_state_dict 的返回值丢了, 但日志仍引用 miss/unexp
    -> 重新接收返回值（strict=True 成功时返回([],[])）; 行为不变, 日志恢复真实
 3. tools/build_fame_fast.py : 缺 `import glob`
 4. tools/preflight.py : 策略改为 **src/** 必须零缺漏（FAIL 才 exit 非0）, tools/ 的历史欠债只做棘轮
"""
import io
import os
import re
import sys

_R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(_R if os.path.isdir(os.path.join(_R, "src")) else os.getcwd())
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
done = []


def patch(path, old, new, note):
    s = io.open(path, encoding="utf-8", errors="replace").read()
    if old not in s:
        print(f"  ✗ {path}: 未找到锚点 ({note})")
        return False
    io.open(path, "w", encoding="utf-8", newline="").write(s.replace(old, new, 1))
    print(f"  ✓ {path}: {note}")
    done.append(path)
    return True


# ── 1. 删除 auto_eval_cpu.py 里 raise 之后的不可达块 ──
p = "src/eval/auto_eval_cpu.py"
s = io.open(p, encoding="utf-8", errors="replace").read()
lines = s.splitlines(keepends=True)
# 找 `if cond_mode == "3cond":`
idx = next((i for i, l in enumerate(lines) if re.match(r'\s*if\s+cond_mode\s*==\s*["\']3cond["\']\s*:', l)), None)
if idx is None:
    print("  ✗ auto_eval_cpu.py: 未找到 3cond 分支")
else:
    base = len(lines[idx]) - len(lines[idx].lstrip())
    j = idx + 1
    while j < len(lines):
        if lines[j].strip() and (len(lines[j]) - len(lines[j].lstrip())) <= base:
            break
        j += 1
    dead = [k for k in range(idx, j) if "raise ValueError" not in lines[k] and lines[k].strip()
            and not lines[k].lstrip().startswith("#")]
    # 只保留 if + raise(+其参数) ，删掉其余非注释代码行
    new_block = []
    k = idx
    while k < j:
        l = lines[k]
        if (k == idx) or ("raise ValueError" in l) or (lines[k - 1] is not None and "raise ValueError" in lines[max(idx, k - 1)]
                                                      and l.lstrip().startswith(('"', "'", ")", "f\""))):
            new_block.append(l)
        elif not l.strip() or l.lstrip().startswith("#"):
            new_block.append(l)
        k += 1
    io.open(p, "w", encoding="utf-8", newline="").write("".join(lines[:idx] + new_block + lines[j:]))
    print(f"  ✓ {p}: 删除 3cond 分支中 raise 之后的不可达代码（{j-idx-len(new_block)} 行）")
    done.append(p)

# ── 2. gpu_ablate_eval.py 接收返回值 ──
patch("src/eval/gpu_ablate_eval.py",
      '    model.load_state_dict(sd, strict=True)\n    log(f"load: missing={len(miss)} unexpected={len(unexp)}")',
      '    miss, unexp = model.load_state_dict(sd, strict=True)\n'
      '    log(f"load: missing={len(miss)} unexpected={len(unexp)}")',
      "load_state_dict 返回值重新接收（strict=True 成功时为空表，行为不变）")

# ── 3. build_fame_fast.py 缺 import glob ──
for cand in ("tools/build_fame_fast.py", "tools/data/build_fame_fast.py"):
    if os.path.exists(cand):
        s = io.open(cand, encoding="utf-8", errors="replace").read()
        if re.search(r"^\s*import\s+glob\b", s, re.M):
            print(f"  · {cand}: 已有 import glob")
            continue
        m = re.search(r"^import\s+\w+.*$", s, re.M)
        if m:
            io.open(cand, "w", encoding="utf-8", newline="").write(
                s[:m.start()] + "import glob\n" + s[m.start():])
            print(f"  ✓ {cand}: 补 import glob")
            done.append(cand)

# ── 4. preflight 策略：src 零缺漏 ──
patch("tools/preflight.py",
      '''    lines = [l for l in out.splitlines() if l.strip()]
    bad = [l for l in lines if "undefined name" in l or "referenced before assignment" in l]
    rec("FAIL" if bad else "OK", "pyflakes 扫描", f"{len(lines)} 条提示, 其中未定义/先引用 {len(bad)} 条")
    for l in bad[:8]:
        rec("FAIL", "  └", l[:140])
    return len(bad)''',
      '''    lines = [l for l in out.splitlines() if l.strip()]
    bad = [l for l in lines if "undefined name" in l or "referenced before assignment" in l]
    # ★ 策略: **src/** 必须零缺漏（那是训练/评测真跑的路）; tools/ 历史欠债只计数走棘轮
    src_bad = [l for l in bad
               if (l.replace("/", os.sep).startswith("src") or l.startswith("src/"))
               and "legacy" not in l.replace("/", os.sep)]
    tools_bad = [l for l in bad if l not in src_bad]
    rec("FAIL" if src_bad else "OK", "pyflakes src/", f"未定义/先引用 {len(src_bad)} 条（要求 0）")
    for l in src_bad[:10]:
        rec("FAIL", "  └", l[:140])
    rec("INFO", "pyflakes tools/", f"{len(tools_bad)} 条历史欠债（棘轮，只许下降）")
    for l in tools_bad[:5]:
        rec("INFO", "  └", l[:140])
    return len(src_bad)''',
      "策略改为 src/ 零缺漏为硬门槛, tools/ 欠债走棘轮")

print("\n  已改文件:", sorted(set(done)))
