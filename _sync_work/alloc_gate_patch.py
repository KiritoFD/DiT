"""在 src/train/train.py 顶部焊一道"禁止 expandable_segments"的代码闸门。

用户裁定 (2026-10-03): 绝不允许 expandable_segments。
PyTorch 的 OOM 报错会主动建议 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ——
那是报错文案, 不是解决方案。这里在**任何 CUDA 分配之前**强制剔除该环境变量
(torch 的 caching allocator 在首次 CUDA 分配时才读它, 所以此处删除有效)。

事务式 + 幂等。
"""
import io
import os
import shutil
import sys
import time

ROOT = "/root/Workspace/xy/DiT"
os.chdir(ROOT)
TR = "src/train/train.py"

ANCHOR = "from src.model import DiT_2Cond_models\n"
MARK = "[alloc-gate]"

GATE = '''# ═══════════════════════════════════════════════════════════════════════════
# ★★ 铁律 (2026-10-03, 用户裁定): **绝不启用 expandable_segments** ★★
#   PyTorch 的 OOM 报错会主动建议 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,
#   **看到就忽略** —— 那是报错文案, 不是解决方案; 本项目已判定不允许。
#   这道闸门在**任何 CUDA 分配之前**强制剔除该环境变量: torch 的 caching
#   allocator 是在**首次 CUDA 分配时**才读它, 所以在此删除有效 (import torch
#   之后再删就晚了)。
#   显存不足时的正确应对: 调小 batch / 换省显存实现, **不许动分配器**。
# ═══════════════════════════════════════════════════════════════════════════
import os as _os_alloc_gate

for _k in ("PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF"):
    _v = _os_alloc_gate.environ.get(_k)
    if _v and "expandable" in _v.lower():
        _os_alloc_gate.environ.pop(_k, None)
        print(f"[alloc-gate] ✗ 检测到 {_k}={_v!r} -> **已强制剔除** "
              f"(铁律: 不允许 expandable_segments)", flush=True)
del _k, _v
'''


def main():
    src = io.open(TR, encoding="utf-8").read()
    if MARK in src:
        print(f"[skip] {TR} 已有闸门 (幂等)")
        return 0
    if ANCHOR not in src:
        print(f"[FATAL] 找不到锚点 {ANCHOR!r} —— 不做任何修改")
        return 2
    n = src.count(ANCHOR)
    ts = time.strftime("%Y%m%d-%H%M%S")
    bdir = f"_sync_work/_AB_patch/backup_gate_{ts}"
    os.makedirs(bdir, exist_ok=True)
    shutil.copy2(TR, os.path.join(bdir, "train.py"))
    print(f"[backup] -> {bdir}/train.py")
    new = src.replace(ANCHOR, GATE + ANCHOR, 1)
    io.open(TR, "w", encoding="utf-8", newline="").write(new)
    print(f"[write] {TR}  (锚点匹配 {n} 处, 在第 1 处之前插入闸门)")
    import py_compile
    try:
        py_compile.compile(TR, doraise=True, cfile="/tmp/_chk3.pyc")
        print("  ✓ 语法通过")
    except py_compile.PyCompileError as e:
        print(f"  ✗ 语法错误\n{e}")
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
