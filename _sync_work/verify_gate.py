"""验证 expandable_segments 闸门: 位置正确 + 功能生效。"""
import io
import os
import re

TR = "/root/Workspace/xy/DiT/src/train/train.py"
src = io.open(TR, encoding="utf-8").read()
lines = src.splitlines()

# ---- 1. 闸门是否存在、在哪一行 ----
gate_lines = [i + 1 for i, l in enumerate(lines) if "[alloc-gate]" in l]
gate_block = [i + 1 for i, l in enumerate(lines) if "_os_alloc_gate" in l]
print("=== [1] 闸门位置 ===")
print(f"  含 [alloc-gate] 的行: {gate_lines}")
print(f"  含 _os_alloc_gate 的行: {gate_block[:6]}")

# 首次 CUDA 使用的位置 (闸门必须在此**之前**)
cuda_lines = [i + 1 for i, l in enumerate(lines)
              if re.search(r"torch\.cuda|\.cuda\(\)|to\(device\)|device=|cuda:", l)]
print(f"  首次出现 CUDA 相关调用的行: {cuda_lines[:5]}")
if gate_block and cuda_lines:
    if min(gate_block) < min(cuda_lines):
        print(f"  ✓ 闸门 (行 {min(gate_block)}) 在首次 CUDA 调用 (行 {min(cuda_lines)}) **之前**")
    else:
        print(f"  ✗✗ 闸门在 CUDA 调用之后! 无效")
else:
    print("  ⚠ 无法比较 (闸门或 CUDA 行未找到)")

# ---- 2. 功能测试: 提取闸门代码, 在脏环境变量下 exec ----
print("\n=== [2] 功能测试 (注入脏变量) ===")
m = re.search(r"(# ═+\n# ★★ 铁律.*?del _k, _v\n)", src, re.S)
if not m:
    print("  ✗ 提取不到闸门代码块")
else:
    code = m.group(1)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    print(f"  exec 前: PYTORCH_CUDA_ALLOC_CONF={os.environ.get('PYTORCH_CUDA_ALLOC_CONF')!r}")
    exec(compile(code, "<gate>", "exec"), {})
    v = os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
    print(f"  exec 后: {v!r}")
    print("  ✓ 闸门生效, 变量已被剔除" if v is None else "  ✗✗ 闸门无效, 变量仍在")

    # 干净环境下不应报错
    print("\n=== [3] 干净环境下应静默通过 ===")
    os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    try:
        exec(compile(code, "<gate>", "exec"), {})
        print("  ✓ 无异常")
    except Exception as e:                                    # noqa: BLE001
        print(f"  ✗ 异常: {type(e).__name__}: {e}")

# ---- 4. 梯度检查点的移除仍然干净 ----
print("\n=== [4] 梯度检查点应已彻底移除 ===")
for pat in ("use_checkpoint", "grad_ckpt", "torch.utils.checkpoint"):
    n = len([l for l in lines if pat in l])
    print(f"  {pat!r} 出现在 train.py 的行数 = {n}" + (" ✓" if n == 0 else " ✗"))
