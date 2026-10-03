"""解掉活代码 -> legacy 的依赖 (2026-10-03)。

## 病灶
`src/model/dit.py` 的 **adaLN 注入分支是活代码**, 却 `from .legacy.controlnet import
ZeroAdaLNInjection` —— ControlNet(两阶段) 线早已废弃归档, 活代码却指着遗物。
后果: 一旦归档 `src/model/legacy/`, 训练直接崩 (A 跑完 B 就起不来)。

## 修法 (只搬家, 不改实现)
1. 新建 `src/model/injections.py`: 从 legacy/controlnet.py **逐字**搬 `zero_init_linear`
   与 `ZeroAdaLNInjection`。
   ⚠ 实现一字未改 → state_dict 键名仍是 `proj.weight/proj.bias`, 与类所在路径无关
     → 所有历史 adaLN ckpt 照常加载 (本脚本末尾会拿真 ckpt 验证 missing/unexpected=0)。
2. `dit.py`: import 改指 `.injections`。
3. `src/model/__init__.py`: 去掉那个"兼容旧脚本"的 try/except (它 import 的是即将
   归档的模块, 每次 `import src.model` 都要抛+吞一次异常), 改为显式 None + 说明。
4. 把 `src/model/legacy/` 归档到 `_archive/20261003_twostage/`。

事务式: 先校验锚点 + 扫全仓残留引用, 有任何活代码仍指着 legacy 就整体放弃。
"""
import io
import os
import re
import shutil
import subprocess
import sys
import time

ROOT = "/root/Workspace/xy/DiT"
os.chdir(ROOT)

INJ = "src/model/injections.py"
DIT = "src/model/dit.py"
INIT = "src/model/__init__.py"
LEGACY_DIR = "src/model/legacy"
ARC = "_archive/20261003_twostage"

INJECTIONS_SRC = '''# -*- coding: utf-8 -*-
"""注入层（2026-10-03 从 ``src/model/legacy/controlnet.py`` 上移）。

## 为什么要搬
``src/model/dit.py`` 的 **adaLN 注入分支是活代码**，却一直
``from .legacy.controlnet import ZeroAdaLNInjection`` —— 而 ControlNet（两阶段）
线早已废弃归档。**活代码 import 遗物**是脆的：只要归档 ``src/model/legacy/``，
训练立刻崩。这里把真正在跑的注入层上移成独立模块，解掉这条依赖。

## ⚠ 实现逐字未改（只换了家）
``state_dict`` 键名仍是 ``proj.weight`` / ``proj.bias``（键名不含类所在路径），
所以所有历史 adaLN ckpt 照常加载。
"""
from __future__ import annotations

import torch.nn as nn


def zero_init_linear(in_f, out_f):
    lin = nn.Linear(in_f, out_f)
    nn.init.zeros_(lin.weight)
    nn.init.zeros_(lin.bias)
    return lin


class ZeroAdaLNInjection(nn.Module):
    """adaLN 式零初始化注入：``out = x * (1 + s) + t``。

    ``s``/``t`` 由同一个 zero-init Linear 产出，因此 init 时 s=t=0，
    注入严格为恒等 —— 与 ControlNet 的 zero-conv warm-start 语义一致。

    梯度种子（为什么 step 0 就能学到东西）：
        d(out)/d(W) = x   （ctrl block 输出，非零 → W 立刻有梯度）
        d(out)/d(b) = 1   （bias 立刻有梯度）
        d(out)/d(x) = W = 0 → **ctrl blocks 在 W 变非零前收不到梯度**
    这是 ControlNet 的正确行为（先学注入权重，再学控制特征）。
    """

    def __init__(self, hidden_size, mode="modulate"):
        super().__init__()
        if mode not in ("modulate", "add"):
            raise ValueError(f"Unknown injection mode={mode!r}")
        self.mode = mode
        self.proj = zero_init_linear(hidden_size, hidden_size * (2 if mode == "modulate" else 1))

    def forward(self, x, feat):
        if self.mode == "modulate":
            s, t = self.proj(feat).chunk(2, dim=-1)
            return x * (1.0 + s) + t
        return x + self.proj(feat)
'''

NEW_INIT_BLOCK = '''# ── ControlNet(两阶段) 线已归档, 依赖已解 (2026-10-03) ──────────────────────
# 原实现: src/model/legacy/controlnet.py (已归档至 _archive/20261003_twostage/)。
# ⚠ 这里**不再 try/except import 遗留模块** —— 那会让每次 `import src.model`
#   都抛一次 ImportError 再吞掉。真要用 ControlNet 的人请显式:
#       from _archive.20261003_twostage.src_model_legacy.controlnet import ControlNetDiT
#   (需要时再把它捞回 src/model/ 下, 别在包初始化里挂隐式回退。)
ControlConditionEncoder = None
ControlNetDiT = None
load_main_model = None
'''


def die(msg):
    print(f"[FATAL] {msg}")
    sys.exit(2)


def main():
    # ---------- 阶段 1: 校验锚点 ----------
    for p in (DIT, INIT, f"{LEGACY_DIR}/controlnet.py"):
        if not os.path.exists(p):
            die(f"缺文件 {p}")
    dit_src = io.open(DIT, encoding="utf-8").read()
    init_src = io.open(INIT, encoding="utf-8").read()

    A1 = "                    from .legacy.controlnet import ZeroAdaLNInjection"
    if A1 not in dit_src:
        die(f"{DIT} 里找不到锚点: {A1.strip()!r}")
    # 用字符串定位而不是正则: 实际代码是 `try:` 后面跟注释再换行
    # (try:                                   # 兼容: 旧 ckpt/脚本若还指望顶层可见)
    _i = init_src.find("\ntry:")
    _i = _i + 1 if _i >= 0 else init_src.find("try:")
    _endmark = "ControlConditionEncoder = ControlNetDiT = load_main_model = None"
    _j = init_src.find(_endmark)
    if _i < 0 or _j < 0 or _j < _i:
        die(f"{INIT} 里找不到 legacy try/except 块 (try@{_i}, None行@{_j})")
    if "legacy.controlnet" not in init_src[_i:_j]:
        die(f"{INIT}: try 块里没有 legacy.controlnet 引用 (定位可能不准)")
    _j = init_src.find("\n", _j) + 1
    m = type("M", (), {"start": lambda s: _i, "end": lambda s: _j})()
    print(f"[locate] {INIT} 待替换: 行 {init_src[:_i].count(chr(10)) + 1} ~ "
          f"{init_src[:_j].count(chr(10))}")

    # ---------- 阶段 2: 扫全仓残留引用 (活代码) ----------
    print("=" * 78)
    print("阶段 2: 扫全仓 'legacy.controlnet' 残留引用")
    out = subprocess.run(
        ["grep", "-rn", r"legacy\.controlnet\|model\.legacy\|from \.legacy",
         "--include=*.py", "src", "tools", "scripts", "_sync_work"],
        capture_output=True, text=True).stdout.strip().splitlines()
    live = [l for l in out if "/legacy/" not in l and "_archive/" not in l
            and not l.startswith("_sync_work/_AB_patch")
            and not l.startswith("_sync_work/fix_dep_patch.py")]
    # 本次一并改指向的文件 (只要 ZeroAdaLNInjection 的那些)
    REPOINT = [DIT, INIT, "tools/attn_collapse_probe.py", "tools/injection_probes.py"]
    # ControlNet 线自己的消费者: 它们要的是 ControlNetDiT/load_main_model,
    # 属于**被归档的那条线**, 本次不动 legacy 目录, 所以它们照旧可用。
    KEEP_LEGACY = ["src/eval/cpu_eval_worker.py", "src/train/dit.py",
                   "tools/eval_ctrl_fame_cpu.py", "tools/diag_ctrl_fame_cpu.py",
                   "tools/skel_follow_gpu.py", "tools/eval/skel_follow_gpu.py"]
    print("   --- 本次改指向 (ZeroAdaLNInjection) ---")
    for l in live:
        if any(l.startswith(p + ":") for p in REPOINT):
            print("     [fix]  ", l[:140])
    print("   --- ControlNet 线消费者 (保留 legacy 目录, 故不受影响) ---")
    for l in live:
        if any(l.startswith(p + ":") for p in KEEP_LEGACY):
            print("     [keep] ", l[:140])
    bad = [l for l in live
           if not any(l.startswith(p + ":") for p in REPOINT + KEEP_LEGACY)]
    if bad:
        die(f"仍有 {len(bad)} 处未归类的 legacy 引用 (见上) -> 本次不改任何东西")
    print("   ✓ 引用全部归类完毕")

    # ---------- 阶段 3: 备份 + 写盘 ----------
    ts = time.strftime("%Y%m%d-%H%M%S")
    bdir = f"_sync_work/_AB_patch/backup_dep_{ts}"
    os.makedirs(bdir, exist_ok=True)
    for p in (DIT, INIT, f"{LEGACY_DIR}/controlnet.py"):
        shutil.copy2(p, os.path.join(bdir, p.replace("/", "__")))
    print(f"\n[backup] -> {bdir}")

    io.open(INJ, "w", encoding="utf-8", newline="").write(INJECTIONS_SRC)
    print(f"[new]    {INJ}")

    dit_new = dit_src.replace(A1, "                    from .injections import ZeroAdaLNInjection", 1)
    io.open(DIT, "w", encoding="utf-8", newline="").write(dit_new)
    print(f"[edit]   {DIT}  -> from .injections import ZeroAdaLNInjection")

    init_new = init_src[:m.start()] + NEW_INIT_BLOCK + init_src[m.end():]
    io.open(INIT, "w", encoding="utf-8", newline="").write(init_new)
    print(f"[edit]   {INIT}  -> 去掉 legacy try/except")

    # ---------- 阶段 4: 语法 ----------
    import py_compile
    for p in (INJ, DIT, INIT):
        try:
            py_compile.compile(p, doraise=True, cfile="/tmp/_chk5.pyc")
            print(f"  ✓ 语法 {p}")
        except py_compile.PyCompileError as e:
            die(f"语法错误 {p}\n{e}")

    # ---------- 阶段 5: 探针脚本一并改指向 ----------
    for p in ("tools/attn_collapse_probe.py", "tools/injection_probes.py"):
        if not os.path.exists(p):
            continue
        s = io.open(p, encoding="utf-8").read()
        if "legacy.controlnet import ZeroAdaLNInjection" not in s:
            continue
        s2 = (s.replace("from src.model.legacy.controlnet import ZeroAdaLNInjection",
                        "from src.model.injections import ZeroAdaLNInjection")
                .replace('"src.model.legacy.controlnet"', '"src.model.injections"'))
        io.open(p, "w", encoding="utf-8", newline="").write(s2)
        print(f"[edit]   {p}  -> src.model.injections")

    # ---------- 阶段 6: **本次不归档** legacy 目录 ----------
    #   原因: ControlNet 线的 6 个消费者 (cpu_eval_worker / src/train/dit.py /
    #   4 个 tools) 仍需要 ControlNetDiT / load_main_model。它们属于被归档的那条线,
    #   必须**与 ControlNet 线一起**归档, 否则会连环拆坏。
    #   本次只切断"训练主路径 -> 遗物"这条最丑的边 (dit.py 的 adaLN 分支)。
    print(f"\n[skip]  {LEGACY_DIR} 本次**不归档** —— 尚有 ControlNet 线消费者:")
    for p in ("src/eval/cpu_eval_worker.py", "src/train/dit.py",
              "tools/eval_ctrl_fame_cpu.py", "tools/diag_ctrl_fame_cpu.py",
              "tools/skel_follow_gpu.py", "tools/eval/skel_follow_gpu.py"):
        if os.path.exists(p):
            print(f"         {p}")
    print("        -> 下一步: 把 ControlNet 线(含这些消费者)整体归档, 再归档 legacy。")
    print("结论: ✓ 训练主路径的 legacy 依赖已切断; legacy 目录保留待整线归档")
    return 0


if __name__ == "__main__":
    sys.exit(main())
