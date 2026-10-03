"""彻底移除梯度检查点 (用户裁定: 绝对不允许)。

涉及 7 处:
  src/model/dit.py
    903:  use_checkpoint=True,                    <- 删 (构造参数)
    1148: self.use_checkpoint = use_checkpoint   <- 删 (属性)
    2445: if self.use_checkpoint: ... else: ...  <- 删整个 if 分支, 保留 else 里的正常循环
                                                   (并把正常循环**去掉 4 空格缩进**)
  src/train/train.py
    56:  from torch.utils.checkpoint import checkpoint as grad_ckpt   <- 删 (未被任何地方使用的僵尸 import)
    405: use_checkpoint=args.use_checkpoint                          <- 删 (构造 kwarg)
  src/eval/model_io.py
    73:  use_checkpoint=bool(g("use_checkpoint", False)),             <- 删
  src/train/cli.py
    178: --use-checkpoint ...                                          <- 删
  src/model/controlnet.py / src/model/train.py / src/model/modules.py
    legacy 路径里也传了该 kwarg / 提示文案 -> 一并清理

事务式: 先全量校验锚点, 任一缺失整体放弃。幂等: 已删则跳过。带备份 + 语法自检。
"""
import io
import os
import shutil
import sys
import time

ROOT = "/root/Workspace/xy/DiT"
os.chdir(ROOT)

DIT = "src/model/dit.py"
TR = "src/train/train.py"
MIO = "src/eval/model_io.py"
CLI = "src/train/cli.py"
CN = "src/model/controlnet.py"
MTR = "src/model/train.py"
MOD = "src/model/modules.py"


def split_branch_removal(src, cond_line):
    """把 `if <cond>:` ... `else:` <body> 改成只剩 <body> (body 缩进 -4)。"""
    lines = src.splitlines(keepends=True)
    idx = [i for i, l in enumerate(lines) if l.rstrip("\n") == cond_line]
    if not idx:
        return None, f"找不到 {cond_line!r}"
    i = idx[0]
    ind = len(lines[i]) - len(lines[i].lstrip())
    else_line = " " * ind + "else:"
    j = None
    for k in range(i + 1, len(lines)):
        if lines[k].rstrip("\n") == else_line:
            j = k
            break
        # 提前撞到同级或更浅的行 -> 说明没有 else 分支
        if lines[k].strip() and (len(lines[k]) - len(lines[k].lstrip())) <= ind:
            if k > i:
                return None, f"{cond_line!r} 后没有同级 else: (撞到 {lines[k]!r})"
    if j is None:
        return None, f"{cond_line!r} 后找不到 {else_line!r}"
    # body = j+1 起, 直到缩进 <= ind 的非空行
    k = j + 1
    while k < len(lines):
        if lines[k].strip() and (len(lines[k]) - len(lines[k].lstrip())) <= ind:
            break
        k += 1
    body = lines[j + 1:k]
    body = [l[4:] if l.startswith("    ") else l for l in body]      # 去 4 空格
    return lines[:i] + body + lines[k:], None


def main(apply_changes=True):
    targets = [DIT, TR, MIO, CLI, CN, MTR, MOD]
    srcs = {}
    for p in targets:
        if not os.path.exists(p):
            print(f"[skip] {p} 不存在")
            continue
        srcs[p] = io.open(p, encoding="utf-8").read()

    plan = []      # (path, 描述)
    new = dict(srcs)

    # ---- 1. dit.py: 删 if 分支 (结构手术) ----
    if DIT in new and "if self.use_checkpoint:" in new[DIT]:
        out, err = split_branch_removal(new[DIT], "        if self.use_checkpoint:")
        if err:
            print(f"[FATAL] {DIT} 分支手术失败: {err}")
            return 3
        new[DIT] = "".join(out)
        plan.append((DIT, "删除 if self.use_checkpoint: 整个分支, 保留正常循环(缩进-4)"))
    elif DIT in new:
        print(f"[skip] {DIT}: if self.use_checkpoint: 已不在 (幂等)")

    # ---- 2. 其余精确字符串删除 ----
    DELS = [
        (DIT, "        use_checkpoint=True,\n", "", "删 DiT 构造参数 use_checkpoint=True"),
        (DIT, "        self.use_checkpoint = use_checkpoint\n", "",
         "删 self.use_checkpoint 属性"),
        (TR, "from torch.utils.checkpoint import checkpoint as grad_ckpt\n", "",
         "删僵尸 import grad_ckpt (train.py 里从未使用)"),
        (TR, "            use_checkpoint=args.use_checkpoint,\n", "",
         "删 train 侧构造 kwarg"),
        (MIO, '        use_checkpoint=bool(g("use_checkpoint", False)),\n', "",
         "删评测侧构造 kwarg"),
        (CLI, '    parser.add_argument("--use-checkpoint", type=_str_to_bool, default=False,\n'
              '                        help="Enable gradient checkpointing on DiT blocks '
              '(cuts activation memory).")\n', "",
         "删 --use-checkpoint CLI 参数"),
        (CN, "                    use_checkpoint=False, learn_sigma=None,\n",
         "                    learn_sigma=None,\n", "controlnet.py: 去掉 kwarg"),
        (CN, "        use_checkpoint=use_checkpoint, learn_sigma=learn_sigma,\n",
         "        learn_sigma=learn_sigma,\n", "controlnet.py: 去掉 kwarg"),
        (MTR, "            use_checkpoint=args.use_checkpoint,\n", "",
         "src/model/train.py: 去掉 kwarg"),
        (MOD, "显存占用显著更高 —— 请相应调小 batch 或启用 use_checkpoint。",
         "显存占用显著更高 —— 请相应调小 batch。",
         "modules.py: 提示文案去掉 use_checkpoint"),
    ]
    for path, old, rep, desc in DELS:
        if path not in new:
            continue
        if old in new[path]:
            new[path] = new[path].replace(old, rep, 1)
            plan.append((path, desc))
        elif rep and rep in new[path]:
            print(f"[skip] {path}: {desc} (已处理)")
        elif not rep:
            print(f"[skip] {path}: {desc} (锚点已不在)")
        else:
            print(f"[FATAL] {path}: 找不到锚点 -> {desc}\n        {old.strip()[:90]!r}")
            return 4

    print("=" * 78)
    print(f"待应用 {len(plan)} 处:")
    for p, d in plan:
        print(f"  {p} :: {d}")
    print("=" * 78)
    if not plan:
        print("无需改动。")
        return 0
    if not apply_changes:
        return 0

    ts = time.strftime("%Y%m%d-%H%M%S")
    bdir = f"_sync_work/_AB_patch/backup_ckpt_{ts}"
    os.makedirs(bdir, exist_ok=True)
    for p in {x[0] for x in plan}:
        shutil.copy2(p, os.path.join(bdir, p.replace("/", "__")))
    print(f"[backup] -> {bdir}")

    for p, content in new.items():
        if content != srcs.get(p):
            io.open(p, "w", encoding="utf-8", newline="").write(content)
            print(f"[write] {p}")

    import py_compile
    ok = True
    for p in {x[0] for x in plan}:
        try:
            py_compile.compile(p, doraise=True, cfile="/tmp/_chk2.pyc")
            print(f"  ✓ 语法 {p}")
        except py_compile.PyCompileError as e:
            ok = False
            print(f"  ✗ 语法 {p}\n{e}")
    print("结论:", "✓ 梯度检查点已彻底移除" if ok else "✗ 语法错误")
    return 0 if ok else 5


if __name__ == "__main__":
    apply = "--dry" not in sys.argv
    sys.exit(main(apply))
