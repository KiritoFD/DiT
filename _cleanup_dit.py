# -*- coding: utf-8 -*-
"""
_cleanup_dit.py — 清理 src/model/dit.py 中的死代码（旧模型类 + XL 变体）。

按**顶级定义名**定位并删除，而不是硬编码行号 —— 行号会随编辑漂移，
按名字匹配更稳健，且能明确知道删掉了什么。

删除清单
--------
1. 原版 DiT（单条件，依赖 timm）及其组件 DiTBlock / FinalLayer
2. DiT_3Cond（三条件：callig+script+char，已被 DiT_2Cond 取代）及其所有变体
3. DiT_2Cond_XL_2（XL 变体，从未在当前 pipeline 使用）
4. timm 依赖（只有上述旧类在用）

保留
----
- TimestepEmbedder / LabelEmbedder（DiT_2Cond 在用）
- get_2d_sincos_pos_embed 系列（DiT_2Cond.initialize_weights 在用）
- DiT_2Cond 及其 S/2 等变体（当前主干）
- modulate（DiT 标准函数，无害）

安全策略
--------
- 只删**顶级**定义（行首无缩进的 def/class），不动类内方法
- 精确匹配名字（避免 DiT_2Cond 被 DiT_2Cond_XL_2 的前缀匹配误伤）
- 删除前先备份为 .bak，并打印被删内容的前几行供人工确认
"""
import re, os, sys, shutil

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

TARGET = "src/model/dit.py"

# 精确匹配的顶级定义名
DELETE_DEFS = [
    # --- 旧版 timm 组件（仅原版 DiT / DiT_3Cond 使用）---
    "DiTBlock", "FinalLayer",
    # --- 原版单条件 DiT ---
    "DiT",
    "DiT_XL_2", "DiT_XL_4", "DiT_XL_8",
    "DiT_L_2", "DiT_L_4", "DiT_L_8",
    "DiT_B_2", "DiT_B_4", "DiT_B_8",
    "DiT_S_2", "DiT_S_4", "DiT_S_8",
    "DiT_models",
    # --- 三条件 DiT（已废弃）---
    "DiT_3Cond",
    "DiT_3Cond_S_2", "DiT_3Cond_B_2", "DiT_3Cond_L_2", "DiT_3Cond_XL_2",
    "DiT_3Cond_models",
    # --- XL 变体 ---
    "DiT_2Cond_XL_2",
]

# 精确匹配的语句行（用正则匹配整行）
DELETE_LINE_PATTERNS = [
    r"^from timm\.models\.vision_transformer import .*$",
]


def find_top_level_blocks(lines):
    """返回 [(name, start, end)]，end 为下一个顶级定义前的最后一行（不含）。"""
    blocks = []
    cur_name, cur_start = None, None
    for i, ln in enumerate(lines):
        if ln.startswith("def ") or ln.startswith("class "):
            m = re.match(r"^(?:def|class)\s+([A-Za-z_][A-Za-z0-9_]*)", ln)
            if m:
                if cur_name is not None:
                    blocks.append((cur_name, cur_start, i))
                cur_name, cur_start = m.group(1), i
    if cur_name is not None:
        blocks.append((cur_name, cur_start, len(lines)))
    return blocks


def trim_trailing_blank(lines, end, start):
    """把 end 往前收，去掉被删块尾部的连续空行（保留一个分隔）。"""
    e = end
    while e - 1 > start and lines[e - 1].strip() == "":
        e -= 1
    return e


def main():
    with open(TARGET, encoding="utf-8") as f:
        lines = f.readlines()

    blocks = find_top_level_blocks(lines)
    by_name = {n: (s, e) for n, s, e in blocks}

    # 标记要删除的行
    drop = set()
    removed = []
    for name in DELETE_DEFS:
        if name not in by_name:
            print(f"  [skip] {name}: 未找到顶级定义")
            continue
        s, e = by_name[name]
        e = trim_trailing_blank(lines, e, s)
        for i in range(s, e):
            drop.add(i)
        removed.append((name, s + 1, e, lines[s].rstrip()[:70]))

    for pat in DELETE_LINE_PATTERNS:
        rx = re.compile(pat)
        for i, ln in enumerate(lines):
            if rx.match(ln.rstrip("\n")):
                drop.add(i)
                removed.append(("<import/statement>", i + 1, i + 1,
                                ln.rstrip()[:70]))

    if not drop:
        print("# 没有需要删除的内容")
        return

    print(f"# 将删除 {len(drop)} 行，来自 {len(removed)} 个定义/语句：\n")
    for name, s, e, head in sorted(removed, key=lambda x: x[1]):
        print(f"  L{s:>5}-{e:<5} {name:<22} {head}")

    shutil.copy(TARGET, TARGET + ".bak")
    print(f"\n# 已备份 -> {TARGET}.bak")

    out = [ln for i, ln in enumerate(lines) if i not in drop]
    with open(TARGET, "w", encoding="utf-8", newline="") as f:
        f.writelines(out)

    print(f"# 已重写 {TARGET}: {len(lines)} -> {len(out)} 行 "
          f"(-{len(lines)-len(out)})")

    # 校验：保留的关键定义仍在
    remain = {n for n, _, _ in find_top_level_blocks(out)}
    must_keep = ["TimestepEmbedder", "LabelEmbedder", "DiT_2Cond",
                 "DiT_2Cond_S_2", "get_2d_sincos_pos_embed", "modulate"]
    print("\n# 校验保留项：")
    ok = True
    for k in must_keep:
        present = k in remain
        if not present:
            ok = False
        print(f"  {'OK ' if present else 'MISSING'} {k}")
    print("\n# 校验删除项：")
    for k in ("DiT_3Cond", "DiT_2Cond_XL_2", "DiT_models"):
        gone = k not in remain
        print(f"  {'OK 已删' if gone else 'STILL PRESENT'} {k}")
    print("\n# 结果:", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
