"""把 gradio 的 ckpt 下拉改成**白名单精选约 10 个**（自动扫描有 1283 个，每实验3个也太多）。

白名单按已知 strict 质量排序，每个实验取步数最大的那个（少数取2个用于对照）。
"""
import io
import os

os.chdir("/root/Workspace/xy/DiT")
P = "gradio_stdskel.py"
s = io.open(P, encoding="utf-8").read()

START = s.index("def _scan_ckpts(")
END = s.index("CKPT_LIST = _scan_ckpts()")
old = s[START:END]

NEW = '''# ★ [2026-09-21] 精选白名单 —— 自动扫描有 1283 个 ckpt，下拉根本没法用。
#   (实验名, 该实验取几个最新步, 显示备注)
_PREFERRED = [
    ("v13_base_50k",                     1, "strict 0.5703 · 当前最佳"),
    ("v15b_supcon",                      1, "v15b + SupCon 词表(余弦0.0019) · 训练中"),
    ("v15a_multistyle_k4",               1, "strict 0.5699 · 多风格(旧词表)"),
    ("v15b_multistyle_k4",               1, "多风格 + CA(旧词表)"),
    ("v13_wd01",                         1, "wd=0.1 · 同step比base +0.0165"),
    ("v15c_multistyle_k4",               1, "多风格 + 每层xattn ctx"),
    ("v14_style87_fullft",               1, "87 书家 fullft"),
    ("v14_style87_s3",                   1, "87 书家 stage3"),
    ("v13_styletok",                     1, "style token 32 + xattn"),
    ("v10b_stdskel_fame3_c41x_cos_e",    1, "旧基线(原gradio所用)"),
]


def _scan_ckpts():
    """按白名单收集 ckpt: [(显示名, 路径)]，每个实验取步数最大的 N 个。"""
    import glob as _g
    out = []
    for exp, keep, note in _PREFERRED:
        ps = _g.glob(f"assets/results/{exp}/*/checkpoints/*.pt")
        if not ps:
            continue

        def _st(p):
            try:
                return int(os.path.basename(p).replace(".pt", ""))
            except ValueError:
                return -1

        ps.sort(key=_st, reverse=True)
        for p in ps[:keep]:
            step = _st(p)
            out.append((f"{exp} @{step}  ({note})", p))
    return out


'''
s = s[:START] + NEW + s[END:]

io.open(P, "w", encoding="utf-8").write(s)
import ast

ast.parse(s)
print("  ✓ 白名单已写入")
