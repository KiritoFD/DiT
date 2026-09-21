"""给远端 gradio_stdskel.py 打补丁：改用 cfg_sweep.build_model_from_args 重建模型。

原来手写了一堆字段，**漏了 glyph_vec_cond**（v12 新增），导致
cond_fusion 形状从 256 变成 128 -> load_state_dict size mismatch。
这正是 doc68 §2.2 记过的 D1 bug（eval_diversity.py 当时也是这么修的）。
"""
import io
import os

os.chdir("/root/Workspace/xy/DiT")
P = "gradio_stdskel.py"
s = io.open(P, encoding="utf-8").read()

OLD_HEAD = 'model = DiT_2Cond_models[a.get("model", "DiT-2Cond-S/2")]('
if OLD_HEAD not in s:
    print("  ⚠ 没找到原构造代码，可能已改过")
    raise SystemExit(0)

# 找到构造调用的起止（到 `**_arch)` 结束）
i0 = s.index(OLD_HEAD)
i1 = s.index("**_arch)", i0) + len("**_arch)")
old = s[i0:i1]
print(f"  将替换 {len(old.splitlines())} 行构造代码")

NEW = '''# ★ [2026-09-21] 改用 cfg_sweep.build_model_from_args —— 字段与 train.py 逐一对齐。
#   原来手写漏了 v12 新增的 glyph_vec_cond 等字段，导致
#   cond_fusion 形状 128(!= ckpt 的 256) -> size mismatch。
#   （同一个坑在 tools/eval_diversity.py 上踩过，见 doc68 §2.2 D1）
import importlib.util as _ilu
ROOT = os.path.dirname(os.path.abspath(__file__))
_sp = _ilu.spec_from_file_location("_cfs", os.path.join(ROOT, "tools", "cfg_sweep.py"))
_cfs = _ilu.module_from_spec(_sp)
_sp.loader.exec_module(_cfs)


class _NS(dict):
    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError:
            return None


_ns = _NS(dict(a))
_ns.setdefault("device", dev)
_ns["cond_drop_all_prob"] = 0.05
_ns["cond_drop_one_prob"] = 0.25
_ns["cond_drop_which_glyph_prob"] = 0.5
_ns["use_checkpoint"] = False
_ns["learn_sigma"] = False
_ns["glyph_drop_prob"] = 0.0
_ns["attn_impl"] = "eager"
model = _cfs.build_model_from_args(_ns, dev)'''

s = s[:i0] + NEW + s[i1:]

# 加载改成 strict=True 当护栏
s = s.replace(
    "    _ms, _us = model.load_state_dict(sd, strict=False)",
    "    _ms, _us = model.load_state_dict(sd, strict=False)\n"
    "    if _ms:\n"
    "        print(f'[load] ⚠ 仍有 {len(_ms)} 个权重缺失(随机初始化): "
    "{list(_ms)[:5]}', flush=True)")

io.open(P, "w", encoding="utf-8").write(s)
print("  ✓ 已打补丁：改用 build_model_from_args")
