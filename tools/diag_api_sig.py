# -*- coding: utf-8 -*-
"""diag_api_sig.py — 打印远程(src.eval)关键 API 的真实签名, 供写实验脚本对齐。"""
import inspect
import sys

sys.path.insert(0, "/root/Workspace/xy/DiT")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.eval import inference, model_io  # noqa: E402

for mod, names in ((inference, ["make_eval_cache", "sample_latents",
                                "load_eval_vae", "build_diffusion"]),
                   (model_io, ["load_model_from_ckpt"])):
    for n in names:
        f = getattr(mod, n, None)
        if f is None:
            print(f"{mod.__name__}.{n}: 不存在")
            continue
        try:
            print(f"{mod.__name__}.{n}{inspect.signature(f)}")
        except Exception as e:  # noqa: BLE001
            print(f"{mod.__name__}.{n}: <{type(f).__name__}> sig err {e}")
    print()

# 指标函数名探测 (ink_iou / skel_iou / frag_ratio 在哪)
import re  # noqa: E402
for mod in (inference,):
    src = open(mod.__file__, encoding="utf-8", errors="replace").read()
    for kw in ("ink_iou", "skel_iou", "frag_ratio", "hole_pred"):
        hits = [l.strip()[:120] for l in src.splitlines()
                if kw in l and re.match(r"\s*(def |    [a-z_]+ *=)", l)]
        print(f"[{kw}] 定义/赋值处 {len(hits)}:")
        for h in hits[:4]:
            print("   ", h)
