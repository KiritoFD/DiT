# -*- coding: utf-8 -*-
"""修 tools/calibrate_mid_structure.py 的 ckpt 加载: 绕开 45 vs 88 预训练表断言。"""
import io
import os

P = "/root/Workspace/xy/DiT/tools/calibrate_mid_structure.py"
raw = io.open(P, encoding="utf-8", newline="").read()
parts = raw.split("\r\n")
src = "\n".join(parts)

OLD = '''    ck = th.load(a.ckpt, map_location="cpu", weights_only=False)
    sd = ck.get("delta", ck.get("model", ck))
    sd = {(k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
          for k, v in sd.items() if isinstance(v, th.Tensor)}
    args = ck.get("args", None)
    d = vars(args) if args is not None else {}
    in_ch = sd["x_embedder.proj.weight"].shape[1]
    _sp = _ilu.spec_from_file_location("_cfs", os.path.join(ROOT, "tools", "cfg_sweep.py"))
    _cfs = _ilu.module_from_spec(_sp); _sp.loader.exec_module(_cfs)
    import argparse as _ap
    _ns = _ap.Namespace(**d)
    for _k, _v in (("image_size", 256), ("vae_downscale", 8), ("latent_channels", 4),
                   ("aux_latent_shards_dirs", ""), ("num_calligraphers", 87),
                   ("num_characters", 7765), ("callig_embed_dim", 128),
                   ("char_embed_dim", 384), ("condition_fusion", "factorized_cat")):
        if getattr(_ns, _k, None) is None:
            setattr(_ns, _k, _v)
    # \u2605 \u6539\u7528\u9879\u76ee\u81ea\u5e26\u7684 load_model_from_ckpt \u2014\u2014 \u624b\u6413 build+load_state_dict \u4f1a\u5728
    #   strict=True \u4e0b\u62a5 "Unexpected key(s): y_callig_embedder.null_embed",
    #   \u56e0\u4e3a\u8be5 key \u53ea\u5728 freeze_callig_table=True (freeze_table()) \u65f6\u624d\u5b58\u5728\u3002
    #   model_io \u5df2\u5904\u7406: torch.compile \u7684 `_orig_mod.` \u524d\u7f00 + freeze_table + strict \u62a4\u680f\u3002
    from src.eval.model_io import load_model_from_ckpt
    model, _ns = load_model_from_ckpt(a.ckpt, device=dev, use_ema=True, verbose=True)
    print(f"[model] {a.ckpt}  in_ch={in_ch}  loaded via src.eval.model_io (strict=True)")'''

NEW = '''    ck = th.load(a.ckpt, map_location="cpu", weights_only=False)
    import argparse as _ap
    args = ck.get("args", None)
    _ns = args if not isinstance(args, dict) else _ap.Namespace(**args)
    # \u63a8\u7406\u53e3\u5f84\u7528 ema (\u4e0e eval \u4e00\u81f4); \u7f3a\u5931\u5219\u56de\u9000\u8bad\u7ec3\u6743\u91cd
    _use_ema = ("ema" in ck) and (ck["ema"] is not None)
    sd = ck["ema"] if _use_ema else ck.get("delta", ck.get("model", ck))
    sd = {(k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
          for k, v in sd.items() if isinstance(v, th.Tensor)}
    in_ch = sd["x_embedder.proj.weight"].shape[1]
    # \u2605 \u4e0d\u8d70 src.eval.model_io.load_model_from_ckpt: \u5b83\u7684 apply_post_construction \u4f1a
    #   \u65ad\u8a00 callig_emb_pretrained(45 \u884c) \u4e0e num_calligraphers=87(\u8868 88 \u884c) \u914d\u5957 \u2014\u2014
    #   \u672c ckpt \u91cc\u8fd9\u4e24\u8005\u4e0d\u914d\u5957(\u9884\u8bad\u7ec3\u8868\u662f"\u8fde\u7eed\u7d22\u5f15 45 \u4eba") \u3002
    #   \u6821\u51c6\u53ea\u9700\u8981\u6a21\u578b\u80fd\u524d\u5411, \u4e14\u4e0b\u9762\u7528 strict=True **\u5168\u91cf\u8986\u76d6**
    #   \u6743\u91cd(\u542b 88 \u884c\u8868\u4e0e null_embed), \u9884\u8bad\u7ec3\u8868\u672c\u6765\u5c31\u65e0\u610f\u4e49(\u4f1a\u88ab\u8986\u76d6),
    #   \u6240\u4ee5\u8fd9\u91cc\u624b\u52a8\u590d\u523b train.py \u7684\u540e\u5904\u7406: freeze_table() \u62c6\u51fa null_embed\u3002
    from src.eval.model_io import build_model_from_args
    model = build_model_from_args(_ns, device=dev)
    model.y_callig_embedder.freeze_table()   # \u62c6\u51fa null_embed, \u5bf9\u9f50 ckpt key
    model.load_state_dict(sd, strict=True)   # strict=True \u5f53\u62a4\u680f
    model.eval()
    print(f"[model] {a.ckpt}  in_ch={in_ch}  ema={_use_ema}  strict=True OK "
          f"(params={sum(p.numel() for p in model.parameters()):,})")'''

assert OLD in src, "OLD not found"
src = src.replace(OLD, NEW)
out = "\r\n".join(src.split("\n"))
io.open(P, "w", encoding="utf-8", newline="").write(out)
print("patched OK, CRLF:", out.count("\r\n"))
