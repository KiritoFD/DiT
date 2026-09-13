# -*- coding: utf-8 -*-
"""远程：检查 s19-flow-midcommon 0025000.pt 的 key 结构。"""
import torch
p = "assets/results/ctrl_skel/20260828-224832-ctrl-skel-s19-flow-midcommon/checkpoints/0025000.pt"
ck = torch.load(p, map_location="cpu", weights_only=False)
print("type:", type(ck).__name__)
if isinstance(ck, dict):
    print("top keys:", list(ck.keys()))
    for k, v in ck.items():
        if isinstance(v, dict):
            ks = list(v.keys())
            print(f"  [{k}] dict, {len(ks)} keys")
            # 分类前缀
            from collections import Counter
            pref = Counter(ks[:0])
            pfx = Counter()
            for kk in ks:
                head = kk.split(".")[0]
                pfx[head] += 1
            print("    prefix histogram:", dict(pfx))
            # 打印样本
            ctrl = [k for k in ks if k.startswith("ctrl_encoder")]
            inj = [k for k in ks if k.startswith("injections")]
            main = [k for k in ks if k.startswith("main")]
            print("    #ctrl_encoder:", len(ctrl), " sample:", ctrl[:4])
            print("    #injections:", len(inj), " sample:", inj[:4])
            print("    #main:", len(main), " sample:", main[:4])
            if ctrl:
                print("    ctrl_encoder head types:", sorted({k.split('.')[1] for k in ctrl}))[:20]
        else:
            print(f"  [{k}] -> {type(v).__name__} = {str(v)[:60]}")
