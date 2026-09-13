#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json

BASE = "/root/Workspace/xy/DiT/"
FIX = {
    "exp_s6_top6_struct_fp32.json": 4000,
    "exp_s6_top6_struct_fp32_full.json": 500,
    "exp_s6_top6_diff_then_struct.json": 4000,
}
for f, ep in FIX.items():
    p = BASE + f
    d = json.load(open(p, encoding="utf-8"))
    d["epochs"] = ep
    json.dump(d, open(p, "w"), indent=2, ensure_ascii=False)
    print(f, "epochs ->", ep, flush=True)
print("done", flush=True)