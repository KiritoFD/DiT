# -*- coding: utf-8 -*-
"""CPU 冒烟: 只测 v18 (12ch) / v19 (xattn) 的可启动性 (build + fwd/bwd + cfg)。"""
import re
import sys

src = open("_review/ladder_smoke.py", encoding="utf-8").read()
src = re.sub(r'NAMES = \[.*?\]',
             'NAMES = ["v18_pretrain_S_cat_12ch_fame_kxl_tj_px60", '
             '"v19_pretrain_S_xattn_fame_kxl_tj_px60"]', src, flags=re.S)
exec(compile(src, "smoke18_19", "exec"))
