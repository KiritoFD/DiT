#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys, json
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
d = json.load(sys.stdin)
print(len(d), "entries:")
for k, v in d.items():
    print(" ", k, v.get("step"), v.get("mse", "ERR"), v.get("ssim", ""))