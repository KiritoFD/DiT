# -*- coding: utf-8 -*-
"""汇总 T1（笔画方差 F 比）+ D3（空间分布）+ T0（表运动）三张表。"""
import glob
import io
import json
import os

RUN_ORDER = ["v13_base_50k", "v13_12ch_post", "v13_styletok", "v13_wd01",
             "v15a_multistyle_k4", "v15b_multistyle_k4", "v15b_supcon", "v15c_fixed"]


def t1_rows(d):
    out = {}
    for f in sorted(glob.glob(os.path.join(d, "t1_*_0*.json"))):
        base = os.path.basename(f)[3:-5].split("__")[0]
        try:
            j = json.load(io.open(f, encoding="utf-8"))
        except Exception:
            continue
        ng = j.get("nongeo", {})
        out[base] = {
            "genF": j["生成图"]["overall_F"], "gtF": j["GT"]["overall_F"],
            "ratio": j.get("ratio_gen_over_gt"),
            "z": j["生成图"]["z"],
            "ratio_ng": ng.get("ratio_gen_over_gt"),
        }
    return out


def main():
    d = "assets/t1_remote/assets"
    t1 = t1_rows(d)
    if not t1:
        print("没找到 t1_*_0*.json")
        return
    print("=" * 84)
    print("T1 — 笔画统计量的 组间/组内 方差比 F（不训练分类器，纯方差结构）")
    print("=" * 84)
    print("GT 天花板: F = %.4f  (置换零分布 ~0.99)" % list(t1.values())[0]["gtF"])
    print()
    print("%-24s %9s %9s %9s %9s %9s" %
          ("run", "生成F", "÷GT", "非几何÷GT", "z值", "判定"))
    print("-" * 84)
    rows = [(v["ratio"] or 0, k) for k, v in t1.items()]
    for r, k in sorted(rows, reverse=True):
        v = t1[k]
        z = v["z"]
        if z >= 3:
            verdict = "✅ 强"
        elif z >= 1.5:
            verdict = "◐ 边缘"
        elif z >= -0.5:
            verdict = "✗ 与随机无异"
        else:
            verdict = "✗ 比随机差"
        ng = v["ratio_ng"]
        print("%-24s %9.4f %9.3f %9s %9.2f %9s"
              % (k, v["genF"], r, ("%.3f" % ng) if ng else "-", z, verdict))


if __name__ == "__main__":
    main()
