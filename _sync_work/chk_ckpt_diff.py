"""联合 ckpt 的权重 vs 初始 v33: 差多少? 判断"没保存上/没更新"还是"训练真推跑了"。"""
import torch as th

V33 = "assets/results/v33_stage1_xs/20261001-113156-v33-stage1-xs/checkpoints/0030000.pt"
J5 = "assets/results/v34_e2e/joint_005000.pt"


def pick(c, names=("gen_ema", "gen", "ema", "model", "delta")):
    for k in names:
        if k in c and isinstance(c[k], dict):
            return k, c[k]
    return "<top>", c


c33 = th.load(V33, map_location="cpu", weights_only=False)
k33, s33 = pick(c33)
print(f"v33 keys={list(c33)[:6]} -> 取 {k33} ({len(s33)} 项)")

for tag, p in (("joint_005000", J5),):
    cj = th.load(p, map_location="cpu", weights_only=False)
    print(f"\n{tag} keys={list(cj)}")
    for kk in ("gen", "gen_ema"):
        if kk not in cj:
            continue
        s = cj[kk]
        common = [k for k in s if k in s33 and hasattr(s[k], "shape")
                  and s[k].shape == s33[k].shape]
        num = sum((s33[k].float() - s[k].float()).pow(2).sum().item() for k in common)
        den = sum(s33[k].float().pow(2).sum().item() for k in common)
        print(f"  [{kk}] 与 v33 共同参数 {len(common)}/{len(s)}  "
              f"相对 L2 距离 = {(num ** 0.5) / max(den ** 0.5, 1e-9):.4f}")
        worst = sorted(
            ((((s33[k].float() - s[k].float()).norm() /
                max(s33[k].float().norm(), th.tensor(1e-9))).item(), k) for k in common),
            reverse=True)[:3]
        print("    变化最大的参数:", [(f"{v:.3f}", k[:48]) for v, k in worst])
    if "gen" in cj and "gen_ema" in cj:
        g, ge = cj["gen"], cj["gen_ema"]
        ks = [k for k in g if k in ge and hasattr(g[k], "shape")
              and g[k].shape == ge[k].shape]
        num = sum((g[k].float() - ge[k].float()).pow(2).sum().item() for k in ks)
        den = sum(g[k].float().pow(2).sum().item() for k in ks)
        print(f"  gen vs gen_ema 相对距离 = {(num ** 0.5) / max(den ** 0.5, 1e-9):.6f}")
