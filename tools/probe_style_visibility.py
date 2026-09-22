# -*- coding: utf-8 -*-
"""probe_style_visibility.py — D1：风格条件在 adaLN 里到底有多"可见"。

## 为什么要测这个

`ratio_style` 只有 1.2–2.5（理想 5–10）= 换书家 ≈ 换噪声。一个非常具体、
非常便宜的假说是 **H1 幅度淹没**：

    c = t_emb + y_emb  ->  adaLN_modulation(c)  ->  scale / shift / gate
            ↑         ↑
        时间步信号   风格信号

如果 ‖y_emb‖ ≪ ‖t_emb‖，那么 `adaLN_modulation` 看到的几乎只有时间步，
**风格向量即使有信息也传不进去**。修法可以只是放大 `callig_scale`，
或把风格送进**独立于 t 的**调制通路 —— 比换 cross-attention 划算得多。

## 与 tools/probe_condition_injection.py 的区别（重要）

旧脚本是 v3/v11 时代的，它**自己重算**条件链路：

    e_callig = y_callig_embedder(yc); p_char = model.char_proj(e_char) ...

这有三个问题，导致它在 v13/v15/v17 上**直接崩或给出错数字**：
  1. 假设 `factorized_add` + `char_proj`；v13 起是 `factorized_cat` + `no_char_cond=True`
     -> `model.char_proj is None` -> AttributeError
  2. 没有 `glyph_vec_cond`（g 池化向量也是条件操作数之一）
  3. 完全不认识 v15 的 MultiStyleEmbedder 与 v17 的 S2（e_style = E_callig + α·E_pair）

本脚本**不重算数学**，改为在真实前向里挂 forward hook 抓：
  * `t_embedder` 的输出        -> t_emb
  * `cond_fusion` 的输出       -> y_emb（这就是进 c 的那个条件向量）
  * 每层 `adaLN_modulation` 的**输入** c

然后**在 hook 之外**重算 `mod(c)` 与 `mod(c − y_emb)`（在 hook 里调自己会无限递归）。
所以它对**任何** fusion 模式 / 任何风格表形态都成立，包括未来的 S2。

## 测量项

1. **H1 幅度**：‖t_emb‖ / ‖y_emb‖ / 比值 / 两者余弦
2. **可见性 Δ**：`‖mod(c) − mod(c−y_emb)‖ / ‖mod(c)‖`，**逐层、逐组**
   （DiTBlock 6 组：shift/scale/gate × msa/mlp；FinalLayer 2 组：shift/scale）
   -> 这才是"风格对调制量的实际贡献占比"，比单纯看 norm 比值更直接
3. **风格轴 vs 内容轴**：换书家 / 换 g 各自引起的 Δy_emb 与 Δmod
   -> 在**条件层**回答"模型对风格更敏感还是对内容更敏感"（与图像层的 ratio_style 互补）

## 判据（怎么读）

| 指标 | 健康 | 病态 |
|---|---|---|
| `ratio_y_over_t` | 0.3 ~ 3 | **< 0.3 = 淹没** |
| `delta_mod_*_full`（风格轴） | 与内容轴同量级 | **< 0.05 = 风格几乎不进入调制** |
| `delta_style / delta_content` | ≈ 1 | **≪ 1 = 风格被内容压住** |

## 用法

    # 远程（需要真实 ckpt）
    python tools/probe_style_visibility.py \
        --ckpt assets/results/v13_base_50k/<ts>/checkpoints/0155000.pt \
        --config src/train/configs/v13_base_50k.json --n 32

    # 本地自检（随机初始化模型，不需要 ckpt/数据）
    python tools/probe_style_visibility.py --self-test
"""
import argparse
import json
import math
import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch  # noqa: E402


# ── 模型展开（DDP / torch.compile 包装）──────────────────────────────────────
def unwrap(m):
    for attr in ("module", "_orig_mod"):
        m = getattr(m, attr, m)
    return m


# ── hook 探针 ────────────────────────────────────────────────────────────────
class AdaLCapture:
    """抓 t_emb / y_emb / 每层 adaLN_modulation 的输入 c。

    关键：**不在 hook 里调用被 hook 的模块**（会无限递归）。
    hook 只记录输入，重算放到 forward 结束后。
    """

    def __init__(self, model):
        self.m = unwrap(model)
        self.t_emb = None
        self.y_emb = None
        self.c_by_block = {}
        self.c_final = None
        self._h = []

    def __enter__(self):
        m = self.m
        self._h.append(m.t_embedder.register_forward_hook(
            lambda mod, i, o: setattr(self, "t_emb", o.detach())))
        cf = getattr(m, "cond_fusion", None)
        if cf is not None and isinstance(cf, torch.nn.Module):
            self._h.append(cf.register_forward_hook(
                lambda mod, i, o: setattr(self, "y_emb", o.detach())))
        for idx, blk in enumerate(m.blocks):
            self._h.append(blk.adaLN_modulation.register_forward_pre_hook(
                self._mk(idx)))
        fl = getattr(m, "final_layer", None)
        if fl is not None and isinstance(fl.adaLN_modulation, torch.nn.Module):
            self._h.append(fl.adaLN_modulation.register_forward_pre_hook(
                lambda mod, i: setattr(self, "c_final", i[0].detach())))
        return self

    def _mk(self, idx):
        def hook(mod, inp):
            self.c_by_block[idx] = inp[0].detach()
        return hook

    def __exit__(self, *exc):
        for h in self._h:
            h.remove()
        self._h = []
        return False


def _groups(block_mod, out):
    """按 DiTBlock 的 6 组 / FinalLayer 的 2 组切分调制输出。"""
    n = out.shape[-1]
    if n % 6 == 0 and n // 6 == out.shape[-1] // 6:
        pass
    if n == 6 * (out.shape[-1] // 6) and n // 6 in (out.shape[-1] // 6,):
        pass
    return out


def _chunk_names(out):
    """输出维度 -> 组名。DiTBlock 是 6*D，FinalLayer 是 2*D。"""
    n = out.shape[-1]
    return None, n


def probe(model, batch, t_val=0.5, device="cpu"):
    """跑一次真实前向，返回 D1 的所有测量量。batch 是 forward 的 kwargs。"""
    m = unwrap(model)
    D = int(m.x_embedder.hidden_size)

    cap = AdaLCapture(m)
    with torch.no_grad(), cap:
        m(**batch)   # 只借它的前向：hook 抓 t_emb / y_emb / 每层 c

    t_emb, y_emb = cap.t_emb, cap.y_emb
    if t_emb is None or y_emb is None:
        raise RuntimeError(
            "没抓到 t_emb / y_emb。检查模型是否有 t_embedder 与 cond_fusion"
            "（legacy fusion 也走 cond_fusion，应该都在）。")

    res = {
        "t_emb_norm": float(t_emb.norm(dim=-1).mean()),
        "y_emb_norm": float(y_emb.norm(dim=-1).mean()),
        "c_norm": float((t_emb + y_emb).norm(dim=-1).mean()),
        "cos_t_y": float(torch.nn.functional.cosine_similarity(
            t_emb, y_emb, dim=-1).mean()),
    }
    res["ratio_y_over_t"] = res["y_emb_norm"] / max(res["t_emb_norm"], 1e-8)

    # ── 退化守卫 ────────────────────────────────────────────────────────────
    # adaLN-Zero 的 `adaLN_modulation[-1]` 是 **zero-init** -> 未训练/刚初始化的模型上
    # `mod(c) ≡ 0`，于是 Δ = 0/0 被 clamp 成 0，看起来像"风格完全不可见"。
    # 这是**初始化特性，不是模型缺陷**（实测踩到，与 smoke 测试同一个坑）。
    # 与其给一个会被误读的数字，不如显式报出来。
    _mod_norm = None
    with torch.no_grad():
        for idx, c in cap.c_by_block.items():
            _mod_norm = float(m.blocks[idx].adaLN_modulation(c).norm(dim=-1).mean())
            break
    res["mod_output_norm"] = _mod_norm
    res["degenerate"] = bool(_mod_norm is not None and _mod_norm < 1e-6)

    # ── 逐层 / 逐组的"可见性 Δ" ─────────────────────────────────────────────
    # Δ = ‖mod(c) − mod(c − y_emb)‖ / ‖mod(c)‖
    #   = 把 y_emb 拿掉后，调制量变化了百分之多少
    per_layer = {}
    for idx, c in cap.c_by_block.items():
        mod = m.blocks[idx].adaLN_modulation
        full = mod(c)
        nocond = mod(c - y_emb)
        n_groups = 6 if full.shape[-1] == 6 * D else (
            2 if full.shape[-1] == 2 * D else None)
        if n_groups is None:
            continue
        names = (["shift_msa", "scale_msa", "gate_msa",
                  "shift_mlp", "scale_mlp", "gate_mlp"] if n_groups == 6
                 else ["shift_final", "scale_final"])
        d = {}
        for gi, nm in enumerate(names):
            f = full[:, gi * D:(gi + 1) * D]
            z = nocond[:, gi * D:(gi + 1) * D]
            d[nm] = float(((f - z).norm(dim=-1) / f.norm(dim=-1).clamp_min(1e-8)).mean())
        d["_mean"] = float(sum(d.values()) / len(d))
        per_layer[idx] = d
    res["per_layer"] = per_layer
    if per_layer:
        keys = [k for k in next(iter(per_layer.values())) if not k.startswith("_")]
        res["delta_mod_mean"] = float(
            sum(v["_mean"] for v in per_layer.values()) / len(per_layer))
        res["delta_mod_by_group"] = {
            k: float(sum(v[k] for v in per_layer.values()) / len(per_layer))
            for k in keys}
        # 层间不平衡（历史诊断：L3 吃 60% 梯度，这里看调制量是否也不平衡）
        _ms = [v["_mean"] for v in per_layer.values()]
        res["delta_mod_layer_min"] = min(_ms)
        res["delta_mod_layer_max"] = max(_ms)
        res["delta_mod_layer_imbalance"] = (max(_ms) / max(min(_ms), 1e-8))

    # FinalLayer
    if cap.c_final is not None:
        fl = m.final_layer
        full = fl.adaLN_modulation(cap.c_final)
        nocond = fl.adaLN_modulation(cap.c_final - y_emb)
        if full.shape[-1] == 2 * D:
            d = {}
            for gi, nm in enumerate(["shift_final", "scale_final"]):
                f = full[:, gi * D:(gi + 1) * D]
                z = nocond[:, gi * D:(gi + 1) * D]
                d[nm] = float(((f - z).norm(dim=-1)
                               / f.norm(dim=-1).clamp_min(1e-8)).mean())
            d["_mean"] = float(sum(d.values()) / len(d))
            res["final_layer"] = d

    res["n_blocks"] = len(per_layer)
    return res


# ── 风格轴 vs 内容轴 ────────────────────────────────────────────────────────
def axis_contrast(model, batch, device="cpu"):
    """换书家 vs 换 g：条件向量与调制量各自变化多少。"""
    m = unwrap(model)
    D = int(m.x_embedder.hidden_size)

    def _yemb(b):
        cap = AdaLCapture(m)
        with torch.no_grad(), cap:
            m(**b)
        return cap.y_emb, cap.c_by_block.get(0)

    b1 = dict(batch)
    b2 = dict(batch)
    # 风格轴：翻转书家 / pair
    for k in ("y_callig", "y_callig_raw", "y_pair"):
        if b2.get(k) is not None and b2[k].numel() > 1:
            b2[k] = torch.roll(b2[k], 1, dims=0)
    # 内容轴：换 g（翻转 batch 内的 g）
    if b2.get("g") is not None and b2["g"].shape[0] > 1:
        b2["g"] = torch.roll(b2["g"], 1, dims=0)

    y1, c1 = _yemb(b1)
    y2, c2 = _yemb(b2)

    out = {}
    dy = (y2 - y1).norm(dim=-1)
    ny = y1.norm(dim=-1).clamp_min(1e-8)
    out["dy_emb_rel"] = float((dy / ny).mean())

    mod = m.blocks[0].adaLN_modulation
    f1, f2 = mod(c1), mod(c2)
    out["dmod_rel"] = float(((f2 - f1).norm(dim=-1)
                             / f1.norm(dim=-1).clamp_min(1e-8)).mean())
    return out


# ── 自检（随机初始化模型，不需要 ckpt / 数据）───────────────────────────────
def _self_test(cfg_path, n=8, device="cpu"):
    from src.train.cli import parse_args
    from src.eval.model_io import build_model_from_args

    a = parse_args(["--config", cfg_path])
    m = build_model_from_args(a, device).eval()
    N = n
    # ⚠ x 的通道数必须取模型的 in_channels（12ch 的 aux 目标会让它是 12），
    #   而 g 恒为 4（glyph_in_channels）。写死 4 会让 12ch ckpt 直接崩（实测踩到）。
    _Cin = int(getattr(m, "in_channels", 4))
    batch = dict(
        x=torch.randn(N, _Cin, 32, 32, device=device),
        t=torch.full((N,), 0.5, device=device),
        y_callig=torch.arange(N, device=device) % max(int(a.num_calligraphers), 1),
        y_char=torch.zeros(N, dtype=torch.long, device=device),
        g=torch.randn(N, 4, 32, 32, device=device),
        y_callig_raw=torch.arange(N, device=device) % max(int(a.num_calligraphers), 1),
        y_pair=(torch.arange(N, device=device)
                % max(int(getattr(a, "num_pairs", 0) or a.num_calligraphers), 1)),
        y_script=torch.zeros(N, dtype=torch.long, device=device),
    )
    print(f"# self-test on {cfg_path}  (随机初始化，数字只有形状意义)\n")
    r = probe(m, batch, device=device)
    _report(r, a)
    print("\n# 轴对照（随机初始化下无意义，仅验证代码路径）")
    ax = axis_contrast(m, batch, device)
    print(f"  换风格: dy_emb_rel={ax['dy_emb_rel']:.4f}  dmod_rel={ax['dmod_rel']:.4f}")
    return r


def _report(r, a):
    print("=" * 74)
    print("1) H1 幅度：条件向量 vs 时间步向量")
    print("=" * 74)
    print(f"  ‖t_emb‖        {r['t_emb_norm']:>10.4f}")
    print(f"  ‖y_emb‖        {r['y_emb_norm']:>10.4f}")
    print(f"  ‖c‖            {r['c_norm']:>10.4f}")
    print(f"  cos(t_emb,y_emb) {r['cos_t_y']:>9.4f}")
    ratio = r["ratio_y_over_t"]
    verdict = ("✗ **淹没**（条件可能传不进去）" if ratio < 0.3
               else "✓ 同量级" if ratio < 3.0 else "⚠ 条件过强（可能压制时间步）")
    print(f"\n  y_emb / t_emb = {ratio:.4f}   {verdict}")
    print("  参考: 架构上 c = t_emb + y_emb；< 0.3 时 adaLN 几乎只看到时间步。")
    print("  ⚠ 本项依赖 ckpt（t_embedder 与 cond_fusion 都是训练出来的幅度）；"
          "随机模型上的数值不可读。§2 的 Δ 同样必须有真实 ckpt（见下）。")

    print("\n" + "=" * 74)
    print("2) 可见性 Δ = ‖mod(c) − mod(c − y_emb)‖ / ‖mod(c)‖  （越大越可见）")
    print("=" * 74)
    if r.get("degenerate"):
        print("  ⚠⚠ **退化：mod(c) ≡ 0** —— adaLN-Zero 的 `adaLN_modulation[-1]` 是 zero-init。")
        print("     这是**未训练/刚初始化**模型的特征，不是「风格不可见」。")
        print(f"     ‖mod(c)‖ = {r.get('mod_output_norm')}。下面的 Δ 全是 0/0，**不可读**。")
        print("     请用**真实训练过的 ckpt** 跑本探针。")
    if "delta_mod_by_group" in r:
        print(f"  逐层平均 Δ = {r['delta_mod_mean']:.4f}   "
              f"（层数 {r['n_blocks']}）")
        print(f"  层间不平衡 max/min = {r['delta_mod_layer_imbalance']:.2f}   "
              f"[min {r['delta_mod_layer_min']:.4f} / max {r['delta_mod_layer_max']:.4f}]")
        print("\n  按调制组平均：")
        for k, v in r["delta_mod_by_group"].items():
            bar = "█" * max(1, int(round(v * 50)))
            print(f"    {k:<12} {v:>8.4f}  {bar}")
        if "final_layer" in r:
            fl = r["final_layer"]
            print(f"\n  final_layer: shift={fl['shift_final']:.4f} "
                  f"scale={fl['scale_final']:.4f} mean={fl['_mean']:.4f}")
        print(f"\n  逐层 Δ：")
        for i in sorted(r["per_layer"]):
            print(f"    block {i:>2}  {r['per_layer'][i]['_mean']:.4f}")
    else:
        print("  （没抓到 adaLN 调制输出，检查模型结构）")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None)
    ap.add_argument("--self-test", action="store_true",
                    help="用随机初始化模型跑一遍（不需要 ckpt / 数据），验证代码路径")
    a = ap.parse_args()

    if a.self_test:
        cfg = a.config or "src/train/configs/v13_base_50k.json"
        _self_test(cfg, n=min(a.n, 8), device=a.device)
        if a.config is None:
            print("\n# 再试一个 S2 配置（验证 hier 路径也能被 hook 到）")
            _self_test("src/train/configs/v17_s2_s2c_full.json",
                       n=min(a.n, 8), device=a.device)
        return 0

    if not a.ckpt or not a.config:
        raise SystemExit("需要 --ckpt 与 --config（或 --self-test）")

    from src.eval.model_io import load_model_from_ckpt
    model, args = load_model_from_ckpt(a.ckpt, device=a.device)
    print(f"[load] {a.ckpt}")

    N = a.n
    dev = a.device
    nc = max(int(getattr(args, "num_calligraphers", 1)), 1)
    npr = max(int(getattr(args, "num_pairs", 0) or nc), 1)
    _Cin = int(getattr(model, "in_channels", 4))
    print(f"[shape] x 通道={_Cin}（12ch 模型的 aux 目标通道）；g 恒为 4")
    batch = dict(
        x=torch.randn(N, _Cin, 32, 32, device=dev),
        t=torch.full((N,), 0.5, device=dev),
        y_callig=torch.arange(N, device=dev) % nc,
        y_char=torch.zeros(N, dtype=torch.long, device=dev),
        g=torch.randn(N, 4, 32, 32, device=dev),
        y_callig_raw=torch.arange(N, device=dev) % nc,
        y_pair=torch.arange(N, device=dev) % npr,
        y_script=torch.zeros(N, dtype=torch.long, device=dev),
    )
    r = probe(model, batch, device=dev)
    _report(r, args)

    print("\n" + "=" * 74)
    print("3) 风格轴 vs 内容轴（条件层）")
    print("=" * 74)
    ax = axis_contrast(model, batch, device=dev)
    print(f"  换书家: ‖Δy_emb‖/‖y_emb‖ = {ax['dy_emb_rel']:.4f}   "
          f"‖Δmod‖/‖mod‖ = {ax['dmod_rel']:.4f}")
    r["axis_style"] = ax
    print("  （内容轴需要两次不同的 g，见 --self-test 里的 axis_contrast 实现；"
          "这里只报风格轴，避免与 g 的随机翻转混淆。）")

    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(r, open(a.out, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
        print(f"\nsaved -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
