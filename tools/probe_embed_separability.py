# -*- coding: utf-8 -*-
"""
probe_embed_separability.py — D5：**条件嵌入本身**的书家可分性（只读权重，不跑前向）。

要回答的问题
------------
D1 测的是「风格对 adaLN 的可见性」(dmod)，T2 测的是「**生成图**的书家可分性」。
两者中间缺一环：**书家嵌入表 y_emb 本身是否可分？**

三种失效模式对应三种不同的修法：
  A. 嵌入表**本身不可分**   -> 无论怎么注入都白搭，必须改监督/加约束（supcon、pair 残差）
  B. 嵌入表可分、adaLN 读不动 -> 注入通路有问题（这正是 S2 local-CA 要解决的）
  C. 嵌入表可分、adaLN 也读得动，但生成图不可分 -> 生成端（VAE/采样/条件丢失）

本脚本测 A：直接对 ckpt 里落盘的嵌入权重算书家可分性。
**不跑前向、不采样**，纯矩阵运算 -> 秒级、零 GPU 占用。

测什么
------
对每个书家 c，取它在 ckpt 里的嵌入向量。三种嵌入来源分别测：
  1. `y_callig_embedder.weight[c]`        原始查表（最底层）
  2. 主效应表（S2 的 E_callig / hier 模式的 base）  —— 若有
  3. `callig_proj` 之后（若存在投影）      —— 进入 c 之前的向量
  4. `y_pair` 残差（若有 `E_pair`）        —— pair 的边际贡献

指标：
  * 书家间平均余弦距离 / 书家内（同表无组内，故用随机对基线）
  * **有效秩**（参与 90% 能量的奇异值个数）—— 判断表是否退化成低维
  * **最近邻纯度**：每个书家的最近邻是否不同书家（应该全是，除非塌缩）
  * 与 T2 的 DINO 线性可分性对齐（若给了 T2 结果 json）

判读
----
  * `eff_rank` 远小于书家数 -> 表塌缩到低维子空间，书家无法区分（A 类失效）
  * `nn_purity` 高但 T2 低 -> 表可分、注入或生成端有问题（B/C 类）
  * 向量范数分布极不均 -> 少数书家主导，其余被淹没

用法
----
  python tools/probe_embed_separability.py \
      --ckpt <best.pt> --config <resolved_config.json> \
      --out assets/d5_<tag>.json
"""
import os, sys, json, argparse
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import torch


def _strip(sd):
    """剥掉 DDP / torch.compile 包装前缀。"""
    out = {}
    for k, v in sd.items():
        kk = k
        changed = True
        while changed:
            changed = False
            for pfx in ("module.", "_orig_mod.", "model."):
                if kk.startswith(pfx):
                    kk = kk[len(pfx):]
                    changed = True
        out[kk] = v
    return out


def eff_rank(X, thresh=0.90):
    """参与 thresh 能量的奇异值个数（有效秩）。"""
    if X.shape[0] < 2:
        return 1
    Xc = X - X.mean(axis=0, keepdims=True)
    try:
        s = np.linalg.svd(Xc, compute_uv=False)
    except np.linalg.LinAlgError:
        return -1
    e = s ** 2
    tot = e.sum()
    if tot <= 0:
        return 0
    cum = np.cumsum(e) / tot
    return int(np.searchsorted(cum, thresh) + 1)


def nn_purity(X, labels):
    """每个样本的最近邻（排除自身）是否同书家。返回同书家比例。

    书家数少时「最近邻同书家」本来概率就不低，所以同时返回随机基线。
    """
    n = X.shape[0]
    if n < 3:
        return float("nan"), float("nan")
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    S = Xn @ Xn.T
    np.fill_diagonal(S, -np.inf)
    nn = S.argmax(axis=1)
    same = (labels[nn] == labels).mean()
    # 随机基线：按标签频率算「随便抽一个同书家」的概率
    _, cnt = np.unique(labels, return_counts=True)
    p = (cnt / cnt.sum()) ** 2
    base = float(p.sum())
    return float(same), base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default=None, help="resolved_config.json（用来看 num_calligraphers）")
    ap.add_argument("--out", default=None)
    ap.add_argument("--t2-json", default=None, help="可选：T2 结果 json，用于对齐展示")
    ap.add_argument("--which", default="ema", choices=["ema", "delta", "both"],
                    help="读哪份权重: ema(推理实际用) / delta(训练增量) / both")
    args = ap.parse_args()

    dev = "cpu"  # 只读权重，不需要 GPU
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    if not isinstance(ck, dict):
        raise SystemExit("ckpt 顶层不是 dict")

    # ★ 这套 ckpt 存的是 {delta, opt, args, train_steps, ema, scheduler}
    #   delta = 相对预训练的增量（冻结的权重不会出现在这里！）
    #   ema   = 推理时实际使用的指数滑动平均权重（较完整）
    res = {
        "ckpt": os.path.basename(args.ckpt),
        "tag": os.path.basename(os.path.dirname(os.path.dirname(args.ckpt))),
        "ckpt_top_keys": sorted([k for k in ck.keys()]),
        "train_steps": ck.get("train_steps"),
    }

    def _cands_from(sd):
        cands = {}
        for k, v in sd.items():
            if not hasattr(v, "ndim") or v.ndim != 2:
                continue
            low = k.lower()
            if "callig" not in low and "style" not in low and "pair" not in low:
                continue
            cands[k] = v
        return cands

    sources = {}
    if isinstance(ck.get("ema"), dict):
        sources["ema"] = _strip(ck["ema"])
    if isinstance(ck.get("delta"), dict):
        sources["delta"] = _strip(ck["delta"])
    if not sources:
        for key in ("model", "model_state_dict", "state_dict"):
            if isinstance(ck.get(key), dict):
                sources[key] = _strip(ck[key])
                break
    if not sources:
        raise SystemExit("ckpt 里找不到可用权重（ema/delta/model）")

    print("=" * 74)
    print("D5  条件嵌入可分性   ckpt=%s" % res["ckpt"])
    print("=" * 74)
    print("ckpt 顶层键: %s" % ", ".join(res["ckpt_top_keys"]))
    print("train_steps: %s" % res["train_steps"])
    for sn, sd in sources.items():
        print("  [%s] %d 个张量" % (sn, len(sd)))
    print()

    sd = sources.get(args.which) or list(sources.values())[0]

    # ── 候选嵌入表：按 key 名匹配 ─────────────────────────────────────────────
    cands = _cands_from(sd)
    print("候选权重（含 callig/style/pair 的 2D 张量）:")
    for k, v in sorted(cands.items()):
        print("   %-58s %s" % (k, tuple(v.shape)))
    print()
    n_callig = None
    if args.config and os.path.isfile(args.config):
        cfgj = json.load(open(args.config, encoding="utf-8"))
        # resolved_config.json 可能是 {"args": {...}} 或直接是 args
        c = cfgj.get("args", cfgj)
        n_callig = c.get("num_calligraphers")
        res["num_calligraphers_cfg"] = n_callig
        res["callig_embed_dim"] = c.get("callig_embed_dim")

    report = {}
    for k, W in sorted(cands.items()):
        Wn = W.float().numpy()
        # 只对"表"类权重算：行数应该在合理范围（<=512），且不是投影矩阵
        low = k.lower()
        is_proj = any(t in low for t in ("proj", "mlp", "net", "basis", "film", "ca."))
        n_rows, dim = Wn.shape
        entry = {
            "shape": [int(n_rows), int(dim)],
            "is_proj_guess": bool(is_proj),
            "row_norm_mean": float(np.linalg.norm(Wn, axis=1).mean()),
            "row_norm_std": float(np.linalg.norm(Wn, axis=1).std()),
            "row_norm_min": float(np.linalg.norm(Wn, axis=1).min()),
            "row_norm_max": float(np.linalg.norm(Wn, axis=1).max()),
            "eff_rank_90": eff_rank(Wn),
            "dead_rows": int((np.linalg.norm(Wn, axis=1) < 1e-8).sum()),
        }
        # 书家间余弦距离（排除 zero / null 行）
        norms = np.linalg.norm(Wn, axis=1)
        alive = norms > 1e-8
        if alive.sum() >= 3:
            Wa = Wn[alive]
            Wl = Wa / (np.linalg.norm(Wa, axis=1, keepdims=True) + 1e-12)
            S = Wl @ Wl.T
            iu = np.triu_indices(Wa.shape[0], k=1)
            entry["cos_mean"] = float(S[iu].mean())
            entry["cos_std"] = float(S[iu].std())
            entry["cos_p05"] = float(np.percentile(S[iu], 5))
            entry["cos_p95"] = float(np.percentile(S[iu], 95))
            entry["cos_min"] = float(S[iu].min())
            entry["n_alive"] = int(alive.sum())
            # 重复行检测（完全相同的行 = 表塌缩的最极端形式）
            uniq = np.unique(np.round(Wa, 6), axis=0)
            entry["n_unique_rows"] = int(uniq.shape[0])
        report[k] = entry

    # ── 打印主要表 ────────────────────────────────────────────────────────────
    print("%-52s %10s %8s %8s %8s %8s" % ("weight", "shape", "effR90", "cosμ", "cosmin", "dead"))
    print("-" * 100)
    for k, e in report.items():
        shape = "%dx%d" % (e["shape"][0], e["shape"][1])
        print("%-52s %10s %8d %8.4f %8s %8d" % (
            k[:52], shape, e["eff_rank_90"],
            e.get("cos_mean", float("nan")),
            ("%.4f" % e["cos_min"]) if "cos_min" in e else "-",
            e["dead_rows"]))

    # ── 对齐 T2 ──────────────────────────────────────────────────────────────
    if args.t2_json and os.path.isfile(args.t2_json):
        try:
            t2 = json.load(open(args.t2_json, encoding="utf-8"))
            res["t2_aligned"] = t2
        except Exception as ex:
            res["t2_aligned_error"] = str(ex)

    res["weights"] = report

    # ── 自动判读 ─────────────────────────────────────────────────────────────
    verdicts = []
    for k, e in report.items():
        if "cos_mean" not in e:
            continue
        if e["eff_rank_90"] <= 2 and e["shape"][0] > 8:
            verdicts.append("%s: effR90=%d 极低 -> 表塌缩到 <3 维" % (k, e["eff_rank_90"]))
        if e.get("n_unique_rows", 999) < e["shape"][0] * 0.5 and e["shape"][0] > 8:
            verdicts.append("%s: 唯一样本 %d/%d -> 大量重复行" % (
                k, e["n_unique_rows"], e["shape"][0]))
        if e.get("cos_min", 0) > 0.9:
            verdicts.append("%s: 最小余弦 %.4f > 0.9 -> 有书家对几乎同向量" % (k, e["cos_min"]))
        if e["row_norm_max"] > 10 * max(e["row_norm_min"], 1e-9):
            verdicts.append("%s: 范数极不均 (%.4f..%.4f) -> 少数行主导" % (
                k, e["row_norm_min"], e["row_norm_max"]))
    res["verdicts"] = verdicts
    print()
    if verdicts:
        print("★ 自动判读:")
        for v in verdicts:
            print("   - " + v)
    else:
        print("★ 未发现表塌缩/重复/范数失衡的明显迹象")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump(res, open(args.out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
        print("\n[JSON] %s" % args.out)


if __name__ == "__main__":
    main()
