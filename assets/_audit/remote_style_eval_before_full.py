#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""style_eval.py —— 风格轴离线评测（**只读已生成 PNG，不动推理端** ✓）。

一条命令产出：跨字符书家检索(mAP/Top-1) · Style Prototype Margin · HWD-CN · Style-MMD ·
XCTAcc · Brush-Statistics Distance，并给出三类基线 Real–Real / Real–Other-Writer / TTF-only ✓。

内容/风格正交（审查要求 ✓）
    `Style@All`            —— 全部生成样本
    `Style@ContentCorrect` —— 仅内容判定正确的样本, **并报告覆盖率** ✓
    内容判定: 用同一个判别器的 `char_head` 预测生成图, 与**同行标准字形渲染图(std_path)**的预测
    是否一致 ✓ —— 自洽、无需把数据集 character_id 映射到判别器的 7766 类标签空间 ✓

用法
    python tools/style_eval.py --gen eval200fix=<gen_dir> \
        --gen-map exp-std/csv/eval200_fixed.csv --out <out_dir> [--baselines 1] [--bsd 1]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.eval.style import bank as B                                     # noqa: E402
from src.eval.style import store                                         # noqa: E402
from src.eval.style.brush import (BrushFeat, aggregate, bsd,             # noqa: E402
                                  features_from_gray)
from src.eval.style.encoder import StyleEncoder                          # noqa: E402
from src.eval.style.metrics import (EmbSet, build_prototypes, cross_char_retrieval,  # noqa: E402
                                    hwd_cn, prototype_margin, style_mmd_by_slot, xctacc)
from PIL import Image                                                    # noqa: E402


def _ids(rows, key):
    """把字符串标签映射成连续 int id ✓（同一份映射贯穿全程, 保证可比 ✓）。"""
    u = sorted({getattr(r, key) for r in rows})
    return {v: i for i, v in enumerate(u)}


def _mid(m: dict, key) -> int:
    """**自动扩展**的标签映射 ✓。

    ★ 必需: 评测集的字符常常**不在参考库里** ✗（实测 `KeyError: '恆'` ✓）——
      这正是"unseen-char"情形 ✓。给未见标签分配新 id ⇒ 该样本在检索里**没有同字候选**
      （`exclude_same_char` 自然失效 ✓）但仍是**合法 query** ✓，正是我们要评的 ✓。
    """
    k = str(key)
    if k not in m:
        m[k] = len(m)
    return m[k]


def to_embset(rows, emb, wmap, smap, cmap, kmap, name="set"):
    return EmbSet(emb, np.array([wmap[r.writer] for r in rows]),
                  np.array([smap[r.script] for r in rows]),
                  np.array([cmap[r.char] for r in rows]),
                  np.array([kmap[r.work] for r in rows]),
                  [os.path.basename(r.path) for r in rows])


def ocr_pred(enc, paths, cache, batch=32):
    """用判别器的 char_head 取预测类别 ✓（仅用于"内容是否正确"的自洽判定 ✓）。"""
    import torch
    out = []
    for i in range(0, len(paths), batch):
        arrs = [np.asarray(Image.open(p).convert("RGB")) for p in paths[i:i + batch]]
        with torch.no_grad():
            lg, _, _ = enc.net(enc.to_tensor(arrs).to(enc.device))
        out += lg.argmax(-1).cpu().tolist()
    return np.array(out)


def brush_feats(enc, paths, cache_key, batch=16):
    """按批提取笔法特征并聚合 ✓（走图像本身, 与神经网络无关 ✓）。"""
    feats = []
    for p in paths:
        try:
            feats.append(features_from_gray(np.asarray(Image.open(p).convert("RGB"))))
        except Exception:                                                # noqa: BLE001
            continue
    agg = aggregate(feats)
    if os.path.isdir(cache_key):
        np.savez_compressed(os.path.join(cache_key, "brush_%d.npz" % abs(hash(tuple(paths[:5])))),
                            width=agg.width, gray_ink=agg.gray_ink, edge=agg.edge, curv=agg.curv,
                            dir_width=agg.dir_width,
                            scalars=np.array([json.dumps(agg.scalars)], dtype=object))
    return agg


def _f(v, nd=4):
    """安全格式化（NaN/None ⇒ "--" ✓）。"""
    try:
        x = float(v)
        return "--" if (x != x) else ("%.*f" % (nd, x))
    except Exception:                                                    # noqa: BLE001
        return "--"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default="exp-std/csv/train.csv", help="真实真迹 CSV（建参考库 ✓）")
    ap.add_argument("--root", action="append", default=[], help="图像根（可多次）")
    ap.add_argument("--gen", action="append", default=[], help="name=dir（生成样本目录 ✓）")
    ap.add_argument("--gen-map", default="", help="生成样本按**索引**对应的 CSV ✓")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", default="pretrained_models/eval_classifier.pt")
    ap.add_argument("--kind", default="callig", choices=["callig", "cls"])
    ap.add_argument("--device", default="", help="cpu/cuda（空=自动 ✓）。训练占满 GPU 时用 cpu ✓")
    ap.add_argument("--mask-ink", type=int, default=0, help="1=背景对照（把墨迹抹掉 ✓）")
    ap.add_argument("--max-ref", type=int, default=0, help="限制参考库规模（0=全部）")
    ap.add_argument("--query-frac", type=float, default=0.3)
    ap.add_argument("--baselines", type=int, default=1)
    ap.add_argument("--bsd", type=int, default=1)
    ap.add_argument("--save-bank-pack", default="",
                    help="把参考库嵌入导出成**可移植 pack** ✓（远端编码一次, 拉回本地复用 ✓）")
    ap.add_argument("--bank-pack", default="",
                    help="直接载入参考库 pack（**跳过参考库编码** ✓ ⇒ 本地评测只需编码测试输出 ✓）")
    ap.add_argument("--bsd-ref-max", type=int, default=3000,
                    help="BSD 参考侧最多用多少张（骨架化较慢；按槽位均衡抽样 ✓，0=全部）")
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    roots = args.root or [".", "data", "exp-std"]

    print("== ① 建参考库 ==", flush=True)
    if args.bank_pack:
        # ★ 用户建议的路径 ✓: 远端把参考库编码一次 -> 存 pack -> 拉回本地 ->
        #   本地 GPU 只编码**测试输出** ⇒ 秒级完成, 且与远端路径无关 ✓
        emb_all, rdict = store.load_pack(args.bank_pack)
        rows = [B.RefRow(path=d.get("path", ""), src_path="", writer=d.get("writer", ""),
                         script=d.get("script", ""), char=d.get("char", ""),
                         work=d.get("work", ""), split=(d.get("split") or ""))
                for d in rdict]
        ref_i = [i for i, r in enumerate(rows) if r.split == "ref"]
        qry_i = [i for i, r in enumerate(rows) if r.split == "query"]
        if not ref_i:                           # 兼容没写 split 的旧 pack ✓
            ref_i, qry_i = B.split_index_by_work(rows, query_frac=args.query_frac)
        print("  ✓ 载入参考库 pack: %s（%d 张, dim=%d ✓ 跳过参考库编码 ✓）"
              % (os.path.basename(args.bank_pack), len(rows), emb_all.shape[1]))
    else:
        rows = B.build_bank(args.csv, roots=roots)
        if not rows:
            print("参考库为空 ✗（图像路径不可解析?）"); return 2
        if args.max_ref and len(rows) > args.max_ref:
            rows = B.cap_balanced(rows, args.max_ref)
            print("  参考库**按槽位均衡**压到 %d 张 ✓（避免截断丢掉整族书家 ✗）" % len(rows))
        ref_i, qry_i = B.split_index_by_work(rows, query_frac=args.query_frac)
        rows = B.tag_split(rows, ref_i, qry_i)  # ★ 写回划分 ⇒ pack 可自描述 ✓
        print("  ref=%d  real_query=%d（按**作品**划分 ✓ 保证 query 的作品不在 gallery 中 ✓）"
              % (len(ref_i), len(qry_i)))

    print("== ② 加载风格编码器 ==", flush=True)
    enc = StyleEncoder.load(args.ckpt, kind=args.kind, device=(args.device or None))
    print("  dim=%d device=%s mask_ink=%s" % (enc.dim, enc.device, bool(args.mask_ink)), flush=True)

    print("== ③ 提特征（带缓存 ✓）==", flush=True)
    cache = os.path.join(args.out, "emb_cache.npz")
    if args.bank_pack:
        pass                                     # emb_all 已从 pack 载入 ✓
    else:
        emb_all, _ = enc.embed_paths([r.path for r in rows], cache=cache, batch=args.batch,
                                     mask_ink=bool(args.mask_ink), progress=True)
        if args.save_bank_pack:
            store.save_pack(args.save_bank_pack, emb_all, rows)
            print("  ✓ 参考库 pack 已导出: %s %s"
                  % (args.save_bank_pack, store.pack_summary(args.save_bank_pack)), flush=True)
    # ★ 一律用**行索引**索引嵌入 ✓ —— 早先按 `id(row)` 建映射, 而 split_by_work 会 replace
    #   出新对象 ⇒ 映射整体失效 ✗（KeyError: <大整数> ✓）。索引不会失效 ✓。
    rows_ref = [rows[i] for i in ref_i]
    rows_qry = [rows[i] for i in qry_i]
    # 统一标签映射（用**全部**行, 保证 ref/query/gen 的 id 一致 ✓）
    allw, alls, allc, allk = _ids(rows, "writer"), _ids(rows, "script"), _ids(rows, "char"), \
        _ids(rows, "work")
    E_ref = to_embset(rows_ref, emb_all[ref_i], allw, alls, allc, allk)
    E_realq = to_embset(rows_qry, emb_all[qry_i], allw, alls, allc, allk)
    proto = build_prototypes(E_ref)
    print("  原型 %d 个 (writer×script) ✓" % len(proto), flush=True)

    gen_map_rows = list(csv.DictReader(open(args.gen_map, encoding="utf-8-sig"))) \
        if args.gen_map else []

    report = {"encoder": {"ckpt": os.path.basename(args.ckpt), "kind": args.kind,
                          "dim": enc.dim, "mask_ink": bool(args.mask_ink)},
              "bank": {"n_rows": len(rows), "n_ref": len(rows_ref), "n_real_query": len(rows_qry),
                       "n_proto": len(proto)},
              "baselines": {}, "sets": {}}

    def full_metrics(Eq: EmbSet, tag: str, content_ok=None, bsd_ref: BrushFeat = None,
                     q_paths=None) -> dict:
        r_ret = cross_char_retrieval(Eq, E_ref, topk=(1, 5))
        r_pm = prototype_margin(Eq, proto)
        r_hwd = hwd_cn(Eq, E_ref)
        r_mmd = style_mmd_by_slot(Eq, E_ref)
        r_xct = xctacc(Eq, E_ref)
        out = {"n_query": len(Eq), "retrieval": {k: v for k, v in r_ret.items()
                                                 if k != "per_writer_map"},
               "prototype_margin": {k: v for k, v in r_pm.items() if k != "worst_pairs"},
               "hwd_cn": {"macro": r_hwd["hwd_cn_macro"], "n_used": r_hwd["n_used"]},
               "style_mmd": {"macro": r_mmd["mmd_macro"], "n_used": r_mmd["n_used"]},
               "xctacc": r_xct}
        if content_ok is not None:
            m = np.asarray(content_ok, dtype=bool)
            out["content"] = {"coverage": float(m.mean()), "n_correct": int(m.sum())}
            if m.sum() >= 8:
                sub = Eq.subset(m)
                rr = cross_char_retrieval(sub, E_ref, topk=(1,))
                out["style_at_content_correct"] = {"map": rr["map"], "top1": rr["top1"],
                                                   "n_used": rr["n_used"]}
        if args.bsd and bsd_ref is not None and q_paths:
            fq = brush_feats(enc, q_paths, args.out)
            out["bsd"] = {k: v for k, v in bsd(fq, bsd_ref).items()}
        print("  [%s] retrieval mAP=%.4f top1=%.4f | margin=%+.4f | HWD-CN=%.4f | "
              "MMD=%.5f | XCTAcc=%.4f"
              % (tag, out["retrieval"].get("map", float("nan")),
                 out["retrieval"].get("top1", float("nan")),
                 out["prototype_margin"].get("margin_mean", float("nan")),
                 out["hwd_cn"]["macro"], out["style_mmd"]["macro"],
                 out["xctacc"].get("xctacc", float("nan"))), flush=True)
        return out

    # 参考侧的笔法特征（基线/生成都用同一参考 ✓）
    bsd_ref = None
    if args.bsd:
        print("== ④ 参考侧笔法特征 ==", flush=True)
        picks = rows_ref
        if args.bsd_ref_max and len(rows_ref) > args.bsd_ref_max:
            # 按槽位**均衡抽样** ✓（否则会偏向样本最多的书家 ✗）
            per: dict = {}
            for r in rows_ref:
                per.setdefault(r.slot, []).append(r)
            quota = max(1, args.bsd_ref_max // max(len(per), 1))
            picks = []
            for slot, rs in sorted(per.items()):
                picks += rs[:quota] if len(rs) > quota else rs
            print("  BSD 参考侧抽样 %d -> %d 张（每槽位配额 %d ✓）"
                  % (len(rows_ref), len(picks), quota), flush=True)
        bsd_ref = brush_feats(enc, [r.path for r in picks], args.out)
        print("  参考侧笔法特征: 宽度中位 %.2f | 样本 %d ✓"
              % (float(np.median(bsd_ref.width)) if bsd_ref.width.size else float("nan"),
                 bsd_ref.width.size), flush=True)

    if args.baselines:
        print("== ⑤ 基线 ==", flush=True)
        report["baselines"]["real_real"] = full_metrics(
            E_realq, "Real-Real", bsd_ref=bsd_ref, q_paths=[r.path for r in rows_qry])
        # Real-Other-Writer: 把 query 的书家**故意错位**（同书体内轮换 ✓）
        wl = sorted(set(allw.values()))
        wrong = np.array([wl[(wl.index(int(w)) + 1) % len(wl)] for w in E_realq.writer])
        E_wrong = EmbSet(E_realq.emb, wrong, E_realq.script, E_realq.char, E_realq.work,
                         E_realq.name)
        report["baselines"]["real_other_writer"] = full_metrics(E_wrong, "Real-Other-Writer")
        # TTF-only: 直接用**标准字形渲染图**(std_path)当"生成样本" ✓
        ttf_rows = []
        for r in rows[:2000]:
            sp = (getattr(r, "src_path", "") or "")
        ttf_paths = []
        if args.gen_map and gen_map_rows:
            for row in gen_map_rows:
                sp = (row.get("std_path") or "").strip()
                p = B.resolve(sp, roots)
                if p:
                    ttf_paths.append(p)
        if ttf_paths:
            emb_ttf, _ = enc.embed_paths(ttf_paths, cache=os.path.join(args.out, "emb_ttf.npz"),
                                         batch=args.batch, mask_ink=bool(args.mask_ink))
            gm = [dict(writer=(row.get("calligrapher") or "").strip(),
                       script=(row.get("script") or "").strip(),
                       char=(row.get("character") or "").strip(), work="TTF", path=p)
                  for row, p in zip(gen_map_rows, ttf_paths)]
            E_ttf = EmbSet(emb_ttf, np.array([_mid(allw, g["writer"]) for g in gm]),
                           np.array([_mid(alls, g["script"]) for g in gm]),
                           np.array([_mid(allc, g["char"]) for g in gm]),
                           np.array([_mid(allk, g["work"]) for g in gm]),
                           [os.path.basename(g["path"]) for g in gm])
            report["baselines"]["ttf_only"] = full_metrics(
                E_ttf, "TTF-only", bsd_ref=bsd_ref, q_paths=ttf_paths)
        else:
            report["baselines"]["ttf_only"] = {"skipped": "无 std_path / 未给 --gen-map"}

    print("== ⑥ 生成样本 ==", flush=True)
    for spec in args.gen:
        name, _, d = spec.partition("=")
        d = d or name
        files = sorted([f for f in os.listdir(d) if f.startswith("g") and f.endswith(".png")],
                       key=lambda x: int("".join(c for c in x[1:-4] if c.isdigit()) or -1))
        if not files:
            print("  [%s] 无 g*.png ✗" % name); continue
        paths = [os.path.join(d, f) for f in files]
        emb, _ = enc.embed_paths(paths, cache=os.path.join(args.out, "emb_gen_%s.npz" % name),
                                 batch=args.batch, mask_ink=bool(args.mask_ink))
        # 与 gen-map **按索引**对齐 ✓（g{i}.png ↔ 第 i 行 ✓）
        idxs = [int("".join(c for c in f[1:-4] if c.isdigit())) for f in files]
        rows_g = [gen_map_rows[i] for i in idxs if i < len(gen_map_rows)]
        keep = [k for k, i in enumerate(idxs) if i < len(gen_map_rows)]
        emb = emb[keep]
        if not rows_g:
            print("  [%s] 无法与 --gen-map 对齐 ✗" % name); continue
        gm = [dict(writer=(r.get("calligrapher") or "").strip(),
                   script=(r.get("script") or "").strip(),
                   char=(r.get("character") or "").strip(),
                   work="gen_%s" % name, path=p)
              for r, p in zip(rows_g, [paths[k] for k in keep])]
        E_gen = EmbSet(emb, np.array([_mid(allw, g["writer"]) for g in gm]),
                       np.array([_mid(alls, g["script"]) for g in gm]),
                       np.array([_mid(allc, g["char"]) for g in gm]),
                       np.array([_mid(allk, g["work"]) for g in gm]),
                       [os.path.basename(g["path"]) for g in gm])
        # ★ 报告"字符是否在参考库内"（unseen-char 占比 ✓ —— 这是必然要分组看的 ✓）
        bank_chars = set(allc) - set(g["char"] for g in gm) if False else None
        fixed = {r.char for r in rows}
        n_unseen = sum(1 for g in gm if g["char"] not in fixed)
        E_gen_unseen = E_gen.subset(np.array([g["char"] not in fixed for g in gm], dtype=bool))
        E_gen_seen = E_gen.subset(np.array([g["char"] in fixed for g in gm], dtype=bool))
        print("  字符分组: seen=%d unseen=%d（unseen 在库里没有同字候选 ✓ 正是要评的 ✓）"
              % (len(E_gen_seen), len(E_gen_unseen)), flush=True)
        if len(E_gen_unseen) >= 8:
            report.setdefault("sets", {}).setdefault(name + "__unseen_char", {})
            report["sets"][name + "__unseen_char"] = full_metrics(
                E_gen_unseen, name + "__unseen_char")
        # 内容判定: 生成图 char 预测 == 同行 std 图 char 预测 ✓
        std_paths = [B.resolve((r.get("std_path") or "").strip(), roots) for r in rows_g]
        ok = None
        if all(std_paths):
            p_gen = ocr_pred(enc, [paths[k] for k in keep], args.out, args.batch)
            p_std = ocr_pred(enc, std_paths, args.out, args.batch)
            ok = (p_gen == p_std)
        report["sets"][name] = full_metrics(E_gen, name, content_ok=ok, bsd_ref=bsd_ref,
                                            q_paths=[paths[k] for k in keep])

    report["elapsed_s"] = round(time.time() - t0, 1)
    with open(os.path.join(args.out, "style_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=float)

    # ── 对比表（基线 + 各生成集, 同一口径 ✓）──
    def _row(tag: str, d: dict) -> str:
        r = d.get("retrieval", {}) or {}
        pm = d.get("prototype_margin", {}) or {}
        b = d.get("bsd", {}) or {}
        return "| %-20s | %6s | %6s | %8s | %8s | %9s | %7s | %7s | %6s |" % (
            tag,
            _f(r.get("map")), _f(r.get("top1")), _f(pm.get("margin_mean")),
            _f((d.get("hwd_cn") or {}).get("macro")),
            _f((d.get("style_mmd") or {}).get("macro")),
            _f((d.get("xctacc") or {}).get("xctacc")),
            _f(b.get("width")), _f(b.get("bsd_macro")))

    lines = ["| 集合 | mAP | Top-1 | Δmargin | HWD-CN | Style-MMD | XCTAcc | BSD宽 | BSD宏 |",
             "|---|---|---|---|---|---|---|---|---|"]
    for k, d in (report.get("baselines") or {}).items():
        if isinstance(d, dict) and "skipped" not in d:
            lines.append(_row(k, d))
    for k, d in (report.get("sets") or {}).items():
        lines.append(_row(k, d))
    table = "\n".join(lines)
    with open(os.path.join(args.out, "style_report.md"), "w", encoding="utf-8") as f:
        f.write("# Style 轴评测（%s, kind=%s, mask_ink=%s）\n\n"
                % (os.path.basename(args.ckpt), args.kind, bool(args.mask_ink)))
        f.write("参考库: %d 张 / gallery %d / 真实 query %d / 原型 %d\n\n"
                % (report["bank"]["n_rows"], report["bank"]["n_ref"],
                   report["bank"]["n_real_query"], report["bank"]["n_proto"]))
        f.write(table + "\n\n")
        f.write("判读: 目标是**生成集**的 mAP/Δmargin/XCTAcc 接近 `real_real` 且明显好于 "
                "`real_other_writer` 与 `ttf_only` ✓；HWD-CN/Style-MMD/BSD 越小越好 ✓。\n")
        for k, d in (report.get("sets") or {}).items():
            if isinstance(d, dict) and "content" in d:
                f.write("\n- `%s`: 内容覆盖率 %.3f（Style@ContentCorrect mAP=%s）\n"
                        % (k, d["content"]["coverage"],
                           _f((d.get("style_at_content_correct") or {}).get("map"))))
    print("\n" + table)
    print("\n[done] %s/style_report.{json,md}（%.1fs）" % (args.out, report["elapsed_s"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
