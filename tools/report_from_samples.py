#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""report_from_samples.py — **离线重算指标 / 报告**（2026-10-13 [审查 §4.2]）

审查原文:
  "算新指标、重新出 poster，不应该重新调用生成模型，
   也不应该从已经存在的旧图目录猜哪个文件是最新结果。"

本工具就是那条要求的**可执行入口** ✓:
  · **零模型依赖**: 不加载 VAE / DiT / checkpoint —— 只读 PNG ✓（可在任何机器上跑 ✓）
  · **只接受显式配对** ✓: 要么给 --pairs-csv（sample_id,generated_path,gt_path ✓）,
    要么给 --eval-csv + --gen-dir + --gen-name-template（**声明式**命名规则 ✓, 不猜 ✗）
  · 输出 metrics.json, 里面同时记 **generation_id**（生成身份）与 **metrics_id**（指标身份）✓
    => 换 GT 或加新指标时: metrics_id 变、**generation_id 不变** ✓✓（不会误判为要重新生成 ✗）
  · 若给了 --samples-jsonl（生成清单 ✓）, 会用它算 generation_id 并可写出同一目录 ✓

⚠ 它**不写 DONE.json** ✓ —— DONE 属"一次真实评测"的完成标记（见 src/eval/protocol.py ✓）,
  而重算指标是**派生**动作 ✓（生成没变就不该动完成标记 ✓）。

用法:
  # ① 显式配对（推荐）
  $PY tools/report_from_samples.py --pairs-csv pairs.csv --out ev/ --need-lpips
  # ② 由评测 CSV + 生成目录（**声明式**命名模板 ✓）
  $PY tools/report_from_samples.py --eval-csv exp-std/csv/eval200fix.csv \\
      --gen-dir out/eval200fix --gen-name-template '{img_id}.png' --out out/eval200fix
"""
from __future__ import annotations

import argparse
import csv
import io
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eval.report import (GenerationSample, ScoringSample, generation_id,   # noqa: E402
                             metrics_id, read_samples_jsonl, score_samples, write_metrics)


def read_pairs_csv(path):
    """显式配对表: 需要 sample_id / generated_path / gt_path 三列 ✓（缺列即报错 ✗）"""
    with io.open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit("[report] --pairs-csv 为空 ✗")
    need = ("sample_id", "generated_path", "gt_path")
    miss = [c for c in need if c not in rows[0]]
    if miss:
        raise SystemExit("[report] --pairs-csv 缺列 %s ✗（需要 sample_id/generated_path/gt_path ✓）"
                         % ", ".join(miss))
    return [ScoringSample(r["sample_id"].strip(), r["generated_path"].strip(), r["gt_path"].strip(),
                          meta={k: v for k, v in r.items()
                                if k not in need and v not in (None, "")}) for r in rows]


def build_pairs_from_eval_csv(eval_csv, gen_dir, template, *, id_col="img_id",
                              gt_col="image_path", limit=None):
    """由评测 CSV + 生成目录**按声明式模板**配对 ✓（模板里可用 {img_id} / {idx} ✓）

    刻意**不做模糊匹配** ✗: 模板算出的文件名不存在 ⇒ 直接报错并列出前几个缺失 ✓,
    避免"猜错文件却照样出分"这类历史事故 ✓。
    """
    with io.open(eval_csv, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit("[report] --eval-csv 为空 ✗")
    for c in (id_col, gt_col):
        if c not in rows[0]:
            raise SystemExit("[report] --eval-csv 缺列 %r ✗（现有列: %s）"
                             % (c, ",".join(list(rows[0].keys())[:12])))
    pairs, missing = [], []
    for i, r in enumerate(rows if limit is None else rows[:limit]):
        sid = str(r[id_col]).strip()
        # NOTE 2026-10-13: 不能写成 template.format(img_id=sid, **r) —— CSV 本身带 img_id 列时
        #   会**重复关键字**报 TypeError（实测踩到）。先拷一份再显式覆盖 ✓。
        _kw = dict(r)
        _kw["img_id"], _kw["idx"] = sid, i
        try:
            name = template.format(**_kw)
        except KeyError as _e:
            raise SystemExit("[report] --gen-name-template 引用了 CSV 里没有的列 %s ✗" % _e)
        gp = os.path.join(gen_dir, name)
        gt = r[gt_col].strip()
        if not os.path.isfile(gp) or not os.path.isfile(gt):
            missing.append((sid, gp, gt))
            continue
        pairs.append(ScoringSample(sid, gp, gt, meta={"character": r.get("character", ""),
                                                     "script": r.get("script", "")}))
    if missing:
        print("[report] ✗ %d/%d 个样本按模板找不到文件（**不猜** ✗），前 3 个:" % (len(missing), len(rows)))
        for sid, gp, gt in missing[:3]:
            print("        id=%s gen=%s exists=%s | gt=%s exists=%s"
                  % (sid, gp, os.path.isfile(gp), gt, os.path.isfile(gt)))
        raise SystemExit(2)
    return pairs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs-csv", help="显式配对表（sample_id,generated_path,gt_path）✓ 与 ② 二选一")
    ap.add_argument("--eval-csv", help="评测集 CSV（含 img_id / image_path ✓）")
    ap.add_argument("--gen-dir", help="生成图目录（与 --gen-name-template 搭配 ✓）")
    ap.add_argument("--gen-name-template", default="{img_id}.png",
                    help="**声明式**生成文件名模板 ✓（如 '{img_id}.png'）；不做模糊匹配 ✗")
    ap.add_argument("--id-col", default="img_id", help="评测 CSV 里的样本 id 列（默认 img_id ✓）")
    ap.add_argument("--gt-col", default="image_path", help="评测 CSV 里的 GT 路径列（默认 image_path ✓）")
    ap.add_argument("--samples-jsonl", help="生成清单（可选 ✓）；给了就用它算 generation_id ✓")
    ap.add_argument("--write-samples-jsonl", action="store_true",
                    help="把（由生成图推出的）生成清单写到 --out/samples.jsonl ✓")
    ap.add_argument("--out", required=True, help="输出目录（写 metrics.json ✓）")
    ap.add_argument("--size", type=int, default=None, help="统一缩放边长（与训练内评测对齐时用 ✓）")
    ap.add_argument("--need-lpips", action="store_true")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args(argv)

    if bool(a.pairs_csv) == bool(a.eval_csv):
        raise SystemExit("[report] 必须且只能给一种输入: --pairs-csv 或 --eval-csv ✗（不猜 ✗）")
    if a.eval_csv and not (a.gen_dir and a.gen_name_template):
        raise SystemExit("[report] --eval-csv 需要同时给 --gen-dir 与 --gen-name-template ✓")

    samples = read_pairs_csv(a.pairs_csv) if a.pairs_csv else \
        build_pairs_from_eval_csv(a.eval_csv, a.gen_dir, a.gen_name_template,
                                  id_col=a.id_col, gt_col=a.gt_col, limit=a.limit)
    size = (a.size, a.size) if a.size else None
    print("[report] 待评分 %d 个样本（size=%s lpips=%s）" % (len(samples), size, a.need_lpips))

    payload = score_samples(samples, size=size, need_lpips=a.need_lpips,
                            device=a.device, progress=True)

    # 生成身份: 有清单就按清单算 ✓; 否则由配对表**派生**一个最小清单并**明确标注** ✓
    if a.samples_jsonl:
        gs = read_samples_jsonl(a.samples_jsonl)
        src = "samples_jsonl"
    else:
        gs = [GenerationSample(sample_id=s.sample_id, writer_id=int(s.meta.get("calligrapher_id", -1) or -1),
                               script_id=int(s.meta.get("script_id", -1) or -1),
                               char_id=int(s.meta.get("character_id", -1) or -1) or None,
                               glyph_key=s.meta.get("glyph_id") or None, seed=0)
              for s in samples]
        src = "derived_from_pairs(note: 不含量化生成参数 ⇒ 仅用于一致性核对 ✓)"
    payload["generation_id"] = generation_id(gs)
    payload["generation_id_source"] = src
    payload["metrics_id"] = metrics_id(gs, metric_set=payload["metric_set"])
    out = Path(a.out)
    p = write_metrics(str(out), payload)
    if a.write_samples_jsonl and not a.samples_jsonl:
        from src.eval.report import write_samples_jsonl
        write_samples_jsonl(str(out / "samples.jsonl"), gs)
        print("[report] ✓ 已写生成清单 %s" % (out / "samples.jsonl"))

    ag = payload["aggregate"]
    print("[report] ✓ %s" % p)
    print("         n=%d | near_blank_rate=%.4f | ssim=%.4f | ink_iou=%.4f | skel_iou=%.4f"
          % (payload["n_samples"], payload["near_blank_rate"],
             ag["ssim"]["mean"], ag["ink_iou"]["mean"], ag["skel_iou"]["mean"]))
    print("         generation_id=%s" % payload["generation_id"][:16])
    print("         metrics_id   =%s  (换 GT/加指标 ⇒ 此值变, 而 generation_id 不变 ✓)"
          % payload["metrics_id"][:16])
    return 0


if __name__ == "__main__":
    sys.exit(main())
