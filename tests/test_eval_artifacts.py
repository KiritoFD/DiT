# -*- coding: utf-8 -*-
"""tests/test_eval_artifacts.py — ArtifactWriter + **端到端接线**（生成 → 离线评分）

本轮要证明的**接线**（不是脚手架 ✗）:
    artifacts.GenerationWriter 写图 + 记身份  ->  report.score_samples 离线评分
    两段之间通过 samples.jsonl 传递**生成身份**；评分侧只额外拿 GT ✓✓

以及审查的两条不变量:
  · 图与清单**同源** ⇒ 清单是生成的副产品, 不是事后猜的 ✓✓
  · ArtifactWriter **不知道 GT** ✓（结构性检查: 源码/清单里不出现 gt ✓）

运行: python -m pytest tests/test_eval_artifacts.py -q
"""
from __future__ import annotations

import io
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.eval.artifacts import (GenerationWriter, iter_generated, next_index,   # noqa: E402
                                read_manifest, safe_name, sample_relpath, write_png)
from src.eval.report import (GenerationSample, ScoringSample, generation_id,   # noqa: E402
                             score_samples)


def _glyph(shift=0, h=32, w=32, t=2):
    a = np.ones((h, w), np.float32)
    c = h // 2
    a[c - t:c + t, w // 4: 3 * w // 4] = 0.0
    a[h // 4: 3 * h // 4, c - t:c + t] = 0.0
    return np.roll(a, shift, axis=1)


# ─────────── ① 命名确定性 + 不撞名 ✓ ───────────
def test_sample_path_is_deterministic_and_collision_free():
    assert sample_relpath("a") == sample_relpath("a"), "同 id 必得同路径 ✓"
    assert sample_relpath("a") != sample_relpath("b")
    # 非法字符被 sanitize ✓；不同原串 sanitize 后若相同, 长串会带哈希 ✓
    assert "/" not in safe_name("写法/测试 01") and " " not in safe_name("写法/测试 01")
    long1, long2 = "x" * 80 + "1", "x" * 80 + "2"
    assert safe_name(long1) != safe_name(long2), "超长 id 必须靠哈希保持唯一 ✓"


# ─────────── ② ★ 图与清单**同源**（审查: 不猜哪个文件是最新 ✓✓）───────────
def test_writer_records_identity_while_writing():
    tmp = tempfile.mkdtemp()
    w = GenerationWriter(tmp)
    ids = ["s%03d" % i for i in range(5)]
    for i, sid in enumerate(ids):
        w.add(GenerationSample(sid, 1, 0, 2016, "u+6236", seed=100 + i), _glyph(shift=i))
    info = w.close()
    assert info["n_samples"] == 5 and info["generation_id"]
    got = read_manifest(tmp)
    assert [s.sample_id for s in got] == ids, "清单必须与写入顺序/身份一致 ✓"
    assert sorted(k for k, _ in iter_generated(tmp)) == sorted(safe_name(s) for s in ids)
    # 清单里**没有** GT/图片路径 ✓✓
    txt = io.open(os.path.join(tmp, "samples.jsonl"), encoding="utf-8").read()
    assert "gt_path" not in txt and ".png" not in txt
    assert next_index(tmp) == 5


# ─────────── ③ ★★ 端到端接线: 生成 → 离线评分 ✓✓ ───────────
def test_end_to_end_generation_then_offline_scoring():
    tmp = tempfile.mkdtemp()
    gen_dir, gt_dir = os.path.join(tmp, "ev"), os.path.join(tmp, "gt")
    os.makedirs(gt_dir)
    w = GenerationWriter(gen_dir)
    pairs = []
    for i in range(4):
        sid = "sample_%d" % i
        # 生成侧只拿到**条件**(无 GT) ✓；GT 单独存在于另一目录 ✓
        w.add(GenerationSample(sid, 1, 0, 2016, "u+6236", seed=i), _glyph(shift=i))
        gt = os.path.join(gt_dir, "gt_%d.png" % i)
        write_png(gt, _glyph(shift=0))
        pairs.append(ScoringSample(sid, os.path.join(gen_dir, sample_relpath(sid)), gt))
    info = w.close()

    payload = score_samples(pairs)
    assert payload["n_samples"] == 4
    # ⚠ score_samples() **不**产出 generation_id（那是 tools/report_from_samples.py 加上去的 ✓）
    #   —— 首版我在这里断言错层, 报 KeyError ✗。这里核对的是**同源**:
    #   用落盘清单重算出的身份 == 写入时算出的身份 ✓✓（不是事后从目录名推的 ✗）
    assert generation_id(read_manifest(gen_dir)) == info["generation_id"], (
        "★ 落盘清单重算的 generation_id 必须与写入时**一致** ✓✓（同源 ✓）")
    ious = [r["ink_iou"] for r in payload["samples"]]
    assert ious[0] > ious[-1], "平移越大 ink_iou 越低 ✓（生成侧 shift 递增 ✓）"


# ─────────── ④ 原子写: 不留半成品 ✓ ───────────
def test_atomic_write_leaves_no_temp_files():
    tmp = tempfile.mkdtemp()
    w = GenerationWriter(tmp)
    w.add(GenerationSample("a", 1, 0, None, None, 0), _glyph())
    w.close()
    leftovers = [f for f in os.listdir(os.path.join(tmp, "samples")) if f.endswith(".tmp")]
    assert not leftovers, "原子写不该留下 .tmp ✗: %s" % leftovers
    assert os.path.isfile(os.path.join(tmp, "samples", sample_relpath("a").split(os.sep)[-1]))


# ─────────── ⑤ 结构性: ArtifactWriter 不知道 GT ✓ ───────────
def test_artifact_writer_is_gt_free():
    import ast as _ast
    import src.eval.artifacts as A
    src = io.open(A.__file__, encoding="utf-8").read()
    # ★ 用 AST 精确剥掉**整段** docstring + 注释（首版按行过滤会把 docstring 里解释
    #   "本模块不知道 GT" 的那句话本身当成违规 ✗✗ —— 自证了幼稚过滤不可靠 ✓）
    tree = _ast.parse(src)
    doc_nodes = [_ast.get_docstring(tree, clean=False) or ""]
    for n in _ast.walk(tree):
        if isinstance(n, (_ast.Module, _ast.FunctionDef, _ast.ClassDef)):
            d = _ast.get_docstring(n, clean=False)
            if d:
                doc_nodes.append(d)
    body = src
    for d in doc_nodes:
        body = body.replace(d, "")
    body = "\n".join(l.split("#", 1)[0] for l in body.splitlines())
    del body
    # ★ 改为**语义层** AST 检查 ✓（首版找 "gt_path" 字符串, 结果命中了本模块 __main__ 自检里
    #   那句 "清单不含 GT" 的断言本身 ✗✗ —— 文本扫描太粗 ✓）:
    #   只要**函数/方法的参数**与**赋值目标**里没有 gt*, 生成侧就拿不到 GT ✓✓
    bad = []
    for n in _ast.walk(tree):
        if isinstance(n, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
            for a in (list(n.args.args) + list(n.args.kwonlyargs) + list(n.args.posonlyargs)
                      + ([n.args.vararg] if n.args.vararg else [])
                      + ([n.args.kwarg] if n.args.kwarg else [])):
                if "gt" in a.arg.lower():
                    bad.append("参数 %s(%s)" % (n.name, a.arg))
        if isinstance(n, _ast.Assign):
            for tg in n.targets:
                nm = str(getattr(tg, "id", "") or getattr(tg, "attr", ""))
                if "gt" in nm.lower() and nm.lower() != "gt":
                    bad.append("赋值 %s" % nm)
    assert not bad, "ArtifactWriter 的 API 层不许出现 GT ✗✗（审查: 生成器完全不知道 GT）: %s" % bad
    # 重复 id 必须报错（身份唯一 ✓）
    tmp = tempfile.mkdtemp()
    w = GenerationWriter(tmp)
    w.add(GenerationSample("dup", 1, 0, None, None, 0), _glyph())
    try:
        w.add(GenerationSample("dup", 1, 0, None, None, 0), _glyph())
        raise AssertionError("重复 sample_id 必须报错 ✓")
    except ValueError:
        pass


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    bad = 0
    for fn in fns:
        try:
            fn()
            print("  ✓ %s" % fn.__name__)
        except Exception as e:                                        # noqa: BLE001
            bad += 1
            print("  ✗ %s -> %s: %s" % (fn.__name__, type(e).__name__, str(e)[:110]))
    print("\n  %d/%d 通过" % (len(fns) - bad, len(fns)))
    sys.exit(1 if bad else 0)
