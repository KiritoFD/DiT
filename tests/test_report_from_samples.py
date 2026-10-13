# -*- coding: utf-8 -*-
"""tests/test_report_from_samples.py — 离线重算入口的行为测试（[审查 §4.2]）

重点验证**审查点名的两条**:
  ① "不应该重新调用生成模型" ⇒ 本工具不 import torch 模型 / 不读 ckpt ✓（结构性检查 ✓）
  ② "不应该从旧图目录猜哪个文件是最新结果" ⇒ 模板算不出文件就**报错退出**, 绝不跳过/模糊匹配 ✓✓

运行: python -m pytest tests/test_report_from_samples.py -q
"""
from __future__ import annotations

import csv
import importlib.util
import io
import os
import sys
import tempfile

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_spec = importlib.util.spec_from_file_location(
    "rfs", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "tools", "report_from_samples.py"))
rfs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rfs)


def _png(path, arr):
    Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8)).save(path)
    return path


def _glyph(shift=0, h=32, w=32, thick=2):
    a = np.ones((h, w), np.float32)
    c = h // 2
    a[c - thick:c + thick, w // 4: 3 * w // 4] = 0.0
    a[h // 4: 3 * h // 4, c - thick:c + thick] = 0.0
    return np.roll(a, shift, axis=1)


def _setup(tmp):
    """造一个最小但**真实形态**的评测集: eval CSV(带 img_id/image_path) + 生成图 ✓"""
    gen_dir = os.path.join(tmp, "gen")
    gt_dir = os.path.join(tmp, "gt")
    os.makedirs(gen_dir), os.makedirs(gt_dir)
    rows = []
    for i, (iid, shift) in enumerate([(22004, 0), (9150, 0)]):
        gt = _png(os.path.join(gt_dir, "%d.png" % iid), _glyph(shift=0))
        _png(os.path.join(gen_dir, "%d.png" % iid), _glyph(shift=shift))
        rows.append({"img_id": str(iid), "image_path": gt, "character": "戶",
                     "script": "楷", "calligrapher_id": "53", "script_id": "0",
                     "character_id": "2016", "glyph_id": "2016"})
    csvp = os.path.join(tmp, "eval.csv")
    with io.open(csvp, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return csvp, gen_dir


# ─────────── ① ★ 结构性: 不加载生成模型 ✓（审查"不重新调用生成模型"）───────────
def test_tool_has_no_model_dependency():
    src = io.open(rfs.__file__, encoding="utf-8").read()
    for bad in ("torch.load", "load_model_from_ckpt", "build_model_from_args",
                "sample_latents", "AutoencoderKL", "from diffusers"):
        assert bad not in src, ("离线重算工具**不许**触碰生成模型 ✗✗（审查: 不重新调用生成模型），"
                                f"发现: {bad}")
    assert "import torch" not in src, "本工具连 torch 都不应直接 import ✓（只走 numpy/PIL ✓）"


# ─────────── ② 模板配对: 正确 + **绝不猜** ✓✓ ───────────
def test_build_pairs_by_template_and_refuse_to_guess():
    tmp = tempfile.mkdtemp()
    csvp, gen_dir = _setup(tmp)
    pairs = rfs.build_pairs_from_eval_csv(csvp, gen_dir, "{img_id}.png")
    assert [p.sample_id for p in pairs] == ["22004", "9150"], "配对必须按 CSV 的 img_id ✓"
    assert all(os.path.isfile(p.generated_path) and os.path.isfile(p.gt_path) for p in pairs)
    # ★ 模板算不出文件 ⇒ 必须**报错退出**, 不许"跳过/模糊匹配" ✗
    try:
        rfs.build_pairs_from_eval_csv(csvp, gen_dir, "nonexistent_{img_id}.png")
        raise AssertionError("找不到文件时必须报错退出 ✓（审查: 不该猜哪个文件是最新结果）")
    except SystemExit as e:
        assert int(e.code) == 2, "应以 exit=2 明确失败 ✓"


# ─────────── ③ 端到端: 由 eval CSV 出 metrics.json（含两种身份 ✓）───────────
def test_end_to_end_writes_metrics_with_both_identities():
    tmp = tempfile.mkdtemp()
    csvp, gen_dir = _setup(tmp)
    out = os.path.join(tmp, "out")
    rc = rfs.main(["--eval-csv", csvp, "--gen-dir", gen_dir, "--out", out,
                   "--write-samples-jsonl"])
    assert rc == 0
    import json
    m = json.load(io.open(os.path.join(out, "metrics.json"), encoding="utf-8"))
    for k in ("generation_id", "metrics_id", "aggregate", "near_blank_rate", "samples"):
        assert k in m, "metrics.json 缺 %s ✗" % k
    assert m["n_samples"] == 2 and len(m["samples"]) == 2
    # 生成图 = GT ⇒ 墨迹 IoU 必须是满分 ✓（原来这里被我留了一行**恒真空断言** ✗, 已换掉 ✓）
    assert abs(m["aggregate"]["ink_iou"]["mean"] - 1.0) < 1e-6
    # 生成清单必须已写出, 且**不含任何 GT/图片路径** ✓✓
    sj = io.open(os.path.join(out, "samples.jsonl"), encoding="utf-8").read()
    assert "gt_path" not in sj and ".png" not in sj, "生成清单里不许出现 GT/图片路径 ✗✗"


# ─────────── ④ 参数守卫: 两种输入**必须且只能**给一种 ✓ ───────────
def test_input_mode_guards():
    tmp = tempfile.mkdtemp()
    csvp, gen_dir = _setup(tmp)
    for argv in ([], ["--pairs-csv", "x.csv", "--eval-csv", "y.csv"],
                 ["--eval-csv", csvp]):            # 缺 --gen-dir/模板 ⇒ 报错 ✓
        try:
            rfs.main(argv + ["--out", os.path.join(tmp, "o")])
            raise AssertionError("非法参数组合必须报错 ✓: %r" % (argv,))
        except SystemExit:
            pass


# ─────────── ⑤ 显式配对表路径也能用（推荐路径 ✓）───────────
def test_explicit_pairs_csv_path():
    tmp = tempfile.mkdtemp()
    csvp, gen_dir = _setup(tmp)
    pairs = rfs.build_pairs_from_eval_csv(csvp, gen_dir, "{img_id}.png")
    pp = os.path.join(tmp, "pairs.csv")
    with io.open(pp, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample_id", "generated_path", "gt_path"])
        for p in pairs:
            w.writerow([p.sample_id, p.generated_path, p.gt_path])
    out = os.path.join(tmp, "out2")
    assert rfs.main(["--pairs-csv", pp, "--out", out]) == 0
    import json
    m = json.load(io.open(os.path.join(out, "metrics.json"), encoding="utf-8"))
    assert m["generation_id_source"].startswith("derived_from_pairs"), \
        "没有清单时必须**标注**身份是派生的 ✓（不假装是权威身份 ✗）"


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
