# -*- coding: utf-8 -*-
"""新版审查 §三+§五 六项正确性修复的回归测试（2026-10-16 ✓ 精简版 ✓）"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_weight_hash_is_content_bound():
    from src.eval.in_mem_eval import _model_weight_sha256

    class M(torch.nn.Module):
        def __init__(self, v):
            super().__init__()
            self.lin = torch.nn.Linear(2, 2)
            with torch.no_grad():
                self.lin.weight.fill_(v)

    assert _model_weight_sha256(M(1.0), tag="w::1") != _model_weight_sha256(M(2.0), tag="w::1")


def test_safe_name_no_collision():
    from src.eval.artifacts import safe_name
    assert safe_name("a/b") != safe_name("a b")
    assert safe_name("a/b") == safe_name("a/b")


def test_scoring_strict_on_missing(tmp_path):
    from src.eval.scoring import compute_metrics
    from PIL import Image
    import numpy as np
    for i in range(2):
        Image.fromarray(np.zeros((4, 4, 3), np.uint8)).save(str(tmp_path / f"p{i}.png"))
        Image.fromarray(np.zeros((4, 4, 3), np.uint8)).save(str(tmp_path / f"gt{i}.png"))
    try:
        compute_metrics(str(tmp_path), str(tmp_path), "p", 3, use_lpips=False)
        raise AssertionError("缺失未报错 ✗")
    except FileNotFoundError:
        pass
    assert compute_metrics(str(tmp_path), str(tmp_path), "p", 3, use_lpips=False, strict=False)["n"] == 2


def test_done_key_has_ckpt_sha():
    from src.eval.in_mem_eval import _eval_done_key
    kw = (1000, "s", 0.8, 50, 0.18, "a.csv", "rev")
    assert _eval_done_key(*kw, ckpt_sha="a" * 8) != _eval_done_key(*kw, ckpt_sha="b" * 8)
    assert _eval_done_key(*kw, ckpt_sha="") != _eval_done_key(*kw, ckpt_sha="a" * 8)


def test_done_requires_metrics(tmp_path):
    from src.eval import protocol as P
    from src.eval.protocol import EvalRequest
    req = EvalRequest(
        checkpoint_sha256="a" * 64, vae_sha256="b" * 64, vae_scale=0.18, vae_shift=0.0,
        sampler="heun", num_steps=50, cfg=0.8, gamma=0.0, seed=0, ema=True,
        code_revision="r", dataset_sha256="c" * 64, conditions_sha256="d" * 64,
        sets_spec="s:x.csv:3", extra={})
    d = str(tmp_path)
    P.write_protocol(req, d)
    open(os.path.join(d, "samples.jsonl"), "w").write('{"idx":0}\n')
    open(os.path.join(d, "g0.png"), "wb").write(b"\x89PNG")
    P.write_done(req, d, n_samples=1, png_dir=d)
    ok, why = P.is_complete(req, d)
    assert not ok and "metrics.json" in why
    open(os.path.join(d, "metrics.json"), "w").write("{}")
    P.write_done(req, d, n_samples=1, png_dir=d)
    assert P.is_complete(req, d)[0]


def test_raw_csv_img_id():
    from src.eval.in_mem_eval import _RAW_HDR
    assert _RAW_HDR.index("img_id") == 4
    assert _RAW_HDR[-2:] == ["sample_id", "image_path"]
