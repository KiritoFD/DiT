# -*- coding: utf-8 -*-
"""test_refactor_p3.py —— 2026-10-14 重构 P1-7 的**验收测试** ✓

P1-7: 代码来源清单必须覆盖**实验实际用到的实现** ✓
  审查: 只哈希 `src/**/*.py` ✗ ⇒ Critic（从 `tools/train_writer_brush_critic.py`
  `importlib` 动态加载 ✗）**不在清单里** ⇒ 无法证明实验用了哪版 Critic ✗✗
  且没有对 模型权重 / VAE / 字体 / 数据 manifest / 生成器 / 评测器 分别记录 SHA256 ✗
"""
from __future__ import annotations

import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.train import manifest as MF                                      # noqa: E402


def test_sha256_file_and_missing(tmp_path):
    p = os.path.join(str(tmp_path), "a.bin")
    io.open(p, "wb").write(b"hello")
    h = MF.sha256_file(p)
    assert isinstance(h, str) and len(h) == 64
    assert MF.sha256_file(os.path.join(str(tmp_path), "nope.bin")) is None, \
        "缺失文件必须返回 None ✗（不得当 0 或空串冒充 ✓）"


def test_sha256_tree_detects_change(tmp_path):
    d = str(tmp_path)
    io.open(os.path.join(d, "a.json"), "w", encoding="utf-8").write('{"x":1}')
    t1, n1, trunc1 = MF.sha256_tree(d)
    assert n1 == 1 and not trunc1 and t1["n"] == 1
    io.open(os.path.join(d, "a.json"), "w", encoding="utf-8").write('{"x":2}')
    t2, _, _ = MF.sha256_tree(d)
    assert t1["tree_sha256"] != t2["tree_sha256"], "内容变 ⇒ 指纹必须变 ✓"


def test_sha256_tree_respects_exts(tmp_path):
    d = str(tmp_path)
    io.open(os.path.join(d, "a.json"), "w", encoding="utf-8").write("{}")
    io.open(os.path.join(d, "big.npy"), "wb").write(b"\x00" * 100)
    t, n_seen, _ = MF.sha256_tree(d)                    # 默认白名单 ✓ 不含 .npy ✓
    assert t["n"] == 1 and "big.npy" not in t["files"]


def test_sha256_tree_truncation_is_explicit(tmp_path):
    d = str(tmp_path)
    for i in range(5):
        io.open(os.path.join(d, "f%d.json" % i), "w", encoding="utf-8").write("{}")
    t, n_seen, trunc = MF.sha256_tree(d, max_files=2)
    assert t is None and trunc is True and n_seen == 5, \
        "超限必须**显式**返回 None + truncated ✓（不得假装完整 ✗）"


def test_code_roots_cover_tools():
    """★ P1-7 核心 ✓: 代码哈希必须覆盖 **tools/**（Critic 在那里 ✗）"""
    man = MF.write_source_manifest.__doc__ or ""
    assert "tools" in man, "文档必须说明默认覆盖 tools/ ✓"
    per, files, code_tree = MF._hash_code_roots(("src", "tools"))
    assert per.get("src", {}).get("n", 0) > 50, "src 应有大量 .py ✓"
    assert per.get("tools", {}).get("n", 0) > 0, "**tools/ 必须被哈希** ✓（P1-7 ✗）"
    assert any(p.startswith("tools/") for p in files), files[:3]
    assert len(code_tree) == 64


def test_hash_artifacts_file_dir_and_missing(tmp_path):
    d = str(tmp_path)
    f = os.path.join(d, "w.pt")
    io.open(f, "wb").write(b"weights")
    sub = os.path.join(d, "vae")
    os.makedirs(sub)
    io.open(os.path.join(sub, "config.json"), "w", encoding="utf-8").write("{}")
    art, missing = MF._hash_artifacts({
        "ckpt": f, "vae": sub, "gone": os.path.join(d, "nope.pt"), "empty": ""})
    assert art["ckpt"]["kind"] == "file" and len(art["ckpt"]["sha256"]) == 64
    assert art["vae"]["kind"] == "dir" and art["vae"]["n"] == 1
    assert sorted(missing) == ["empty", "gone"], missing


def test_write_source_manifest_end_to_end(tmp_path):
    d = str(tmp_path)
    f = os.path.join(d, "w.pt")
    io.open(f, "wb").write(b"w")
    man = MF.write_source_manifest(d, artifacts={"ckpt": f, "nope": "x/y.pt"},
                                   src_roots=("src",), verbose=False)
    p = os.path.join(d, "source_manifest.json")
    assert os.path.isfile(p), "必须落盘 ✓"
    on_disk = json.load(io.open(p, encoding="utf-8"))
    assert man["code_roots"] == ["src"]
    assert "code_tree_sha256" in man and "artifacts" in man
    assert on_disk["artifacts_missing"] == ["nope"], on_disk["artifacts_missing"]
    assert "git_short" in man and "argv" in man
    # 向后兼容字段仍在 ✓
    assert "src_tree_sha256" in man and "src_files_sha256" in man


def test_write_source_manifest_backcompat_src_root(tmp_path):
    """旧签名 `src_root="src"` 仍可用 ✓（调用方未迁移时不破坏 ✓）"""
    man = MF.write_source_manifest(str(tmp_path), src_root="src", verbose=False)
    assert man["code_roots"] == ["src"]
    assert man["src_py_count"] > 50
