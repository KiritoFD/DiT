# -*- coding: utf-8 -*-
"""manifest.py — 运行凭据落盘（2026-10-13 [审查 P0]）

审查发现: `train.py` 原来的 source_manifest 只哈希**根目录的旧文件名**
("models.py"/"train.py"/"losses.py"…) ✗ —— 这些文件早已迁到 `src/…` ⇒ 清单实际为空,
"这次跑的是哪份代码"无法追溯 ✗。

本模块把它做成自包含函数（新增文件比在 2800 行文件里做多点手术安全得多 ✓）:
  · source_manifest.json: Git commit(含 dirty) + 工作区改动 + **真实 src/ 全部 .py 哈希**
    + python/platform/torch/argv ⇒ 与 effective_config.json 一起构成可复现凭据 ✓
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys

__all__ = ["write_source_manifest"]


def _git(*a):
    try:
        r = subprocess.run(["git", *a], capture_output=True, text=True, timeout=20)
        return (r.stdout or "").strip()
    except Exception:                                              # noqa: BLE001
        return ""


def write_source_manifest(experiment_dir, extra=None, src_root="src", verbose=True):
    """把"这次跑的是哪份代码"写成 source_manifest.json ✓（失败不致命, 只告警 ✓）"""
    try:
        files = []
        for root, dirs, fs in os.walk(src_root):
            dirs[:] = [d for d in dirs if d != "__pycache__"]
            files += [os.path.join(root, f).replace("\\", "/") for f in fs if f.endswith(".py")]
        files.sort()
        hashes, hjoin = {}, hashlib.sha256()
        for p in files:
            try:
                b = open(p, "rb").read()
            except OSError:
                continue
            h = hashlib.sha256(b).hexdigest()
            hashes[p] = h
            hjoin.update(p.encode() + b"|" + h.encode() + b"\n")
        dirty = bool(_git("status", "--porcelain"))
        man = {
            "git_commit": _git("rev-parse", "HEAD"),
            "git_short": _git("rev-parse", "--short", "HEAD"),
            "git_dirty": dirty,
            "git_dirty_files": _git("status", "--porcelain").splitlines()[:50],
            "src_root": src_root,
            "src_py_count": len(hashes),
            "src_files_sha256": hashes,
            "src_tree_sha256": hjoin.hexdigest(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": __import__("torch").__version__,
            "argv": " ".join(sys.argv[:40]),
        }
        if extra:
            man.update(extra)
        os.makedirs(experiment_dir, exist_ok=True)
        path = os.path.join(experiment_dir, "source_manifest.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(man, f, ensure_ascii=False, indent=2, sort_keys=True)
        if verbose:
            print(f"[source-manifest] {len(hashes)} 个 .py tree_sha256={hjoin.hexdigest()[:16]} "
                  f"commit={man['git_short']}" + (" (dirty ✗)" if dirty else " (clean ✓)"),
                  flush=True)
        return man
    except Exception as e:                                         # noqa: BLE001
        if verbose:
            print(f"[source-manifest] ⚠ 生成失败(不致命): {type(e).__name__}: {e}", flush=True)
        return None
