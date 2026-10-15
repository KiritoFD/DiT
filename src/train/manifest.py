# -*- coding: utf-8 -*-
"""manifest.py — 运行凭据落盘（2026-10-13 [审查 P0]；2026-10-14 [重构 P1-7] 扩展）

历史: `train.py` 原来的 source_manifest 只哈希**根目录的旧文件名** ✗（那些文件早迁到
`src/…` ⇒ 清单实际为空 ✗）⇒ "这次跑的是哪份代码"无法追溯 ✗。

★ 2026-10-14 [重构 P1-7] 审查指出两处不足 ✗:
    ① 只哈希 `src/**/*.py` ✗ —— 但 **Critic 实现是从 `tools/train_writer_brush_critic.py`
       `importlib` 动态加载的** ✗ ⇒ 仅有 src 哈希**不足以证明实验用了哪版 Critic** ✗✗
    ② 没有对**模型权重 / VAE / 字体 / 数据 manifest / 生成器 / 评测器**分别记录 SHA256 ✗
  ⇒ 本版:
    · `src_roots` 支持**多根**（默认 `("src", "tools")` ✓）⇒ Critic 也被哈希 ✓
    · 新增 `artifacts`: 名字 → 路径（文件或目录 ✓）⇒ 各自 sha256 ✓
      目录按后缀白名单哈希（默认 json/csv/txt/yaml/ttf/otf ✓ 避免把十万个 npy 全哈希 ✗）
    · **缺失的依赖显式记入 `artifacts_missing`** ✗（绝不静默省略 ✗）
    · 保留 `src_tree_sha256`（仅 src ✓ 向后兼容 ✓）+ 新增 `code_tree_sha256`（全部根 ✓）
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys

__all__ = ["write_source_manifest", "sha256_file", "sha256_tree"]

# 目录哈希的后缀白名单 ✓（避免把巨大的 npy/npz 数据全量哈希 ✗）
DIR_EXTS = (".json", ".csv", ".txt", ".yaml", ".yml", ".ttf", ".otf", ".ttc", ".otc")


def sha256_file(path: str):
    """单文件 sha256 ✓；缺失/不可读 ⇒ **None**（调用方须显式记为 missing ✗ 不得当 0 ✓）。"""
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def sha256_tree(root: str, exts=DIR_EXTS, max_files: int = 20000):
    """目录指纹 ✓: 按相对路径排序逐个哈希 ⇒ {rel: sha256} + 汇总 ✓。

    `exts=None` 表示**不限后缀** ✓（谨慎使用 ✗ 大目录会很慢 ✗）。
    超过 `max_files` ⇒ 返回 (None, count) 并标记截断 ✗（不假装完整 ✗）。
    """
    if not os.path.isdir(root):
        return None, 0, False
    files = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d != "__pycache__")
        for fn in sorted(filenames):
            if exts and not fn.endswith(tuple(exts)):
                continue
            files.append(os.path.join(dirpath, fn))
    files.sort()
    if len(files) > max_files:
        return None, len(files), True
    per, hjoin = {}, hashlib.sha256()
    for p in files:
        h = sha256_file(p)
        if h is None:
            continue
        rel = os.path.relpath(p, root).replace("\\", "/")
        per[rel] = h
        hjoin.update(rel.encode() + b"|" + h.encode() + b"\n")
    return {"files": per, "n": len(per), "tree_sha256": hjoin.hexdigest()}, len(files), False


def _git(*a):
    try:
        r = subprocess.run(["git", *a], capture_output=True, text=True, timeout=20)
        return (r.stdout or "").strip()
    except Exception:                                              # noqa: BLE001
        return ""


def _hash_code_roots(roots):
    """多根代码哈希 ✓（src + tools ✓ ⇒ 动态加载的 Critic 也被覆盖 ✓）。"""
    per_root, all_files = {}, []
    for root in roots:
        if not os.path.isdir(root):
            per_root[root] = {"files": {}, "n": 0, "missing": True}
            continue
        files = []
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d != "__pycache__"]
            files += [os.path.join(dirpath, f).replace("\\", "/")
                      for f in filenames if f.endswith(".py")]
        files.sort()
        per, hjoin = {}, hashlib.sha256()
        for p in files:
            b = open(p, "rb").read()
            h = hashlib.sha256(b).hexdigest()
            per[p] = h
            hjoin.update(p.encode() + b"|" + h.encode() + b"\n")
        per_root[root] = {"files": per, "n": len(per), "tree_sha256": hjoin.hexdigest()}
        all_files += files
    hall = hashlib.sha256()
    for p in sorted(all_files):
        hall.update(p.encode() + b"|" + per_root[
            next(r for r in roots if p.startswith(r))]["files"][p].encode() + b"\n")
    return per_root, all_files, hall.hexdigest()


def _hash_artifacts(artifacts):
    """依赖身份 ✓: 名字 → 路径（文件/目录）⇒ sha256；缺失**显式记录** ✗。"""
    out, missing = {}, []
    for name, p in (artifacts or {}).items():
        if not p:
            missing.append(name)
            continue
        p = str(p).replace("\\", "/")
        if os.path.isdir(p):
            t, n_seen, trunc = sha256_tree(p)
            if t is None:
                out[name] = {"kind": "dir", "path": p, "truncated": bool(trunc),
                             "n_files_seen": n_seen}
            else:
                out[name] = {"kind": "dir", "path": p, "n": t["n"],
                             "tree_sha256": t["tree_sha256"], "files": t["files"]}
        elif os.path.isfile(p):
            h = sha256_file(p)
            if h is None:
                missing.append(name)
            else:
                out[name] = {"kind": "file", "path": p,
                             "size": os.path.getsize(p), "sha256": h}
        else:
            missing.append(name)
    return out, missing


def write_source_manifest(experiment_dir, extra=None, src_root=None,
                          src_roots=("src", "tools"), artifacts=None, verbose=True):
    """把"这次跑的是哪份代码 + 哪些依赖"写成 source_manifest.json ✓（失败不致命 ✓）

    ★ 默认 `src_roots=("src", "tools")` ✓ —— **必须包含 tools/** ✗:
        实验的 Critic 实现是从 `tools/train_writer_brush_critic.py` `importlib`
        动态加载的 ✗ ⇒ 只哈希 `src/` 无法证明"用了哪版 Critic" ✗✗（审查 P1-7 ✗）。
        `tools/` 下约 1.3k 个 .py ✓ 全量哈希约 1–3 秒 ✓（仅启动时一次 ✓ 不进训练循环 ✓）；
        作为**安全超集**一并纳入 ✓（legacy/archive 也在内 ✓ 无副作用 ✓）。

    `src_root` 仅为向后兼容保留 ✓（等价于 `src_roots=(src_root,)` ✓）。
    `artifacts`: 名字 → 路径（文件/目录 ✓）⇒ 各自 SHA256 ✓；缺失**显式记录** ✗。
    """
    try:
        roots = (src_root,) if src_root else tuple(src_roots)
        per_root, files, code_tree = _hash_code_roots(roots)
        src_only = per_root.get("src", {}).get("tree_sha256", "")
        art, art_missing = _hash_artifacts(artifacts)
        dirty = bool(_git("status", "--porcelain"))
        man = {
            "git_commit": _git("rev-parse", "HEAD"),
            "git_short": _git("rev-parse", "--short", "HEAD"),
            "git_dirty": dirty,
            "git_dirty_files": _git("status", "--porcelain").splitlines()[:50],
            # ── 代码身份（多根 ✓）──
            "code_roots": list(roots),
            "code_py_count": len(files),
            "code_tree_sha256": code_tree,
            "code_roots_detail": per_root,
            # ── 向后兼容（旧字段语义不变: 仅 src/ ✓）──
            "src_root": "src",
            "src_py_count": per_root.get("src", {}).get("n", 0),
            "src_files_sha256": per_root.get("src", {}).get("files", {}),
            "src_tree_sha256": src_only,
            # ── 依赖身份（权重/VAE/字体/数据 manifest/生成器/评测器 ✓）──
            "artifacts": art,
            "artifacts_missing": sorted(set(art_missing)),
            # ── 环境 ──
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
            print(f"[source-manifest] 代码 {len(files)} 个 .py / {len(roots)} 根 "
                  f"code_tree={code_tree[:16]} | 依赖 {len(art)} 项"
                  + (f" **缺失 {len(art_missing)}**: {sorted(art_missing)[:4]}" if art_missing else "")
                  + f" | commit={man['git_short']}"
                  + (" (dirty ✗)" if dirty else " (clean ✓)"), flush=True)
        return man
    except Exception as e:                                         # noqa: BLE001
        if verbose:
            print(f"[source-manifest] ⚠ 生成失败(不致命): {type(e).__name__}: {e}", flush=True)
        return None
