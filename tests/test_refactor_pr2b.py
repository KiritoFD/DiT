# -*- coding: utf-8 -*-
"""test_refactor_pr2b.py —— 2026-10-14 [重构 PR2] **LatentContract** 的验收测试 ✓

PR2 完成标准: "数据覆盖 100%，**无静默 fallback**" ✗

核查发现的旧缺陷 ✗: `_check_shard_names` 只在 `got is not None` 时比对内容 ✓
  ⇒ **shard 里完全没有该 img_id** 时**静默通过** ✗✗
  ⇒ 要到训练中 `_preload_all` / `_get_latent` 才 KeyError（**中途崩溃** ✗）
     或（关 preload 时）在某个 step 才炸 ✗

本组测试锁定 `audit_coverage()` / `latent_contract()`:
  ① 100% 覆盖 ⇒ 通过 ✓（且如实报告每族命中数 ✓）
  ② 缺行 ⇒ **默认 ValueError** ✗（训练开始前就失败 ✓）；strict=False 只大声报告 ✓
  ③ 契约身份: 行序/映射变了 ⇒ rowmap_sha256 必变 ✗
"""
from __future__ import annotations

import csv as _csv
import io
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.latent_dataset import MCCDLatentDataset as LatentDataset                       # noqa: E402

C, H, W = 4, 8, 8


def _write_shards(d, ids, names=None, n_shard=2, key="shard"):
    os.makedirs(d, exist_ok=True)
    ids = list(ids)
    per = max(1, (len(ids) + n_shard - 1) // n_shard)
    for k in range(0, len(ids), per):
        chunk = ids[k:k + per]
        arr = np.random.RandomState(len(chunk)).randn(
            len(chunk), C, H, W).astype(np.float32)
        payload = {"latents": arr, "img_ids": np.asarray(chunk, dtype=np.int64)}
        if names is not None:
            payload["names"] = np.asarray([names[i] for i in chunk], dtype=object)
        np.savez(os.path.join(d, "%s_%05d.npz" % (key, k)), **payload)
    return d


def _write_csv(p, ids, img_root):
    os.makedirs(img_root, exist_ok=True)
    with io.open(p, "w", encoding="utf-8", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["img_id", "image_path", "calligrapher", "script", "character"])
        for i in ids:
            w.writerow([i, "imgs/%d.png" % i, "A", "楷", "永"])
    return p


def _mk(tmp_path, ids=(10, 11, 12, 13), shard_ids=None):
    d = str(tmp_path)
    root = os.path.join(d, "imgs")
    csvp = _write_csv(os.path.join(d, "t.csv"), ids, root)
    sd = _write_shards(os.path.join(d, "shards"), list(shard_ids or ids))
    return csvp, sd, root


def test_full_coverage_passes(tmp_path):
    csvp, sd, root = _mk(tmp_path)
    ds = LatentDataset(csvp, sd, root, load_image=False, preload=False)
    rep = ds.audit_coverage()
    assert rep["ok"] is True and rep["n_samples"] == 4
    fam = rep["families"]["target_latent"]
    assert fam["n_hit"] == 4 and fam["n_missing"] == 0
    assert len(ds) == 4


def test_missing_rows_raise_by_default(tmp_path):
    """★ PR2 核心 ✓: 缺行必须在**构造期**报错 ✗（旧实现静默通过 ✗✗）"""
    csvp, sd, root = _mk(tmp_path, ids=(10, 11, 12, 13), shard_ids=(10, 11))
    with pytest.raises(ValueError, match="数据覆盖不足"):
        LatentDataset(csvp, sd, root, load_image=False, preload=False)


def test_missing_rows_reported_when_not_strict(tmp_path):
    csvp, sd, root = _mk(tmp_path, ids=(10, 11, 12, 13), shard_ids=(10, 11))
    ds = LatentDataset(csvp, sd, root, load_image=False, preload=False,
                       strict_coverage=False)
    rep = ds.audit_coverage(strict=False, verbose=False)
    assert rep["ok"] is False
    fam = rep["families"]["target_latent"]
    assert fam["n_missing"] == 2 and fam["examples"] == [12, 13], fam


def test_missing_examples_are_actionable(tmp_path):
    csvp, sd, root = _mk(tmp_path, ids=tuple(range(100, 110)), shard_ids=tuple(range(100, 107)))
    with pytest.raises(ValueError) as ei:
        LatentDataset(csvp, sd, root, load_image=False, preload=False)
    msg = str(ei.value)
    assert "target_latent 缺 3 行" in msg, msg
    assert "107" in msg and "108" in msg and "109" in msg, msg


# ─────────────── ③ 契约身份（供 manifest 记录 ✓ P1-7 ✓）───────────────
def test_latent_contract_identity(tmp_path):
    csvp, sd, root = _mk(tmp_path)
    ds = LatentDataset(csvp, sd, root, load_image=False, preload=False)
    ct = ds.latent_contract()
    assert ct["n_samples"] == 4
    fam = ct["families"]["target_latent"]
    assert fam["n_ids"] == 4 and fam["n_shards"] >= 1
    assert len(fam["rowmap_sha256"]) == 64
    assert all(v > 0 for v in fam["shard_sizes"].values()), "必须记录 shard 文件大小 ✓"


def test_latent_contract_changes_when_rowmap_changes(tmp_path):
    """★ 行序/映射变了 ⇒ 身份必变 ✗（防"换了 shard 目录却以为同一份数据" ✗）"""
    csvp, sd, root = _mk(tmp_path)
    h1 = LatentDataset(csvp, sd, root, load_image=False,
                       preload=False).latent_contract()["families"]["target_latent"]
    # 换一个 shard 目录（行序不同 ⇒ row 号不同 ✓）
    sd2 = _write_shards(os.path.join(str(tmp_path), "shards2"),
                        [10, 11, 12, 13], n_shard=1)
    h2 = LatentDataset(csvp, sd2, root, load_image=False,
                       preload=False).latent_contract()["families"]["target_latent"]
    assert h1["rowmap_sha256"] != h2["rowmap_sha256"], \
        "行映射变 ⇒ 身份必须变 ✗"


def test_contract_covers_multiple_families(tmp_path):
    """多族（target + skel 变体）都要被审计 ✓"""
    csvp, sd, root = _mk(tmp_path)
    sk = _write_shards(os.path.join(str(tmp_path), "skel"), [10, 11, 12, 13])
    ds = LatentDataset(csvp, sd, root, load_image=False, preload=False,
                       skel_latent_shards_dirs=[sk])
    rep = ds.audit_coverage(verbose=False)
    assert "target_latent" in rep["families"]
    assert "skel_latent_v0" in rep["families"], rep["families"].keys()
    assert rep["ok"] is True


def test_contract_flags_missing_skel_family(tmp_path):
    """target 全覆盖但 **skel 缺行** ⇒ 也必须报错 ✗（这正是 P0-2 的"全 0"那条路 ✓）"""
    csvp, sd, root = _mk(tmp_path)
    sk = _write_shards(os.path.join(str(tmp_path), "skel"), [10, 11])
    with pytest.raises(ValueError, match="skel_latent_v0"):
        LatentDataset(csvp, sd, root, load_image=False, preload=False,
                      skel_latent_shards_dirs=[sk])
