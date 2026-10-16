# -*- coding: utf-8 -*-
"""Unit tests for ramdisk cache manager."""

import os
from pathlib import Path
import pytest
import sys

from tools.infra.ramdisk_cache import (
    RamdiskCacheManager,
    ensure_ramdisk_cache,
    format_bytes,
    get_dir_size,
    swap_symlink,
    restore_symlink,
)


def test_format_bytes():
    assert "100.00 B" in format_bytes(100)
    assert "1.00 KiB" in format_bytes(1024)
    assert "1.50 MiB" in format_bytes(int(1.5 * 1024 * 1024))
    assert "2.00 GiB" in format_bytes(2 * 1024 * 1024 * 1024)


def test_get_dir_size(tmp_path):
    f1 = tmp_path / "a.bin"
    f1.write_bytes(b"x" * 100)
    d = tmp_path / "sub"
    d.mkdir()
    f2 = d / "b.bin"
    f2.write_bytes(b"y" * 250)

    assert get_dir_size(f1) == 100
    assert get_dir_size(d) == 250
    assert get_dir_size(tmp_path) == 350


def test_ramdisk_cache_lifecycle(tmp_path):
    ws = tmp_path / "workspace"
    ws.mkdir()
    rd = tmp_path / "ramdisk"
    rd.mkdir()

    # Create dummy cache files and directories in workspace
    cache_file = ws / "assets" / "std_shape_bank.pt"
    cache_file.parent.mkdir(parents=True)
    cache_file.write_bytes(b"HELLO_BANK_CACHE")

    cache_dir = ws / "data" / "dino_cache"
    cache_dir.mkdir(parents=True)
    (cache_dir / "feats.f16").write_bytes(b"FEATS_123456789")
    (cache_dir / "ids.npy").write_bytes(b"IDS_987654321")

    targets = [
        "assets/std_shape_bank.pt",
        "data/dino_cache",
    ]

    mgr = RamdiskCacheManager(workspace_root=ws, ramdisk_root=rd)

    # 1. Initial status before sync
    st = mgr.get_status(targets=targets)
    assert st["targets"]["assets/std_shape_bank.pt"]["workspace_exists"]
    assert not st["targets"]["assets/std_shape_bank.pt"]["is_symlink"]
    assert not st["targets"]["assets/std_shape_bank.pt"]["ramdisk_exists"]

    # 2. Sync to ramdisk with link=True (skip link test if OS forbids symlinks)
    can_symlink = True
    try:
        test_link = tmp_path / "test_link"
        test_link.symlink_to(f1 := tmp_path / "test_target")
        test_link.unlink()
    except OSError:
        can_symlink = False

    mgr.sync(targets=targets, link=can_symlink, force=False)

    # Verify ramdisk copies exist
    assert (rd / "assets" / "std_shape_bank.pt").read_bytes() == b"HELLO_BANK_CACHE"
    assert (rd / "data" / "dino_cache" / "feats.f16").read_bytes() == b"FEATS_123456789"

    if can_symlink:
        assert cache_file.is_symlink()
        assert cache_dir.is_symlink()
        assert (cache_file.with_name(cache_file.name + ".disk_bak")).exists()

        # Workspace path should still be readable transparently
        assert cache_file.read_bytes() == b"HELLO_BANK_CACHE"
        assert (cache_dir / "feats.f16").read_bytes() == b"FEATS_123456789"

        # 3. Unload and restore
        mgr.unload(targets=targets, remove_data=True)
        assert not cache_file.is_symlink()
        assert not cache_dir.is_symlink()
        assert cache_file.exists()
        assert cache_file.read_bytes() == b"HELLO_BANK_CACHE"
        assert not (rd / "assets" / "std_shape_bank.pt").exists()
        assert not (rd / "data" / "dino_cache").exists()
