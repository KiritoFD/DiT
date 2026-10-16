# -*- coding: utf-8 -*-
"""
ramdisk_cache.py — Manage high-speed training cache in Linux ramdisk (/dev/shm).

Features:
  1. Auto-discover training caches (DINO cache, STD shape bank, latent shards).
  2. Safe copy to /dev/shm with pre-flight capacity check and progress reporting.
  3. Seamless symlink swapping: swap workspace paths to point to /dev/shm
     with zero changes needed in training configs (preserves .disk_bak).
  4. Safe rollback / restore on unload.
  5. Python API + CLI entry points.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Dict, List, Optional, Tuple

DEFAULT_TARGETS = [
    "data/dino_cache/top10_aug_v1",
    "assets/std_shape_bank_64.pt",
    "exp-std/data/shards_img_aug",
]

DEFAULT_RAMDISK_ROOT = Path("/dev/shm/dit_ramdisk") if sys.platform != "win32" else Path(os.environ.get("TEMP", "/tmp")) / "dit_ramdisk"


def get_dir_size(path: Path) -> int:
    """Calculate total byte size of a file or directory tree."""
    if not path.exists() and not path.is_symlink():
        return 0
    if path.is_file() or path.is_symlink():
        try:
            return path.stat().st_size
        except OSError:
            return 0
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            fp = Path(root) / f
            try:
                if not fp.is_symlink():
                    total += fp.stat().st_size
            except OSError:
                pass
    return total


def format_bytes(size: int) -> str:
    for unit in ["B", "KiB", "MiB", "GiB", "TiB"]:
        if abs(size) < 1024.0:
            return f"{size:3.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} PiB"


def get_shm_space(ramdisk_root: Path) -> Tuple[int, int, int]:
    """Return (total, used, free) bytes for the filesystem containing ramdisk_root."""
    parent = ramdisk_root
    while not parent.exists() and parent != parent.parent:
        parent = parent.parent
    usage = shutil.disk_usage(parent)
    return usage.total, usage.used, usage.free


def copy_with_progress(src: Path, dst: Path, buffer_size: int = 64 * 1024 * 1024) -> None:
    """Fast file copy with 64MB buffer and speed reporting for large files."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    size = src.stat().st_size
    t0 = time.time()
    
    # For small files, standard copy2 is fast
    if size < 32 * 1024 * 1024:
        shutil.copy2(src, dst)
        return

    print(f"  [copying] {src.name} ({format_bytes(size)})...", flush=True)
    copied = 0
    with open(src, "rb") as fsrc, open(dst, "wb") as fdst:
        while True:
            buf = fsrc.read(buffer_size)
            if not buf:
                break
            fdst.write(buf)
            copied += len(buf)
            elapsed = max(time.time() - t0, 1e-4)
            rate = copied / (1024 * 1024 * elapsed)
            percent = (copied / size) * 100
            sys.stdout.write(f"\r  [copying] {percent:5.1f}% | {rate:6.1f} MB/s | {format_bytes(copied)}/{format_bytes(size)}")
            sys.stdout.flush()
    shutil.copystat(src, dst)
    elapsed = max(time.time() - t0, 1e-4)
    avg_rate = size / (1024 * 1024 * elapsed)
    print(f"\r  [copied]  {src.name} finished in {elapsed:.1f}s ({avg_rate:.1f} MB/s) ✓", flush=True)


def sync_path_to_ramdisk(src: Path, dst: Path, force: bool = False) -> None:
    """Recursively sync a file or directory to ramdisk."""
    if src.is_file():
        if dst.exists() and not force:
            if dst.stat().st_size == src.stat().st_size and dst.stat().st_mtime >= src.stat().st_mtime:
                print(f"  [cached] {src.name} already in ramdisk and matches (size={format_bytes(dst.stat().st_size)})")
                return
        copy_with_progress(src, dst)
    elif src.is_dir():
        dst.mkdir(parents=True, exist_ok=True)
        for item in src.iterdir():
            if item.name.endswith(".disk_bak"):
                continue
            sync_path_to_ramdisk(item, dst / item.name, force=force)


def swap_symlink(source: Path, target_in_ramdisk: Path) -> bool:
    """
    Atomically swap source to a symlink pointing to target_in_ramdisk.
    Preserves original by moving it to <source>.disk_bak.
    """
    bak = source.with_name(source.name + ".disk_bak")
    if source.is_symlink():
        current_target = Path(os.readlink(source)).resolve()
        if current_target == target_in_ramdisk.resolve():
            return True  # Already correctly linked
        source.unlink()
    elif source.exists():
        if bak.exists():
            if bak.is_dir():
                shutil.rmtree(bak)
            else:
                bak.unlink()
        source.rename(bak)

    try:
        source.symlink_to(target_in_ramdisk)
        return True
    except OSError as e:
        print(f"  [error] Failed to create symlink {source} -> {target_in_ramdisk}: {e}", file=sys.stderr)
        if bak.exists() and not source.exists():
            bak.rename(source)
        return False


def restore_symlink(source: Path) -> bool:
    """Restore original file/directory from .disk_bak if source is a symlink."""
    bak = source.with_name(source.name + ".disk_bak")
    if source.is_symlink():
        source.unlink()
        if bak.exists():
            bak.rename(source)
            print(f"  [restored] {source.name} from backup {bak.name}")
            return True
        print(f"  [unlinked] {source.name} (no backup existed)")
        return True
    elif bak.exists():
        if not source.exists():
            bak.rename(source)
            print(f"  [restored] {source.name} from backup {bak.name}")
            return True
    return False


class RamdiskCacheManager:
    def __init__(self, workspace_root: Optional[Path] = None, ramdisk_root: Optional[Path] = None):
        self.workspace_root = (workspace_root or Path.cwd()).resolve()
        self.ramdisk_root = (ramdisk_root or DEFAULT_RAMDISK_ROOT).resolve()

    def get_status(self, targets: Optional[List[str]] = None) -> Dict[str, dict]:
        targets = targets or DEFAULT_TARGETS
        total, used, free = get_shm_space(self.ramdisk_root)
        status = {
            "_ramdisk": {
                "root": str(self.ramdisk_root),
                "total": total,
                "used": used,
                "free": free,
                "total_str": format_bytes(total),
                "used_str": format_bytes(used),
                "free_str": format_bytes(free),
            },
            "targets": {}
        }
        for rel in targets:
            ws_path = self.workspace_root / rel
            bak_path = ws_path.with_name(ws_path.name + ".disk_bak")
            rd_path = self.ramdisk_root / rel
            
            is_symlink = ws_path.is_symlink()
            link_target = str(os.readlink(ws_path)) if is_symlink else None
            
            status["targets"][rel] = {
                "workspace_path": str(ws_path),
                "workspace_exists": ws_path.exists() or is_symlink,
                "is_symlink": is_symlink,
                "link_target": link_target,
                "has_backup": bak_path.exists(),
                "workspace_size": get_dir_size(ws_path),
                "workspace_size_str": format_bytes(get_dir_size(ws_path)),
                "ramdisk_path": str(rd_path),
                "ramdisk_exists": rd_path.exists(),
                "ramdisk_size": get_dir_size(rd_path),
                "ramdisk_size_str": format_bytes(get_dir_size(rd_path)),
            }
        return status

    def sync(self, targets: Optional[List[str]] = None, link: bool = True, force: bool = False) -> bool:
        targets = targets or DEFAULT_TARGETS
        total, used, free = get_shm_space(self.ramdisk_root)
        print(f"=== Syncing to Ramdisk: {self.ramdisk_root} ===")
        print(f"Ramdisk capacity: {format_bytes(total)} | Used: {format_bytes(used)} | Free: {format_bytes(free)}")
        self.ramdisk_root.mkdir(parents=True, exist_ok=True)

        # 1. Calculate required size
        needed_bytes = 0
        resolved_sources = []
        for rel in targets:
            ws_path = self.workspace_root / rel
            bak_path = ws_path.with_name(ws_path.name + ".disk_bak")
            # If already symlinked, actual source might be bak or already in ramdisk
            if ws_path.is_symlink():
                src_path = bak_path if bak_path.exists() else Path(os.readlink(ws_path)).resolve()
            elif ws_path.exists():
                src_path = ws_path
            elif bak_path.exists():
                src_path = bak_path
            else:
                print(f"[warning] Target {rel} not found on disk, skipping.")
                continue
            sz = get_dir_size(src_path)
            needed_bytes += sz
            resolved_sources.append((rel, src_path, sz))

        print(f"Total cache size to sync: {format_bytes(needed_bytes)}")
        if needed_bytes > free:
            raise RuntimeError(f"Insufficient space in ramdisk {self.ramdisk_root}: needed {format_bytes(needed_bytes)}, free {format_bytes(free)}")

        # 2. Sync each target
        success = True
        for rel, src_path, sz in resolved_sources:
            rd_path = self.ramdisk_root / rel
            print(f"\nProcessing [{rel}] -> {rd_path} ({format_bytes(sz)})")
            sync_path_to_ramdisk(src_path, rd_path, force=force)
            
            if link:
                ws_path = self.workspace_root / rel
                ok = swap_symlink(ws_path, rd_path)
                if ok:
                    print(f"  [linked]  {ws_path} -> {rd_path} ✓")
                else:
                    success = False
        return success

    def unload(self, targets: Optional[List[str]] = None, remove_data: bool = True) -> bool:
        targets = targets or DEFAULT_TARGETS
        print(f"=== Unloading Ramdisk Caches from {self.ramdisk_root} ===")
        for rel in targets:
            ws_path = self.workspace_root / rel
            restore_symlink(ws_path)
            rd_path = self.ramdisk_root / rel
            if remove_data and rd_path.exists():
                if rd_path.is_dir():
                    shutil.rmtree(rd_path)
                else:
                    rd_path.unlink()
                print(f"  [deleted] {rd_path} from ramdisk")
        return True


def ensure_ramdisk_cache(
    targets: Optional[List[str]] = None,
    link: bool = True,
    ramdisk_root: Optional[Path] = None,
    workspace_root: Optional[Path] = None
) -> RamdiskCacheManager:
    """Helper for launcher scripts to ensure caches are present in ramdisk."""
    mgr = RamdiskCacheManager(workspace_root=workspace_root, ramdisk_root=ramdisk_root)
    mgr.sync(targets=targets, link=link, force=False)
    return mgr


def main():
    parser = argparse.ArgumentParser(description="Manage DiT training caches on Linux ramdisk (/dev/shm).")
    parser.add_argument("action", choices=["status", "sync", "load", "unload", "clean"], help="Operation to perform")
    parser.add_argument("--targets", nargs="*", default=None, help="Specific cache targets (defaults to all known caches)")
    parser.add_argument("--workspace", type=Path, default=None, help="Workspace root directory")
    parser.add_argument("--ramdisk", type=Path, default=None, help="Ramdisk root directory (defaults to /dev/shm/dit_ramdisk)")
    parser.add_argument("--no-link", action="store_true", help="Do not create symlinks in workspace")
    parser.add_argument("--force", action="store_true", help="Force re-copying even if size matches")
    parser.add_argument("--keep-data", action="store_true", help="When unloading, keep files in ramdisk")
    args = parser.parse_args()

    mgr = RamdiskCacheManager(workspace_root=args.workspace, ramdisk_root=args.ramdisk)

    if args.action == "status":
        st = mgr.get_status(targets=args.targets)
        rd = st["_ramdisk"]
        print(f"Ramdisk: {rd['root']} | Total: {rd['total_str']} | Used: {rd['used_str']} | Free: {rd['free_str']}")
        print("-" * 80)
        for rel, info in st["targets"].items():
            sym_desc = f" -> {info['link_target']}" if info['is_symlink'] else ""
            print(f"Target: {rel}")
            print(f"  Workspace: {'EXISTS' if info['workspace_exists'] else 'MISSING'} ({info['workspace_size_str']}) {'[SYMLINK]' if info['is_symlink'] else ''}{sym_desc}")
            print(f"  Backup:    {'EXISTS' if info['has_backup'] else 'NONE'}")
            print(f"  Ramdisk:   {'EXISTS' if info['ramdisk_exists'] else 'MISSING'} ({info['ramdisk_size_str']})")
    elif args.action in ("sync", "load"):
        mgr.sync(targets=args.targets, link=not args.no_link, force=args.force)
    elif args.action in ("unload", "clean"):
        mgr.unload(targets=args.targets, remove_data=not args.keep_data)


if __name__ == "__main__":
    main()
