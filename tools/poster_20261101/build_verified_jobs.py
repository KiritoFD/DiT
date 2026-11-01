"""Build strictly identity-verified evaluation jobs from stdskel batches.

Only scans assets/ experiments and the explicit exp-std/v54 history tree.
Each (source batch, step, set) group gets its own verified CSV and image panel.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Iterable

IDENTITY = ("img_id", "character", "script", "calligrapher")
BATCH_CHARACTER = ("character", "char")
REQUIRED = ("step", "set", "idx", "img_id", "script", "calligrapher")
CANONICAL_RELATIVE = (
    "assets/u100_pretrain_20261024/validation_val100.csv",
    "exp-std/csv/eval200_fixed.csv",
    "assets/style_focus_85k_20261029/4090/protocols/S_187_seed{seed}.csv",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader.fieldnames), list(reader)


def identity_sequence(rows: Iterable[dict[str, str]]) -> tuple[tuple[str, ...], ...]:
    result = []
    for row in rows:
        char = row.get("character") or row.get("char") or ""
        result.append(tuple((char if field == "character" else row.get(field) or "").strip()
                            for field in IDENTITY))
    return tuple(result)


def step_number(raw: str) -> int:
    value = int(raw)
    if value < 0:
        raise ValueError(f"negative step: {raw}")
    return value


def group_batch(batch: Path) -> dict[tuple[int, str], list[dict[str, str]]]:
    fields, rows = read_rows(batch)
    if not set(("img_id", "script", "calligrapher")).issubset(fields) or not (set(BATCH_CHARACTER) & set(fields)):
        raise ValueError(f"missing batch identity columns: {fields}")
    grouping = set(("step", "set", "idx"))
    missing = grouping - set(fields)
    if missing:
        raise ValueError(f"missing batch grouping columns {sorted(missing)}")
    groups: dict[tuple[int, str], list[dict[str, str]]] = {}
    for row in rows:
        if not (row.get("character") or row.get("char") or "").strip():
            raise ValueError(f"blank character identity in {batch}")
        if any(not (row.get(field) or "").strip() for field in REQUIRED):
            raise ValueError(f"blank grouping/identity value in {batch}")
        key = (step_number(row["step"]), row["set"].strip())
        groups.setdefault(key, []).append(row)
    for rows_for_group in groups.values():
        rows_for_group.sort(key=lambda row: int(row["idx"]))
        indices = [int(row["idx"]) for row in rows_for_group]
        if indices != list(range(len(indices))):
            raise ValueError(f"idx must be contiguous from zero: {batch} {indices[:8]}")
    # Regression guard for the original parent-directory step bug and cross-step merge.
    assert all(step != 0 or any(int(row["step"]) == 0 for row in values)
               for (step, _), values in groups.items())
    assert len(groups) == len({(int(row["step"]), row["set"]) for row in rows})
    return groups


def request_csvs(batch: Path, root: Path) -> list[Path]:
    candidates = []
    for name in ("request.json", "request/complete.json"):
        metadata = batch.parent / name
        if not metadata.is_file():
            continue
        try:
            data = json.loads(metadata.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        value = next((data[key] for key in ("csv", "csv_path", "protocol_csv")
                      if isinstance(data.get(key), str) and data[key].strip()), None)
        if not value:
            continue
        p = Path(value)
        if not p.is_absolute():
            root_candidate = root / p
            p = root_candidate if root_candidate.is_file() else batch.parent / p
        recorded = next((data[key] for key in ("csv_sha256", "csv_hash")
                         if isinstance(data.get(key), str) and data[key].strip()), None)
        if recorded and p.is_file() and sha256(p) != recorded:
            raise ValueError(f"request CSV SHA mismatch: {metadata} -> {p}")
        candidates.append(p)
    return candidates


def candidate_csvs(root: Path, batch: Path, rows: list[dict[str, str]]) -> list[Path]:
    found = []
    for relative in CANONICAL_RELATIVE:
        seed_match = re.search(r"seed(\d+)", batch.parent.as_posix())
        rel = relative.format(seed=seed_match.group(1) if seed_match else "")
        p = root / rel
        if "{seed}" not in relative or seed_match:
            found.append(p)
    found.extend(request_csvs(batch, root))
    # Only explicit protocol locations from source metadata and canonical roots.
    unique = []
    seen = set()
    expected = identity_sequence(rows)
    for path in found:
        resolved = path.resolve()
        if resolved in seen or not resolved.is_file():
            continue
        seen.add(resolved)
        try:
            fields, csv_rows = read_rows(resolved)
            has_character = "character" in fields or "char" in fields
            required = {"img_id", "script", "calligrapher"}.issubset(fields)
            if has_character and required and identity_sequence(csv_rows) == expected:
                unique.append(resolved)
        except (OSError, UnicodeError, csv.Error, ValueError):
            continue
    return unique


def write_identity_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=IDENTITY, lineterminator="\n")
        writer.writeheader()
        writer.writerows({"img_id": row["img_id"],
                          "character": row.get("character") or row.get("char") or "",
                          "script": row["script"],
                          "calligrapher": row["calligrapher"]} for row in rows)


def panel_for(batch: Path, step: int) -> Path:
    controls = batch.parent / "eval_samples_ctrl"
    matches = [p for p in controls.glob("step*")
               if p.is_dir() and p.name.startswith("step")
               and p.name[4:].isdigit() and int(p.name[4:]) == step]
    if len(matches) != 1:
        raise ValueError(f"expected one panel directory for step {step}; found {len(matches)}")
    panel = matches[0] / "eval200fix"
    if not panel.is_dir():
        raise ValueError(f"missing panel: {panel}")
    return panel


def source_batches(root: Path) -> list[Path]:
    roots = [root / "assets", root / "exp-std" / "v54"]
    return sorted({p for base in roots if base.is_dir()
                   for p in base.rglob("eval_stdskel_batch.csv")})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path,
                        default=Path("assets/metrics_index/verified_jobs.json"))
    parser.add_argument("--protocol-dir", type=Path,
                        default=Path("assets/metrics_index/verified_protocols"))
    args = parser.parse_args()
    root = args.root.resolve()
    out = args.out if args.out.is_absolute() else root / args.out
    protocol_dir = args.protocol_dir if args.protocol_dir.is_absolute() else root / args.protocol_dir

    jobs, manifest, rejected = [], [], []
    for batch in source_batches(root):
        try:
            groups = group_batch(batch)
        except (OSError, UnicodeError, csv.Error, ValueError) as exc:
            rejected.append({"source_batch": str(batch), "reason": str(exc)})
            continue
        for (step, set_name), rows in sorted(groups.items()):
            group_id = hashlib.sha256(
                f"{batch.resolve()}\0{step}\0{set_name}".encode("utf-8")
            ).hexdigest()[:16]
            arm = batch.parent.name
            protocol = f"{set_name}-{group_id}"
            try:
                panel = panel_for(batch, step)
                missing = [i for i in range(len(rows))
                           if not (panel / f"g{i}.png").is_file()
                           or not (panel / f"gt{i}.png").is_file()]
                if missing:
                    raise ValueError(f"missing generated/GT PNGs for {len(missing)}/{len(rows)} rows")
                candidates = candidate_csvs(root, batch, rows)
                if candidates:
                    csv_path = candidates[0]
                else:
                    csv_path = protocol_dir / f"{group_id}.csv"
                    write_identity_csv(csv_path, rows)
                    # Read back and compare exact ordered identity before accepting.
                    _, exported = read_rows(csv_path)
                    if identity_sequence(exported) != identity_sequence(rows):
                        raise ValueError("exported panel identity CSV failed round-trip verification")
                job = {"arm": arm, "step": step, "protocol": protocol,
                       "samples": str(panel), "csv": str(csv_path),
                       "csv_sha256": sha256(csv_path), "n": len(rows)}
                jobs.append(job)
                manifest.append({**job, "source_batch": str(batch), "set": set_name,
                                 "group_id": group_id})
            except (OSError, UnicodeError, csv.Error, ValueError) as exc:
                rejected.append({"source_batch": str(batch), "step": step,
                                 "set": set_name, "reason": str(exc)})

    # Duplicate human-facing identities are a hard error, never a silent drop.
    identities = [(j["arm"], j["step"], j["protocol"]) for j in jobs]
    if len(identities) != len(set(identities)):
        raise RuntimeError("non-unique job identity detected")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(jobs, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest_path = out.with_name(out.stem + ".manifest.json")
    manifest_path.write_text(json.dumps({"n_jobs": len(jobs), "jobs": manifest,
                                         "rejected": rejected},
                                        ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"verified jobs: {len(jobs)} -> {out}")
    print(f"manifest: {manifest_path}; rejected groups: {len(rejected)}")
    for item in rejected:
        print(f"REJECT {item['source_batch']}: {item['reason']}")


if __name__ == "__main__":
    main()
