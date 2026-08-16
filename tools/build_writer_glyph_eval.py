"""Analyze writer x glyph composition without modifying data by default.

The glyph factor is the atomic pair ``(script_id, character_id)`` encoded as::

    glyph_id = script_id * num_characters + character_id

Rows from an input CSV are classified relative to a training CSV as:

``seen_edge``
    The exact ``(writer, glyph)`` edge occurs in training.
``unseen_edge_seen_nodes``
    The edge is new, but both endpoint nodes occur in training.
``unseen_node``
    The writer, glyph, or both do not occur in training.

The default mode is read-only and prints a JSON report.  Files are created only
when ``--output-dir`` is explicitly supplied.  Output selection and ordering
operate on complete writer-glyph edge groups, so repeated images of one edge
are never split across selections.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence


STRATA = ("seen_edge", "unseen_edge_seen_nodes", "unseen_node")
DEFAULT_DEGREE_BINS = (1, 2, 5, 10, 20, 50, 100, 500, 1000)
ANNOTATION_FIELDS = (
    "glyph_id",
    "writer_glyph_stratum",
    "train_writer_degree",
    "train_glyph_degree",
    "train_writer_row_count",
    "train_glyph_row_count",
    "writer_degree_bin",
    "glyph_degree_bin",
    "unseen_writer",
    "unseen_glyph",
)


@dataclass(frozen=True)
class TrainIndex:
    edges: frozenset[tuple[int, int]]
    writers: frozenset[int]
    glyphs: frozenset[int]
    writer_degree: Mapping[int, int]
    glyph_degree: Mapping[int, int]
    writer_row_count: Mapping[int, int]
    glyph_row_count: Mapping[int, int]


@dataclass
class EdgeGroup:
    edge: tuple[int, int]
    script_id: int
    writer_degree_bin: str
    glyph_degree_bin: str
    first_row_index: int
    rows: list[dict[str, object]]


def read_csv(path: str | Path) -> tuple[list[str], list[dict[str, str]]]:
    with open(path, encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader.fieldnames), list(reader)


def _integer(row: Mapping[str, object], column: str) -> int:
    try:
        return int(row[column])
    except KeyError as exc:
        raise ValueError(f"missing required column {column!r}") from exc
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid integer in column {column!r}: {row.get(column)!r}") from exc


def row_ids(
    row: Mapping[str, object],
    num_characters: int,
    writer_column: str = "calligrapher_id",
    script_column: str = "script_id",
    character_column: str = "character_id",
) -> tuple[int, int, int]:
    """Return ``(writer_id, script_id, glyph_id)`` with range validation."""
    if num_characters <= 0:
        raise ValueError("num_characters must be positive")
    writer = _integer(row, writer_column)
    script = _integer(row, script_column)
    character = _integer(row, character_column)
    if writer < 0 or script < 0:
        raise ValueError(f"writer and script IDs must be non-negative: {writer}, {script}")
    if character < 0 or character >= num_characters:
        raise ValueError(
            f"character_id {character} is outside [0, {num_characters})")
    return writer, script, script * num_characters + character


def build_train_index(
    rows: Iterable[Mapping[str, object]],
    num_characters: int,
    writer_column: str = "calligrapher_id",
    script_column: str = "script_id",
    character_column: str = "character_id",
) -> TrainIndex:
    edges: set[tuple[int, int]] = set()
    writer_rows: Counter[int] = Counter()
    glyph_rows: Counter[int] = Counter()
    for row in rows:
        writer, _, glyph = row_ids(
            row, num_characters, writer_column, script_column, character_column)
        edges.add((writer, glyph))
        writer_rows[writer] += 1
        glyph_rows[glyph] += 1

    writer_degree: Counter[int] = Counter()
    glyph_degree: Counter[int] = Counter()
    for writer, glyph in edges:
        writer_degree[writer] += 1
        glyph_degree[glyph] += 1
    return TrainIndex(
        edges=frozenset(edges),
        writers=frozenset(writer_rows),
        glyphs=frozenset(glyph_rows),
        writer_degree=dict(writer_degree),
        glyph_degree=dict(glyph_degree),
        writer_row_count=dict(writer_rows),
        glyph_row_count=dict(glyph_rows),
    )


def parse_degree_bins(value: str | Sequence[int]) -> tuple[int, ...]:
    if isinstance(value, str):
        try:
            bounds = tuple(int(part.strip()) for part in value.split(",") if part.strip())
        except ValueError as exc:
            raise ValueError(f"invalid degree bins: {value!r}") from exc
    else:
        bounds = tuple(int(item) for item in value)
    if not bounds or bounds[0] < 1 or tuple(sorted(set(bounds))) != bounds:
        raise ValueError("degree bins must be unique, increasing positive integers")
    return bounds


def degree_bin(degree: int, bounds: Sequence[int] = DEFAULT_DEGREE_BINS) -> str:
    if degree < 0:
        raise ValueError("degree cannot be negative")
    if degree == 0:
        return "0"
    lower = 1
    for upper in bounds:
        if degree <= upper:
            return str(upper) if lower == upper else f"{lower}-{upper}"
        lower = upper + 1
    return f"{lower}+"


def classify_groups(
    rows: Iterable[Mapping[str, object]],
    train: TrainIndex,
    num_characters: int,
    degree_bounds: Sequence[int] = DEFAULT_DEGREE_BINS,
    writer_column: str = "calligrapher_id",
    script_column: str = "script_id",
    character_column: str = "character_id",
) -> dict[str, list[EdgeGroup]]:
    grouped: dict[str, dict[tuple[int, int], EdgeGroup]] = {
        name: {} for name in STRATA
    }
    for row_index, source_row in enumerate(rows):
        writer, script, glyph = row_ids(
            source_row, num_characters, writer_column, script_column, character_column)
        edge = (writer, glyph)
        writer_seen = writer in train.writers
        glyph_seen = glyph in train.glyphs
        if edge in train.edges:
            stratum = "seen_edge"
        elif writer_seen and glyph_seen:
            stratum = "unseen_edge_seen_nodes"
        else:
            stratum = "unseen_node"

        writer_degree = int(train.writer_degree.get(writer, 0))
        glyph_degree = int(train.glyph_degree.get(glyph, 0))
        writer_bin = degree_bin(writer_degree, degree_bounds)
        glyph_bin = degree_bin(glyph_degree, degree_bounds)
        annotated: dict[str, object] = dict(source_row)
        annotated.update({
            "glyph_id": glyph,
            "writer_glyph_stratum": stratum,
            "train_writer_degree": writer_degree,
            "train_glyph_degree": glyph_degree,
            "train_writer_row_count": int(train.writer_row_count.get(writer, 0)),
            "train_glyph_row_count": int(train.glyph_row_count.get(glyph, 0)),
            "writer_degree_bin": writer_bin,
            "glyph_degree_bin": glyph_bin,
            "unseen_writer": int(not writer_seen),
            "unseen_glyph": int(not glyph_seen),
        })
        group = grouped[stratum].get(edge)
        if group is None:
            group = EdgeGroup(
                edge=edge,
                script_id=script,
                writer_degree_bin=writer_bin,
                glyph_degree_bin=glyph_bin,
                first_row_index=row_index,
                rows=[],
            )
            grouped[stratum][edge] = group
        group.rows.append(annotated)
    return {
        name: sorted(groups.values(), key=lambda group: group.first_row_index)
        for name, groups in grouped.items()
    }


def _quantiles(values: Iterable[int]) -> dict[str, float | int]:
    ordered = sorted(values)
    if not ordered:
        return {}

    def take(fraction: float) -> int:
        return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * fraction))]

    return {
        "min": ordered[0],
        "p10": take(.10),
        "p50": take(.50),
        "p90": take(.90),
        "p99": take(.99),
        "max": ordered[-1],
        "mean": sum(ordered) / len(ordered),
    }


def _bin_counts(groups: Iterable[EdgeGroup], attribute: str) -> dict[str, dict[str, int]]:
    edge_counts: Counter[str] = Counter()
    row_counts: Counter[str] = Counter()
    for group in groups:
        key = str(getattr(group, attribute))
        edge_counts[key] += 1
        row_counts[key] += len(group.rows)
    return {
        "edges": dict(sorted(edge_counts.items())),
        "rows": dict(sorted(row_counts.items())),
    }


def build_summary(
    train: TrainIndex,
    train_rows: int,
    groups: Mapping[str, Sequence[EdgeGroup]],
    input_rows: int,
    num_characters: int,
) -> dict[str, object]:
    strata: dict[str, object] = {}
    for name in STRATA:
        current = list(groups[name])
        rows = sum(len(group.rows) for group in current)
        joint_edges = Counter(
            f"{group.writer_degree_bin}|{group.glyph_degree_bin}"
            for group in current)
        strata[name] = {
            "rows": rows,
            "edges": len(current),
            "repeated_rows": rows - len(current),
            "writer_degree_bins": _bin_counts(current, "writer_degree_bin"),
            "glyph_degree_bins": _bin_counts(current, "glyph_degree_bin"),
            "joint_degree_bin_edges": dict(sorted(joint_edges.items())),
            "script_edge_counts": dict(sorted(Counter(
                str(group.script_id) for group in current).items())),
        }
    return {
        "glyph_formula": f"script_id * {num_characters} + character_id",
        "train": {
            "rows": train_rows,
            "edges": len(train.edges),
            "active_writers": len(train.writers),
            "active_glyphs": len(train.glyphs),
            "writer_edge_degree": _quantiles(train.writer_degree.values()),
            "glyph_edge_degree": _quantiles(train.glyph_degree.values()),
            "writer_row_count": _quantiles(train.writer_row_count.values()),
            "glyph_row_count": _quantiles(train.glyph_row_count.values()),
        },
        "input": {
            "rows": input_rows,
            "edges": sum(len(groups[name]) for name in STRATA),
        },
        "strata": strata,
    }


def select_groups(
    groups: Sequence[EdgeGroup],
    max_edges: int | None,
    seed: int,
) -> list[EdgeGroup]:
    """Select complete edge groups, balanced over script and degree support."""
    if max_edges is None or max_edges >= len(groups):
        return list(groups)
    if max_edges <= 0:
        raise ValueError("max_edges must be positive")
    buckets: dict[tuple[int, str, str], list[EdgeGroup]] = defaultdict(list)
    for group in groups:
        buckets[(group.script_id, group.writer_degree_bin,
                 group.glyph_degree_bin)].append(group)
    rng = random.Random(seed)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    selected: list[EdgeGroup] = []
    keys = sorted(buckets)
    while len(selected) < max_edges:
        progressed = False
        for key in keys:
            if buckets[key] and len(selected) < max_edges:
                selected.append(buckets[key].pop())
                progressed = True
        if not progressed:
            break
    return selected


def _output_fieldnames(input_fieldnames: Sequence[str]) -> list[str]:
    result = list(input_fieldnames)
    for field in ANNOTATION_FIELDS:
        if field not in result:
            result.append(field)
    return result


def write_outputs(
    output_dir: str | Path,
    input_fieldnames: Sequence[str],
    groups: Mapping[str, Sequence[EdgeGroup]],
    summary: dict[str, object],
    max_edges_per_stratum: int | None = None,
    seed: int = 0,
) -> dict[str, dict[str, int | str]]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    fieldnames = _output_fieldnames(input_fieldnames)
    selected_summary: dict[str, dict[str, int | str]] = {}
    for offset, name in enumerate(STRATA):
        selected = select_groups(groups[name], max_edges_per_stratum, seed + offset)
        path = output / f"{name}.csv"
        with open(path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for group in selected:
                # Edge groups remain contiguous, including every repeated row.
                writer.writerows(group.rows)
        selected_summary[name] = {
            "path": str(path),
            "edges": len(selected),
            "rows": sum(len(group.rows) for group in selected),
        }
    summary["outputs"] = selected_summary
    summary_path = output / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    return selected_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", default="5script/train.csv")
    parser.add_argument("--input", "--eval", dest="input_csv", required=True,
                        help="CSV rows to classify relative to --train.")
    parser.add_argument("--num-characters", type=int, default=7026)
    parser.add_argument("--writer-column", default="calligrapher_id")
    parser.add_argument("--script-column", default="script_id")
    parser.add_argument("--character-column", default="character_id")
    parser.add_argument("--degree-bins", default=",".join(map(str, DEFAULT_DEGREE_BINS)))
    parser.add_argument("--output-dir", default=None,
                        help="Write annotated stratum CSVs only when explicitly set.")
    parser.add_argument("--max-edges-per-stratum", type=int, default=None,
                        help="Optional number of whole edge groups selected per stratum.")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bounds = parse_degree_bins(args.degree_bins)
    train_fields, train_rows = read_csv(args.train)
    input_fields, input_rows = read_csv(args.input_csv)
    required = {args.writer_column, args.script_column, args.character_column}
    for label, fields in (("train", train_fields), ("input", input_fields)):
        missing = required.difference(fields)
        if missing:
            raise ValueError(f"{label} CSV is missing columns: {sorted(missing)}")
    index = build_train_index(
        train_rows, args.num_characters, args.writer_column,
        args.script_column, args.character_column)
    groups = classify_groups(
        input_rows, index, args.num_characters, bounds,
        args.writer_column, args.script_column, args.character_column)
    summary = build_summary(
        index, len(train_rows), groups, len(input_rows), args.num_characters)
    summary["paths"] = {"train": str(args.train), "input": str(args.input_csv)}
    if args.output_dir is not None:
        write_outputs(
            args.output_dir, input_fields, groups, summary,
            args.max_edges_per_stratum, args.seed)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
