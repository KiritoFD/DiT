import csv
import tempfile
import unittest
from pathlib import Path

from tools.build_writer_glyph_eval import (
    STRATA,
    build_summary,
    build_train_index,
    classify_groups,
    degree_bin,
    row_ids,
    write_outputs,
)


class WriterGlyphEvalTest(unittest.TestCase):
    num_characters = 10

    def setUp(self):
        # Edge (0, glyph 0) deliberately has a repeated training row.
        self.train_rows = [
            {"image_path": "a.png", "calligrapher_id": "0", "script_id": "0", "character_id": "0"},
            {"image_path": "b.png", "calligrapher_id": "0", "script_id": "0", "character_id": "0"},
            {"image_path": "c.png", "calligrapher_id": "0", "script_id": "0", "character_id": "1"},
            {"image_path": "d.png", "calligrapher_id": "1", "script_id": "0", "character_id": "0"},
            {"image_path": "e.png", "calligrapher_id": "2", "script_id": "1", "character_id": "0"},
        ]
        self.input_rows = [
            # Two repeated seen-edge rows.
            {"image_path": "s1.png", "calligrapher_id": "0", "script_id": "0", "character_id": "0"},
            {"image_path": "s2.png", "calligrapher_id": "0", "script_id": "0", "character_id": "0"},
            # Both endpoints occur in training, but this edge does not.
            {"image_path": "c1.png", "calligrapher_id": "1", "script_id": "0", "character_id": "1"},
            {"image_path": "c2.png", "calligrapher_id": "1", "script_id": "0", "character_id": "1"},
            # Unseen writer and unseen glyph respectively.
            {"image_path": "u1.png", "calligrapher_id": "9", "script_id": "0", "character_id": "0"},
            {"image_path": "u2.png", "calligrapher_id": "0", "script_id": "1", "character_id": "1"},
        ]

    def test_glyph_formula_and_degrees(self):
        self.assertEqual(row_ids(self.train_rows[-1], self.num_characters), (2, 1, 10))
        index = build_train_index(self.train_rows, self.num_characters)
        self.assertEqual(len(index.edges), 4)
        self.assertEqual(index.writer_degree[0], 2)
        self.assertEqual(index.glyph_degree[0], 2)
        self.assertEqual(index.writer_row_count[0], 3)
        self.assertEqual(degree_bin(0), "0")
        self.assertEqual(degree_bin(4), "3-5")

    def test_classification_keeps_repeated_edges_grouped(self):
        index = build_train_index(self.train_rows, self.num_characters)
        groups = classify_groups(self.input_rows, index, self.num_characters)
        self.assertEqual({name: len(groups[name]) for name in STRATA}, {
            "seen_edge": 1,
            "unseen_edge_seen_nodes": 1,
            "unseen_node": 2,
        })
        self.assertEqual(len(groups["seen_edge"][0].rows), 2)
        clean = groups["unseen_edge_seen_nodes"][0]
        self.assertEqual(len(clean.rows), 2)
        self.assertEqual(clean.rows[0]["glyph_id"], 1)
        self.assertEqual(clean.rows[0]["train_writer_degree"], 1)
        self.assertEqual(clean.rows[0]["train_glyph_degree"], 1)

    def test_output_limit_counts_edges_not_rows(self):
        index = build_train_index(self.train_rows, self.num_characters)
        groups = classify_groups(self.input_rows, index, self.num_characters)
        summary = build_summary(
            index, len(self.train_rows), groups, len(self.input_rows),
            self.num_characters)
        fields = list(self.input_rows[0])
        with tempfile.TemporaryDirectory() as tmp:
            selected = write_outputs(
                tmp, fields, groups, summary, max_edges_per_stratum=1, seed=3)
            self.assertEqual(selected["seen_edge"]["edges"], 1)
            self.assertEqual(selected["seen_edge"]["rows"], 2)
            with open(Path(tmp) / "seen_edge.csv", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([row["image_path"] for row in rows], ["s1.png", "s2.png"])
            self.assertTrue((Path(tmp) / "summary.json").is_file())


if __name__ == "__main__":
    unittest.main()
