"""Standalone Pillow fixture tests; no GPU/data/model checkpoints needed."""
from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from compare_ckpt_poster import make_poster, read_and_validate


class PosterTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.rows = [
            {'idx': str(i), 'img_id': f'original/{i:02}.png',
             'char': chr(0x4E00 + i), 'calligrapher': '王羲之', 'script': '行'}
            for i in range(20)
        ]
        self.write_csv('master.csv', self.rows)
        self.sources = []
        for name, reverse in [('gt', False), ('model_a', True), ('model_b', False)]:
            folder = self.root / name
            folder.mkdir()
            records = list(reversed(self.rows)) if reverse else list(self.rows)
            rewritten = []
            for pos, entry in enumerate(records):
                updated = dict(entry)
                updated['idx'] = str(pos)
                rewritten.append(updated)
                col = int(entry['idx'])
                Image.new('RGB', (64, 64), ((col * 13) % 256, (col * 7) % 256,
                                             255 if name == 'gt' else 0)).save(folder / f'g{pos}.png')
            self.write_csv(name + '.csv', rewritten)
            self.sources.append({
                'name': name, 'method': name + ' method',
                'protocol_id': 'heldout187:ema:seed0:cfg1',
                'image_pattern': f'{name}/g{{idx}}.png',
                'sample_index_csv': name + '.csv', 'match_by': 'img_id',
            })
        (self.root / 'summary.json').write_text('{"source":"test-fixture"}')
        for item in self.sources[1:]:
            item['metrics'] = {'SSIM': '0.500', 'Writer AUC': '80.0%'}
            item['metrics_source'] = 'summary.json'
            item['metrics_protocol_id'] = 'heldout187:ema:seed0:cfg1'

    def tearDown(self):
        self.temp.cleanup()

    def write_csv(self, path, rows):
        with (self.root / path).open('w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=['idx', 'img_id', 'char', 'calligrapher', 'script'])
            w.writeheader()
            w.writerows(rows)

    def spec(self, n):
        return {
            'manifest_id': 'mock', 'title': '同字同条件 CKPT对照',
            'protocol_id': 'heldout187:ema:seed0:cfg1',
            'sample_ids': list(range(n)), 'sample_index_csv': 'master.csv',
            'sample_index_mode': 'explicit_idx', 'rows': self.sources,
        }

    def test_10_and_20_generate_with_alignment(self):
        for n in (10, 20):
            with self.subTest(n=n):
                out = self.root / f'poster{n}.png'
                make_poster(self.spec(n), self.root, out, None, 150)
                self.assertTrue(out.exists())
                with Image.open(out) as im:
                    self.assertGreater(im.height, 1800)
                    self.assertGreater(im.width, 700)
                frozen = json.loads(out.with_suffix('.selection.json').read_text())
                self.assertEqual([a['master_idx'] for a in frozen['rows'][1]['images']], list(range(n)))
                self.assertEqual([a['model_idx'] for a in frozen['rows'][1]['images']], [19-i for i in range(n)])
                self.assertEqual(len(frozen['rows'][0]['images']), n)
                self.assertEqual(len(frozen['poster_sha256']), 64)

    def test_reject_misaligned_identity(self):
        self.rows[0]['char'] = '錯'
        self.write_csv('master.csv', self.rows)
        with self.assertRaisesRegex(ValueError, 'char mismatch'):
            read_and_validate(self.spec(10), self.root)

    def test_reject_invented_metrics_or_protocol(self):
        spec = self.spec(10)
        spec['rows'][1]['metrics_source'] = 'missing.json'
        with self.assertRaises(FileNotFoundError):
            read_and_validate(spec, self.root)
        spec = self.spec(10)
        spec['rows'][1]['metrics_protocol_id'] = 'different'
        with self.assertRaisesRegex(ValueError, 'metric protocol'):
            read_and_validate(spec, self.root)

    def test_reject_duplicate_ids(self):
        spec = self.spec(10)
        spec['sample_ids'] = [0] * 10
        with self.assertRaisesRegex(ValueError, 'distinct'):
            read_and_validate(spec, self.root)

    def test_reject_indexed_png_missing(self):
        (self.root / 'model_b' / 'g0.png').unlink()
        with self.assertRaises(FileNotFoundError):
            read_and_validate(self.spec(10), self.root)

    def test_reject_missing_idx_protocol(self):
        spec = self.spec(10)
        spec['rows'][1]['protocol_id'] = 'wrong'
        with self.assertRaisesRegex(ValueError, 'protocol_id mismatch'):
            read_and_validate(spec, self.root)


if __name__ == '__main__':
    unittest.main()
