"""eval_style_v2 shared helpers: hashing, JSON IO, CSV loading, grouped splits.

Discipline: every artifact is traceable via sha256; caches are keyed by
(feature_version, preprocessing, image identity); no composite score is ever
produced. See 1030eval.md for the plan and docs/1030/eval_style_v2/ for outputs.
"""
import csv
import hashlib
import json
from pathlib import Path

FEATURE_VERSION='esv2-1.0'
PREPROCESS='gray256_bicubic_uint8'
REAL_CSV='assets/u100_p1p2_20261025/p2_train.csv'


def sha256_file(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def sha256_array(array):
    import numpy as np
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def save_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    temp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def load_real_rows(csv_path=REAL_CSV):
    """Canonical Top10 real rows; pixel_sha256 dedup keeps first occurrence."""
    rows=list(csv.DictReader(open(csv_path,encoding='utf-8-sig')))
    seen=set();unique=[]
    for r in rows:
        key=r.get('pixel_sha256') or ''
        if key:
            if key in seen:continue
            seen.add(key)
        unique.append(r)
    return unique


def stable_fold(key,salt,n_folds):
    """Deterministic character-grouped fold in [0,n_folds)."""
    digest=hashlib.md5((salt+'|'+str(key)).encode()).hexdigest()
    return int(digest,16)%n_folds


def slot_of(row):
    return (int(row['calligrapher_id']),int(row['script_id']))


def out_dir(*parts):
    base=Path('docs/1030/eval_style_v2')
    for part in parts:base=base/part
    base.parent.mkdir(parents=True,exist_ok=True)
    return base


def cache_dir(*parts):
    base=Path('assets/eval_style_v2_cache')
    for part in parts:base=base/part
    base.parent.mkdir(parents=True,exist_ok=True)
    return base
