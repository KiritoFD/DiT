"""Pinned identities and atomic queue records; no cross-checkpoint row guessing."""
import csv
import hashlib
import json
from pathlib import Path

TOP10={601:0,703:1,479:2,956:3,9001:4,760:5,672:6,365:7,461:8,53:9}


def read(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    temp.replace(path)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def checkpoint_rows(config):
    if getattr(config,'semantic_writer_rows',None):
        return {int(k):int(v) for k,v in config.semantic_writer_rows.items()}
    mapping={int(k):int(v) for k,v in json.loads(Path(config.callig_remap_json).read_text()).items()}
    if set(TOP10)<=set(mapping):return {k:mapping[k] for k in TOP10}
    raise ValueError('Ambiguous row-space checkpoint: supply its audited semantic_writer_rows, never guess identity')


def evaluation_csv(source,target,config):
    mapping=checkpoint_rows(config);rows=read(source)
    # Eval sets must retain semantic raw IDs. Explicit raw_writer_id takes priority.
    for row in rows:
        raw=int(row.get('raw_writer_id') or row.get('original_calligrapher_id') or row['calligrapher_id'])
        if raw not in mapping:raise ValueError(f'No checkpoint semantic row for {raw}: {source}')
        row['raw_writer_id']=str(raw);row['calligrapher_id']=str(mapping[raw])
        row['writer_label_space']='checkpoint_row'
    target=Path(target);target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    return str(target)


def verify_inputs(manifest):
    for path,expected in manifest['input_hashes'].items():
        if sha(path)!=expected:raise ValueError('Pinned experiment input changed: '+path)
