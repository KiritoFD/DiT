"""Discover completed artifacts and freeze actual CSV/PNG identities."""
import csv
import hashlib
import json
from pathlib import Path


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8');temp.replace(path)


def csv_rows(path):return list(csv.DictReader(open(path,encoding='utf-8-sig')))


def discover(machine):
    roots=[Path('assets/style_matrix_20261027/matrix64')/machine/'execution/runs',
           Path('assets/style_matrix_20261028/generation2full')/machine/'runs']
    result=[];seen=set()
    for root in roots:
        for p in root.glob('*/evaluations/*/*/complete.json'):
            if 'shared_train' in str(p) or 'swap' in str(p):continue
            if 'original_protocol' not in str(p) and 'heldout187_seed' not in str(p) and root==roots[1]:continue
            physical=str(p.resolve())
            if physical in seen:continue
            seen.add(physical);identity=read(p);folder=p.parent
            candidates=list(folder.glob('eval_samples_ctrl/step*/eval200fix'))
            if not candidates:continue
            arm=p.parents[3].name;step=int(p.parents[1].name)
            csv_path=identity.get('csv') or identity.get('eval_csv')
            if not csv_path:
                if root==roots[1]:
                    cfg=read(root.parent/'specs'/f'{arm}.json')['config']
                    if p.parent.name=='original_protocol':csv_path=cfg['eval_csv']
                    else:
                        family='J' if arm.startswith('J') else 'S'
                        csv_path=next(v['csv'] for v in read(root.parent/'protocols.json')[family] if v['name']==p.parent.name)
                else:
                    cfg=read(Path('assets/style_matrix_20261027/matrix64')/machine/'specs'/f'{arm}.json')['config']
                    csv_path=cfg['eval_csv']
            if hashlib.sha256(Path(csv_path).read_bytes()).hexdigest()!=identity['csv_sha256']:
                raise ValueError('CSV identity mismatch: '+str(p))
            rows=csv_rows(csv_path);samples=candidates[0]
            if any(not (samples/f'g{i}.png').is_file() for i in range(len(rows))):raise ValueError('Incomplete indexed PNG set: '+str(folder))
            result.append(dict(arm=arm,step=step,protocol=p.parent.name,csv=csv_path,samples=str(samples),
                n=len(rows),csv_sha256=hashlib.sha256(Path(csv_path).read_bytes()).hexdigest(),identity=identity))
    return sorted(result,key=lambda d:(d['arm'],d['step'],d['protocol']))


def reference_bank(csv_path,k=160):
    slots={};seen=set()
    for row in csv_rows(csv_path):
        if row.get('aug'):continue
        path=Path(row['image_path'])
        if not path.is_file():raise FileNotFoundError(path)
        digest=row.get('pixel_sha256') or hashlib.sha256(path.read_bytes()).hexdigest()
        if digest in seen:continue
        seen.add(digest);row.update(pixel_sha256=digest)
        # Source is a dataset label where work provenance is absent: never claim artwork holdout.
        slots.setdefault((row['calligrapher'],row['script']),[]).append(row)
    refs=[]
    for slot,rows in sorted(slots.items()):
        rows=sorted(rows,key=lambda r:hashlib.sha256(r['image_path'].encode()).hexdigest())
        by={}
        for row in rows:by.setdefault((row.get('source','unknown'),row['character']),[]).append(row)
        chosen=[]
        while len(chosen)<k and by:
            for key in sorted(list(by)):
                chosen.append(by[key].pop(0))
                if not by[key]:del by[key]
                if len(chosen)==k:break
        refs.extend(chosen)
    return refs
