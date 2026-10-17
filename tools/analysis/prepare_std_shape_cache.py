"""Re-render the EXACT selected font faces, verify hashes, optionally preview on CPU."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from src.utils.std_soft_glyph import FontFace,render_base_mask


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--store',required=True);p.add_argument('--out',required=True)
    p.add_argument('--font-dir',action='append',default=[]);a=p.parse_args()
    root=Path(a.store);out=Path(a.out)
    if out.exists():raise FileExistsError(out)
    provenance=json.loads((root/'provenance.json').read_text(encoding='utf-8'))
    expected={(Path(f['path'].replace('\\','/')).name,int(f.get('face',0))):f['sha256'] for f in provenance['fonts']}
    with (root/'index.csv').open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
    faces={};masks=[];keys=[]
    for row in rows:
        identity=(row['font_file'],int(row['font_face']))
        if identity not in faces:
            candidates=[Path(row['font_path'])]+[Path(d)/row['font_file'] for d in a.font_dir]
            path=next((p for p in candidates if p.is_file()),None)
            if path is None:raise FileNotFoundError(f'Original font unavailable: {identity}; use --font-dir, no replacement allowed')
            if identity not in expected or sha(path)!=expected[identity]:raise ValueError(f'Font hash mismatch: {path}')
            faces[identity]=FontFace(str(path),identity[1])
        mask=render_base_mask(row['character'],faces[identity],
            render_size=max(256,int(provenance['render_size'])),output_size=256,
            font_ratio=float(provenance['font_ratio']))
        masks.append(torch.from_numpy(np.array(mask,copy=True))[None]);keys.append((int(row['script_id']),row['character']))
    out.parent.mkdir(parents=True,exist_ok=True)
    torch.save({'version':1,'source_kind':'font_rerender','masks':torch.stack(masks),'keys':keys,
        'source_index_sha256':sha(root/'index.csv'),'source_provenance_sha256':sha(root/'provenance.json'),
        'font_sha256':{str(k):expected[k] for k in faces}},out)
    print('STD_HIGHRES_READY',len(keys),out,flush=True)


if __name__=='__main__':main()
