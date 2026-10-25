"""Freeze three-writer GT datasets and their independently trained writer table."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import tarfile

WRITERS=('颜真卿','智永','赵佶')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(root,full_top10=False):
    import numpy as np
    from PIL import Image
    root.mkdir(parents=True,exist_ok=True)
    if (root/'data.json').exists():raise FileExistsError(root/'data.json')
    (root/'images').mkdir()
    cohort=Path('assets/data_cohorts/top20_20261021_v1')
    table=cohort/'writer_tables_supcon_v1'
    mapping=json.loads((table/'callig_remap.json').read_text())
    source=np.load(table/'callig_table.npy')
    cohort_name='top10' if full_top10 else 'top20'
    with (cohort/'datasets_final'/cohort_name/'train.csv').open(encoding='utf-8-sig') as stream:
        training=list(csv.DictReader(stream))
    writers=tuple(sorted({r['calligrapher'] for r in training})) if full_top10 else WRITERS
    scripts=('楷','行','隶') if full_top10 else ('楷',)
    writer_raw={}
    datasets={}; rejected={}
    for split in ('train','validation','test'):
        path=cohort/'datasets_final'/cohort_name/f'{split}.csv'
        with path.open(encoding='utf-8-sig') as stream:
            rows=list(csv.DictReader(stream))
        selected=[]; rejected[split]=0
        for row in rows:
            if row['calligrapher'] not in writers or row['script'] not in scripts:
                continue
            writer_raw[row['calligrapher']]=row['calligrapher_id']
            im=Image.open(row['image_path']).convert('RGB')
            if im.size!=(256,256):
                raise ValueError(f'Unexpected GT size: {im.size}')
            ink=np.asarray(im.convert('L'))<128
            # Only remove blank/massively black images and obvious outer scan rules.
            frame=max(ink[:32].mean(1).max(),ink[-32:].mean(1).max(),
                      ink[:,:32].mean(0).max(),ink[:,-32:].mean(0).max())
            if not full_top10 and (not .01<float(ink.mean())<.6 or frame>.8):
                rejected[split]+=1;continue
            identity=hashlib.sha256(row['image_path'].encode()).hexdigest()[:20]
            name=f'images/{identity}.png';im.save(root/name)
            selected.append(dict(row,file=name,image_sha256=sha(root/name),
                                 writer_index=writers.index(row['calligrapher']),script_index=scripts.index(row['script'])))
        if not selected:
            raise ValueError(f'Empty split: {split}')
        datasets[split]=selected
    sets=[{r['image_path'] for r in datasets[s]} for s in datasets]
    if sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2]:
        raise ValueError('Split image leakage')
    values=np.stack([source[mapping[str(writer_raw[w])]] for w in writers])
    np.save(root/'writer_table.npy',values)
    manifest=dict(writers=writers,scripts=scripts,datasets=datasets,rejected=rejected,
                  writer_table_sha256=sha(root/'writer_table.npy'),
                  source_writer_table=str(table),source_table_sha256=sha(table/'callig_table.npy'),
                  raw_writer_ids=writer_raw,
                  split_csv_sha256={s:sha(cohort/'datasets_final'/cohort_name/f'{s}.csv') for s in datasets})
    if full_top10:manifest['prompt_schema']='writer_script_v1'
    (root/'data.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    with tarfile.open(root/'data.tar.gz','w:gz') as tar:
        for name in ('images','writer_table.npy','data.json'):tar.add(root/name,arcname=name)
    print('DATA_READY', {s:len(r) for s,r in datasets.items()},'rejected',rejected,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--full-top10',action='store_true')
    a=p.parse_args();main(a.root,a.full_top10)
