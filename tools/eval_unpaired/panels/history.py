"""Reconstruct historical sample metadata only after exact saved-GT pixel matching."""
import argparse
import hashlib
from pathlib import Path
from PIL import Image
from tools.eval_unpaired.protocol import discover,csv_rows,save


def pixel(path):return hashlib.sha256(Image.open(path).convert('RGB').tobytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--machine',choices=['4090','48'],required=True);p.add_argument('--out',required=True)
    p.add_argument('--u100-only',action='store_true')
    a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True);lookup={}
    for job in discover(a.machine):
        for i,row in enumerate(csv_rows(job['csv'])):
            key=pixel(Path(job['samples'])/f'gt{i}.png')
            if key in lookup and any(lookup[key][k]!=row[k] for k in ['character','script','calligrapher']):
                raise ValueError('Ambiguous saved-GT identity')
            lookup[key]=row
    paths=[]
    if a.machine=='48':
        if a.u100_only:
            paths.extend(('U100Base',55000,folder) for folder in Path('assets/results/unet_connectivity_100000/baseline_resume').glob('*/eval_samples_ctrl/step0055000/eval200fix'))
        else:
            for step in [75000,85000]:
                for folder in Path('assets/results/unet_deep_sym4_114m_60m').glob(f'*/eval_samples_ctrl/step{step:07d}/eval200fix'):
                    paths.append(('DeepSym4',step,folder))
    else:
        root=Path('assets/_archive_20261027/u100_p1p2_20261025')
        for arm,step in [('P1',10000),('P2',20000)]:
            for folder in root.glob(f'{arm.lower()}_runs/*/eval_samples_ctrl/step{step:07d}/eval200fix'):paths.append((arm,step,folder))
    jobs=[];excluded=[]
    import csv
    for arm,step,folder in paths:
        images=sorted(folder.glob('g[0-9]*.png'),key=lambda f:int(f.stem[1:]));rows=[]
        for i,image in enumerate(images):
            if image.name!=f'g{i}.png':raise ValueError('Historical index gap')
            key=pixel(folder/f'gt{i}.png')
            if key not in lookup:break
            rows.append(lookup[key])
        if len(rows)!=len(images):excluded.append(dict(arm=arm,step=step,matched=len(rows),n=len(images)));continue
        dest=out/f'{arm}_{step}.csv';fields=sorted(set().union(*(r.keys() for r in rows)))
        with dest.open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fields);writer.writeheader();writer.writerows(rows)
        jobs.append(dict(arm=arm,step=step,protocol=f'verified_saved_GT_{len(rows)}',csv=str(dest),samples=str(folder),
            n=len(rows),csv_sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),
            identity=dict(mapping='exact RGB saved-GT pixels to audited canonical metadata; original sampler not re-generated')))
    save(out/'jobs.json',jobs);save(out/'excluded.json',excluded);print('VERIFIED_HISTORY',[(j['arm'],j['step'],j['n']) for j in jobs])


if __name__=='__main__':main()
