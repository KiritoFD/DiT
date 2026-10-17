"""Exact v84A target-shard join; never silently substitute posterior modes."""
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from tools.train_writer_style_pixel import sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    pixel=Path('/root/Workspace/xy/DiT/scratch/writer_pixel_d0d1_20261016_01/assets/data')
    data=torch.load(pixel/'data.pt',map_location='cpu',weights_only=False)
    rows=[r for r in csv.DictReader(open('exp-std/csv/train.csv',encoding='utf-8-sig')) if not (r.get('aug') or '').strip()]
    assert len(rows)==len(data['labels'])
    wanted={int(r['img_id']):i for i,r in enumerate(rows)};assert len(wanted)==len(rows)
    z=np.empty((len(rows),4,32,32),np.float16);found=set();shards=[]
    for path in sorted(Path('exp-std/data/shards_img_aug').glob('shard_*.npz')):
        shard=np.load(path);ids=shard['img_ids'];selected=[(j,wanted[int(i)]) for j,i in enumerate(ids) if int(i) in wanted]
        if selected:
            latent=shard['latents'];assert latent.shape[1:]==(4,32,32)
            for j,i in selected:
                assert i not in found,('Duplicate shard target',i);z[i]=latent[j];found.add(i)
            shards.append({'path':str(path),'sha256':sha(path),'selected_rows':len(selected)})
        print('SHARD_JOIN',path.name,len(found),'/',len(rows),flush=True)
    assert len(found)==len(rows),('Missing target latents',len(rows)-len(found))
    mode=np.load('scratch/disc_lat_full.npz',allow_pickle=True)['z']
    diff=z.astype(float)-mode.astype(float)
    manifest=json.loads((pixel/'manifest.json').read_text());manifest.update({
        'domain':'latent','source_pixel_manifest_sha256':sha(pixel/'manifest.json'),
        'latent_source':'exact v84A target shards_img_aug, no extra VAE encoding/scaling/shift',
        'shards':shards,'mode_cache_relative_rms':float(np.sqrt(np.mean(diff**2))/np.sqrt(np.mean(mode.astype(float)**2))),
        'mode_cache_max_abs_difference':float(np.max(np.abs(diff))),
        'model_inputs':'canonical z0 [4,32,32] only, no STD/script/writer inputs'})
    del data['gray'];data['z']=torch.from_numpy(z.copy());torch.save(data,out/'data.pt')
    manifest['data_cache_sha256']=sha(out/'data.pt')
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print('LATENT_DATA_READY',manifest['mode_cache_relative_rms'],manifest['mode_cache_max_abs_difference'],flush=True)


if __name__=='__main__':main()
