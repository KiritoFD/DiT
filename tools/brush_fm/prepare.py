"""P0: decode the fixed VAE target cache, measure attributes, fit train-only statistics."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.stats import spearmanr
import torch
from diffusers import AutoencoderKL
from src.unet.brush.attributes import measure,normalize,NAMES
from tools.generic_pretrain.contracts import read_rows,write_rows,sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();source=Path(a.source);out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    rows=read_rows(source/'encode.csv');byid={int(r['img_id']):r for r in rows}
    trainids={int(r['img_id']) for r in read_rows(source/'train.csv')}
    toptrain=[r for r in read_rows(source/'top10_train.csv') if r['generic']=='0']
    trainids.update(int(r['img_id']) for r in toptrain)
    writer_map=json.loads((source/'writer_table/callig_remap.json').read_text());scriptmap={'0':0,'3':1,'4':2}
    c=json.loads((source/'train_config.json').read_text());vae=AutoencoderKL.from_pretrained(c['vae_path']).cuda().eval()
    vae.requires_grad_(False);records=[];pairs=[];seen=0
    with ThreadPoolExecutor(max_workers=12) as executor,torch.inference_mode():
        for path in sorted((source/'latents').glob('shard_*.npz')):
            data=np.load(path)
            for start in range(0,len(data['img_ids']),64):
                ids=data['img_ids'][start:start+64]
                z=torch.from_numpy(data['latents'][start:start+64]).cuda().float()
                with torch.autocast('cuda',dtype=torch.float16):pixels=vae.decode(z/c['vae_scaling_factor']).sample
                ink=(1-(pixels.float().clamp(-1,1)+1)/2).mean(1).cpu().numpy()
                result=list(executor.map(measure,ink))
                for iid,(value,valid) in zip(ids,result):
                    row=byid[int(iid)];records.append((int(iid),value,valid,scriptmap[row['script_id']],
                        writer_map.get(row['calligrapher_id'],-1),ord(row['character']),int(iid) in trainids))
                    if len(pairs)<1024:
                        with Image.open(row['image_path']) as image:original=1-np.asarray(image.convert('L'),dtype=np.float32)/255
                        raw,ok=measure(original)
                        if valid and ok:pairs.append((raw,value))
                seen+=len(ids)
            print('ATTRIBUTES',seen,'/',len(rows),flush=True)
    records.sort(key=lambda r:r[0]);ids=np.array([r[0] for r in records]);values=np.stack([r[1] for r in records])
    valid=np.array([r[2] for r in records]);scripts=np.array([r[3] for r in records]);train=np.array([r[6] for r in records])
    normalized,center,scale=normalize(values,scripts,train,valid)
    np.savez_compressed(out/'attributes.npz',ids=ids,values=values,normalized=normalized,valid=valid,scripts=scripts,
        train=train,writers=np.array([r[4] for r in records]),chars=np.array([r[5] for r in records]))
    old=np.stack([p[0] for p in pairs]);new=np.stack([p[1] for p in pairs])
    audit=dict(n=len(ids),valid_fraction=float(valid.mean()),paired_audit_n=len(pairs),names=NAMES,
        original_reconstruction_spearman=[float(spearmanr(old[:,i],new[:,i]).statistic) for i in range(5)],
        median_absolute_roundtrip_delta=np.median(np.abs(old-new),axis=0).tolist(),
        center=center.tolist(),scale=scale.tolist(),statistics='train-only per-script median/MAD; invalid -> Brush null',
        source_manifest_sha256=sha(source/'manifest.json'),edge='gray transition area / binary perimeter / bbox height',
        limitations='paired audit is first1024 valid cache rows, not source-balanced; no artwork-disjoint claim')
    audit['p0_pass']=bool(valid.mean()>=.9 and min(audit['original_reconstruction_spearman'][:3])>=.7)
    (out/'audit.json').write_text(json.dumps(audit,indent=2))
    print('P0_AUDIT',json.dumps(audit),flush=True)


if __name__=='__main__':main()
