"""Lossless portable common assets. Never changes the original P1/P2/VAE."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import tarfile
import numpy as np
import torch
from ..runtime.common import save,sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--p2-config',required=True)
    p.add_argument('--out',default='assets/style_matrix_20261026/common');p.add_argument('--seal',action='store_true')
    p.add_argument('--archive',default='')
    a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    if a.seal:
        manifest=json.loads((out/'manifest.json').read_text())
        for item in manifest['files']:
            assert sha(item['path'])==item['sha256'], item['path']
        save(out/'ready.json',dict(manifest_sha256=sha(out/'manifest.json'),
                                  r4_deferred=True))
        return
    if (out/'manifest.json').exists():raise FileExistsError('Use a new directory or --seal, never overwrite a package')
    config=json.loads(Path(a.p2_config).read_text());rows=list(csv.DictReader(open(config['data_csv'],encoding='utf-8-sig')))
    wanted={int(r['img_id']) for r in rows};found=set()
    checkpoint=torch.load(a.source,map_location='cpu',weights_only=False)
    source_step=int(checkpoint['train_steps'])
    compact={k:checkpoint[k] for k in ('ema','args','config','architecture','train_steps') if k in checkpoint}
    compact['source_checkpoint_sha256']=sha(a.source)
    torch.save(compact,out/'content_p1_ema.pt');del checkpoint,compact
    latents=out/'latents';latents.mkdir()
    for index,path in enumerate(sorted(Path(config['latent_shards_dir']).glob('shard_*.npz'))):
        with np.load(path) as data:
            ids=data['img_ids'];take=np.array([i for i,v in enumerate(ids) if int(v) in wanted],dtype=np.int64)
            if not len(take):continue
            selected={k:data[k][take] if len(data[k])==len(ids) else data[k] for k in data.files}
            dest=latents/f'shard_{index:05d}.npz';np.savez_compressed(dest,**selected)
            with np.load(dest) as verify:
                for key,value in selected.items():assert np.array_equal(value,verify[key])
            found.update(int(v) for v in ids[take])
    assert found==wanted, 'Incomplete training latent coverage'
    bank=torch.load(config['std_shape_cache'],map_location='cpu',weights_only=True,mmap=True)
    keys={(int(r['script_id']),r['character']) for r in rows}
    take=[i for i,key in enumerate(bank['keys']) if tuple(key) in keys]
    assert len(take)==len(keys)
    bank=dict(bank);bank['keys']=[bank['keys'][i] for i in take]
    original=bank['variants'];bank['variants']=original[take]
    assert torch.equal(bank['variants'],original[take]);torch.save(bank,out/'std_shape_bank.pt')
    shutil.copytree(config['std_soft_root'],out/'std_bbox64',dirs_exist_ok=True)
    shutil.copytree(config['condition_tables_root'],out/'tables',dirs_exist_ok=True)
    shutil.copy2(config['data_csv'],out/'train.csv');shutil.copy2(config['eval_csv'],out/'validation.csv')
    config.update(data_csv=str(out/'train.csv'),eval_csv=str(out/'validation.csv'),
                  latent_shards_dir=str(latents),std_shape_cache=str(out/'std_shape_bank.pt'),
                  std_soft_root=str(out/'std_bbox64'),condition_tables_root=str(out/'tables'),img_root='')
    for key in ('callig','font','char'):config[key+'_remap_json']=str(out/'tables'/f'{key}_remap.json')
    save(out/'base_config.json',config)
    files=[dict(path=str(path),bytes=path.stat().st_size,sha256=sha(path))
           for path in sorted(out.rglob('*')) if path.is_file()]
    save(out/'manifest.json',dict(source_checkpoint=a.source,source_checkpoint_sha256=sha(a.source),
        source_step=source_step,strict_ema_subset=True,train_csv_sha256=sha(out/'train.csv'),
        train_n=len(rows),latent_n=len(found),latent_dtype_unchanged=True,std_variants_bitwise_equal=True,files=files))
    if a.archive:
        with tarfile.open(a.archive,'w:gz',compresslevel=1) as archive:
            archive.add(out,arcname=out)
            for folder in sorted({str(Path(r['image_path']).parent) for r in rows}):archive.add(folder,arcname=folder)
    print('COMMON_PACKAGE_COMPLETE',out,flush=True)


if __name__=='__main__':main()
