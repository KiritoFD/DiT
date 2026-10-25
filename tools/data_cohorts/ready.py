"""Verify cached targets/STD coverage and emit budget-free U114 training templates."""
import argparse
import csv
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from tools.data_clean.common import rows,key,sha,read_json,write_json


def run(args):
    data,cache,std = Path(args.datasets),Path(args.cache),Path(args.std)
    output = Path(args.output)
    if output.exists():raise FileExistsError(output)
    manifest = read_json(data/'manifest.json')
    for size,level in manifest['levels'].items():
        for name,expected in level['files'].items():
            if sha(data/f'top{size}'/name) != expected:raise ValueError('Dataset/table file changed')
    train,test,validation = rows(data/'all_train.csv'),rows(data/'all_test.csv'),rows(data/'all_validation.csv')
    expected = {int(r['img_id']):Path(r['image_path']).name for r in train}
    found={};shard_info=[]
    provenance = read_json(cache/'PROVENANCE.json')
    if (provenance['encode_mode'],provenance['scaling_factor'],provenance['shift_factor']) != ('mode',.18215,0.):
        raise ValueError('Unexpected target latent contract')
    for path in sorted(cache.glob('shard_*.npz')):
        with np.load(path) as z:
            ids,names,latents = z['img_ids'],z['names'],z['latents']
            if latents.shape != (len(ids),4,32,32) or not np.isfinite(latents).all():raise ValueError('Invalid target latents')
            for idx,name in zip(ids,names):
                if int(idx) in found:raise ValueError('Duplicate cached target ID')
                found[int(idx)] = str(name)
            shard_info.append(dict(path=str(path),n=len(ids),sha256=sha(path)))
    if any(found.get(i) != name for i,name in expected.items()):raise ValueError('Training cache IDs/names mismatch')
    validation_ids = {int(r['img_id']) for r in validation}
    if set(found)-set(expected)-validation_ids:raise ValueError('Unexpected cache extras outside frozen validation set')
    with (std/'index.csv').open(encoding='utf-8-sig') as f:std_rows=list(csv.DictReader(f))
    std_map={(int(r['script_id']),r['character']):r for r in std_rows}
    if len(std_map)!=len(std_rows):raise ValueError('Duplicate STD keys')
    used={(int(r['script_id']),r['character']) for r in train+test+validation}
    if used-set(std_map):raise ValueError('STD coverage incomplete')
    for pair in used:
        with Image.open(std/std_map[pair]['path']) as im:
            if im.size != (64,64):raise ValueError('STD shape mismatch')
    if read_json(std/'provenance.json')['missing_count']:raise ValueError('STD build reported missing glyphs')
    # One real dataset construction checks the existing loader contract, no optimizer.
    from src.utils.latent_dataset import MCCDLatentDataset
    maps={name:{int(k):int(v) for k,v in read_json(data/f'top20/tables/{name}_remap.json').items()}
          for name in ('callig','font','char')}
    ds = MCCDLatentDataset(str(data/'top20/train.csv'),str(cache),img_root='.',preload=False,
        is_train=True,load_image=False,use_glyph_cond=False,char_remap=maps['char'],
        callig_remap=maps['callig'],font_remap=maps['font'],std_soft_root=str(std),std_soft_preload=False)
    for idx in (0,len(ds)-1):
        sample=ds[idx]
        assert sample['latent'].shape == (4,32,32) and sample['std_soft'].shape == (1,64,64)
        assert 0 <= int(sample['y_callig_raw']) < 20 and 0 <= int(sample['y_script']) < 3
    from src.eval.dataset import make_eval_cache
    eval_map = {int(k):int(v) for k,v in read_json(data/'top20/tables/callig_remap.json').items()}
    a=make_eval_cache(str(data/'top2/test.csv'),'','',256,2,8,4,.18215,callig_id_map=eval_map,std_soft_root=str(std))
    b=make_eval_cache(str(data/'top20/test.csv'),'','',256,2,8,4,.18215,callig_id_map=eval_map,std_soft_root=str(std))
    assert torch.equal(a['noise'],b['noise']) and torch.equal(a['gts'],b['gts']) and torch.equal(a['skels_latent'],b['skels_latent'])
    output.mkdir(parents=True)
    from src.unet.io import Config
    configs={}
    for size in (2,5,10,20):
        folder=data/f'top{size}';tables=folder/'tables'
        c=Config(channels=(192,384,576),num_res_blocks=4,num_writers=size,num_scripts=3,batch=512,
            lr=6e-5,compile=True,max_steps=0,lr_decay_steps=0,
            data_csv=str(folder/'train.csv'),eval_csv=str(folder/'validation.csv'),
            latent_shards_dir=str(cache),img_root='',std_soft_root=str(std),std_shape_aug='off',
            condition_tables_root=str(tables),char_remap_json=str(tables/'char_remap.json'),
            callig_remap_json=str(tables/'callig_remap.json'),font_remap_json=str(tables/'font_remap.json'),
            output=f'assets/results/clean_cohorts/top{size}_u114')
        c.validate(require_budget=False)
        path=output/f'top{size}_u114.template.json';write_json(path,asdict(c));configs[path.name]=sha(path)
    write_json(output/'READY.json',dict(schema='writer-cohort-readiness-v1',data_interface='verified',
        quality_status=manifest['status'],training_started=False,training_budget='unset; templates max_steps=0',
        dataset_manifest_sha256=sha(data/'manifest.json'),n_latents_stored=len(found),n_train_latents=len(expected),
        n_validation_targets_cached_but_excluded_from_training=len(set(found)-set(expected)),n_test=len(test),n_validation=len(validation),
        n_std_pairs=len(used),shards=shard_info,std_provenance_sha256=sha(std/'provenance.json'),
        real_loader_samples_checked=2,nested_eval_gt_std_noise_equal=True,configs=configs,
        contract=dict(shape=[4,32,32],encode_mode='mode',scale=.18215,shift=0)))
    print('READY',dict(train=len(expected),latents_stored=len(found),validation=len(validation),test=len(test),std_pairs=len(used),training_started=False),flush=True)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('datasets','cache','std','output'):p.add_argument('--'+name,required=True)
    run(p.parse_args(argv))


if __name__ == '__main__':main()
