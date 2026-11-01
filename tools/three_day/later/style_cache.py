"""Frozen D2 features of the actual clean training latent cache, CPU-capable."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from tools.three_day.contracts import read,save,sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--root',required=True)
    p.add_argument('--device',default='cpu');a=p.parse_args();torch.set_num_threads(2)
    c=json.loads(Path(a.config).read_text());root=Path(a.root);out=root/'style_reference_D2.pt'
    if out.exists():raise FileExistsError(out)
    from src.utils.latent_dataset import MCCDLatentDataset
    maps={k:{int(x):int(y) for x,y in json.loads(Path(c[k+'_remap_json']).read_text()).items()} for k in ('callig','char','font')}
    dataset=MCCDLatentDataset(c['data_csv'],c['latent_shards_dir'],c['img_root'],preload=True,load_image=False,
        num_preload_workers=2,callig_remap=maps['callig'],char_remap=maps['char'],font_remap=maps['font'])
    ck=torch.load(root/'teachers/D2.pt',map_location='cpu',weights_only=False)
    from src.model.writer_style_latent import build_writer_style_latent
    model=build_writer_style_latent(len(ck['slots']),'geometry').to(a.device).eval().requires_grad_(False)
    model.load_state_dict(ck['model'],strict=True);features=[]
    with torch.inference_mode():
        for start in range(0,len(dataset),64):
            z=torch.from_numpy(dataset._latents[start:start+64]).to(a.device).float()
            features.append(model(z)['embedding'].cpu().half())
            if start%2048==0:print('STYLE_TRAIN_CACHE',start,len(dataset),flush=True)
    rows=dataset.samples
    payload=dict(features=torch.cat(features),writers=[r['calligrapher'] for r in rows],scripts=[r['script'] for r in rows],
        chars=[r['character'] for r in rows],sources=[r.get('source','unknown') for r in rows],
        img_ids=[int(r['img_id']) for r in rows],csv_sha256=sha(c['data_csv']),teacher_sha256=sha(root/'teachers/D2.pt'),
        vae=c['vae_path'],scale=c['vae_scaling_factor'],shift=c['vae_shift_factor'],latent_source=c['latent_shards_dir'])
    torch.save(payload,out);save(out.with_suffix('.json'),dict(n=len(rows),source_csv=c['data_csv'],sha256=sha(out),
        teacher_sha256=payload['teacher_sha256'],contract='same cached z0 as FM; training CSV only, no target heldout GT',device=a.device))


if __name__=='__main__':main()
