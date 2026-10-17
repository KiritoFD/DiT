"""GT-only sidecar: unclipped signed distances at 32px, no canonical-cache edits."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt


def signed_distance(gray):
    ink=np.asarray(gray)<140
    # Positive inside ink. One unit = four latent cells = 16 pixels at 128px.
    d=(distance_transform_edt(ink)-distance_transform_edt(~ink))/16.
    return d.reshape(32,4,32,4).mean((1,3)).astype(np.float16)[None]


def main():
    p=argparse.ArgumentParser();p.add_argument('--pixel-data',required=True);p.add_argument('--latent-data',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    torch.set_num_threads(4);out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    pixel=torch.load(a.pixel_data,map_location='cpu',weights_only=False);latent=torch.load(a.latent_data,map_location='cpu',weights_only=False)
    for key in ('labels','char','source'):assert np.array_equal(pixel[key],latent[key])
    with ThreadPoolExecutor(max_workers=8) as pool:sdf=np.stack(list(pool.map(signed_distance,pixel['gray'][:,0].numpy())))
    torch.save(torch.from_numpy(sdf),out/'unclipped_sdf.pt')
    manifest={'shape':list(sdf.shape),'sign':'positive ink, negative background','threshold_gray_uint8':140,'normalization_pixels_at128':16,
              'clip':'AFTER augmentation and division by affine scale, to [-1,1]','target':'real GT, never STD; training only uses train indices',
              'latent_manifest_sha256':hashlib.sha256(Path(a.latent_data).with_name('manifest.json').read_bytes()).hexdigest(),
              'source_pixel_manifest_sha256':hashlib.sha256(Path(a.pixel_data).with_name('manifest.json').read_bytes()).hexdigest()}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8');print('GEOMETRY_SDF_READY',manifest,flush=True)


if __name__=='__main__':main()
