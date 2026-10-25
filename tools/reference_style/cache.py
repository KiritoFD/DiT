"""Offline frozen DINOv2 CLS cache from the fixed training CSV only."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel
from .prepare import read


class Images(Dataset):
    def __init__(self, rows):self.rows=rows
    def __len__(self):return len(self.rows)
    def __getitem__(self, i):
        with Image.open(self.rows[i]['image_path']) as image:
            gray=np.asarray(image.convert('L').resize((128,128),Image.Resampling.BICUBIC)).copy()
        return torch.from_numpy(gray)[None]


def main():
    p=argparse.ArgumentParser();p.add_argument('--csv',required=True);p.add_argument('--out',required=True)
    p.add_argument('--batch',type=int,default=192);a=p.parse_args()
    torch.set_num_threads(8);torch.backends.cudnn.benchmark=True
    rows=read(a.csv);out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
    digest=hashlib.sha256(Path(a.csv).read_bytes()).hexdigest()
    if out.exists():
        manifest=json.loads(out.with_suffix('.json').read_text())
        if manifest['csv_sha256']!=digest:raise ValueError('Existing cache CSV identity changed')
        print('CACHE_REUSED',len(rows),flush=True);return
    model=AutoModel.from_pretrained('facebook/dinov2-base',local_files_only=True).cuda().eval().requires_grad_(False)
    mean=torch.tensor([.485,.456,.406],device='cuda')[None,:,None,None]
    std=torch.tensor([.229,.224,.225],device='cuda')[None,:,None,None]
    batches=DataLoader(Images(rows),batch_size=a.batch,num_workers=8,pin_memory=True,persistent_workers=True)
    features=np.empty((len(rows),768),np.float32);offset=0;start=time.monotonic()
    with torch.inference_mode():
        for gray in batches:
            x=gray.cuda(non_blocking=True).float()/255
            x=F.interpolate(x,(224,224),mode='bicubic',align_corners=False,antialias=True).clamp_(0,1).expand(-1,3,-1,-1)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                feat=model(pixel_values=(x-mean)/std).last_hidden_state[:,0]
            n=len(gray);features[offset:offset+n]=F.normalize(feat.float(),dim=-1).cpu().numpy();offset+=n
            if offset % (a.batch*25)==0:print('CACHE',offset,len(rows),'img_s',round(offset/(time.monotonic()-start),1),flush=True)
    if not np.isfinite(features).all():raise ValueError('Nonfinite cache')
    np.savez(out,ids=np.array([int(r['img_id']) for r in rows]),features=features,
        writers=np.array([int(r['style_writer_id']) for r in rows]),chars=np.array([ord(r['character']) for r in rows]),
        scripts=np.array([{'楷':0,'行':1,'隶':2}[r['script']] for r in rows]))
    out.with_suffix('.json').write_text(json.dumps(dict(model='facebook/dinov2-base',csv_sha256=digest,
        rows=len(rows),feature='L2 normalized final CLS',preprocess='gray bicubic128 -> antialiased224, ImageNet normalize',
        seconds=time.monotonic()-start,training_only=True),indent=2))
    print('CACHE_DONE',len(rows),flush=True)


if __name__=='__main__':main()
