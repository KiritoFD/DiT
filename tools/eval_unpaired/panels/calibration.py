"""TTF and perturbation controls; export a review sheet, not an aesthetic ground truth."""
import argparse
import csv
import fcntl
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFilter
from tools.eval_unpaired.protocol import read,save
from tools.eval_unpaired.networks import Evaluators


def main():
    p=argparse.ArgumentParser();p.add_argument('--reference-rows',required=True);p.add_argument('--out',required=True)
    p.add_argument('--std-index',default='assets/generic48_kxl_top10_20261022/std_bbox64/index.csv');a=p.parse_args()
    root=Path(a.out);root.mkdir(parents=True,exist_ok=True)
    refs=read(a.reference_rows);rows=refs[::max(1,len(refs)//24)][:24]
    from src.utils.std_soft_glyph import render_base_mask,FontFace
    from tools.style_matrix.runtime.serial import gpu_idle
    lookup={(r['script_id'],r['character']):r for r in csv.DictReader(open(a.std_index))}
    mapping={'楷':'0','行':'3','隶':'4'};controls=[]
    for i,row in enumerate(rows):
        source=lookup[(mapping[row['script']],row['character'])]
        ink=render_base_mask(row['character'],FontFace(source['font_path'],int(source['font_face'])),output_size=256)
        path=root/f'ttf{i}.png';Image.fromarray(255-np.asarray(ink)).convert('RGB').save(path);controls.append(path)
    import torch
    torch.set_num_threads(4)
    with open('/tmp/dit_gpu0.lock','a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);gpu_idle(2107710)
        nets=Evaluators();real=nets.extract([r['image_path'] for r in rows]);ttf=nets.extract(controls)
        metrics={key:dict(real_mean=float(real[key].mean()),ttf_mean=float(ttf[key].mean()),
            real_preferred=float(np.mean(real[key]>ttf[key]))) for key in ttf if not key.startswith('E-')}
        save(root/'ttf_controls.json',dict(metrics=metrics,rows=rows,meaning='matched-character/script TTF; human preference unknown; no primary metric promoted'))
    canvas=Image.new('RGB',(5*128,8*128+24),'#eee');draw=ImageDraw.Draw(canvas)
    for j,name in enumerate(['Real','Blur','Thick','Erased band','TTF']):draw.text((j*128+3,5),name,fill='black')
    for i,row in enumerate(rows[:8]):
        im=Image.open(row['image_path']).convert('RGB').resize((256,256))
        thick=im.filter(ImageFilter.MinFilter(5));broken=im.copy();ImageDraw.Draw(broken).rectangle((0,118,255,125),fill='white')
        for j,img in enumerate([im,im.filter(ImageFilter.GaussianBlur(2)),thick,broken,Image.open(controls[i])]):
            canvas.paste(img.resize((126,126)),(j*128,i*128+24))
    canvas.save(root/'calibration_sheet.jpg',quality=90)


if __name__=='__main__':main()
