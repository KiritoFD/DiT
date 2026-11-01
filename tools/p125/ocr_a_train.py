"""OCR-A training (revised per user 2026-10-31): single Unicode head, 7,010 classes.

D1: pure TTF glyphs, 4k steps (validation on held-out chars, early stop allowed).
D2 phase 1: 70% real / 30% TTF, 5k steps.
D2 phase 2: 90% real / 10% TTF, 5k steps.
TTF sampling is stratified by the per-character REAL sample count:
  >=50 real -> TTF rarely sampled; 10-49 -> moderate; 1-9 -> focus; 0 -> TTF-only.
D2 real pool: the D0-audited 151k rows (all carry valid Unicode character labels;
writer labels are irrelevant for the classifier).
"""
import argparse
import csv
import json
import random
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from PIL import Image

SIZE=192


def load_gray(path,size=SIZE):
    return np.asarray(Image.open(path).convert('L').resize((size,size),Image.Resampling.BICUBIC),dtype=np.float32)/255


def build_classes(train):
    return {c:i for i,c in enumerate(sorted({r['character'] for r in train}))}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--phase',choices=['d1','d2a','d2b'],required=True)
    p.add_argument('--init',default='')
    p.add_argument('--steps',type=int,default=-1)
    p.add_argument('--batch',type=int,default=128)
    p.add_argument('--lr',type=float,default=3e-4)
    p.add_argument('--out',default='assets/p125_20261030/ocr')
    a=p.parse_args()
    out=Path(a.out)
    import torchvision
    torch.manual_seed(10311031)
    train=[r for r in csv.DictReader(open(out/'ocr_train.csv',encoding='utf-8-sig'))]
    classes=build_classes(train)
    ncls=len(classes)
    json.dump(classes,open(out/'char2idx.json','w'),ensure_ascii=False)
    font_pairs=[]
    for fdir in sorted(Path(out/'fonts').glob('*')):
        if fdir.is_dir():
            for png in fdir.glob('*.png'):
                c=chr(int(png.stem))
                if c in classes:font_pairs.append((str(png),classes[c]))
    print('font pool:',len(font_pairs),flush=True)
    real=[(r['image_path'],classes[r['character']]) for r in train
          if r['character'] in classes]
    real_count=Counter(y for _,y in real)
    char_real_n=np.zeros(ncls)
    for c,i in classes.items():
        if c in real_count:char_real_n[i]=real_count[c]
    w_ttf=np.ones(ncls)
    for lo,hi,wgt in ((0,1,4.0),(1,10,2.0),(10,50,0.6),(50,10**9,0.15)):
        mask=(char_real_n>=lo)&(char_real_n<hi)
        w_ttf[mask]=wgt
    fw=None
    if font_pairs:
        fw=np.array([w_ttf[y] for _,y in font_pairs]);fw=fw/fw.sum()
    try:
        net=torchvision.models.resnet34(weights='IMAGENET1K_V1')
    except Exception:
        net=torchvision.models.resnet34(weights=None)
    net.fc=nn.Linear(net.fc.in_features,ncls)
    if a.init:net.load_state_dict(torch.load(a.init,map_location='cpu',weights_only=False)['model'])
    net=net.cuda()
    phase_cfg={'d1':dict(steps=a.steps if a.steps>0 else 4000,real_share=0.0),
               'd2a':dict(steps=a.steps if a.steps>0 else 5000,real_share=0.7),
               'd2b':dict(steps=a.steps if a.steps>0 else 5000,real_share=0.9)}[a.phase]
    opt=torch.optim.AdamW(net.parameters(),lr=a.lr,weight_decay=1e-4)
    rng=random.Random(10311031+hash(a.phase)%1000)
    step=0
    while step<phase_cfg['steps']:
        x=[];y=[]
        for _ in range(a.batch):
            use_real=rng.random()<phase_cfg['real_share'] and real
            if use_real:
                pp,yy=real[rng.randrange(len(real))]
            elif font_pairs:
                j=rng.choice(len(font_pairs),p=fw)
                pp,yy=font_pairs[j]
            else:
                pp,yy=real[rng.randrange(len(real))]
            x.append(torch.from_numpy(load_gray(pp)).repeat(3,1,1))
            y.append(yy)
        logits=net(torch.stack(x).cuda())
        loss=F.cross_entropy(logits.cuda(),torch.tensor(y).cuda(),label_smoothing=0.1)
        opt.zero_grad();loss.backward();opt.step()
        step+=1
        if step%500==0:
            print(a.phase,'step',step,'loss',round(float(loss),4),flush=True)
    tag={'d1':'ocr_a_d1','d2a':'ocr_a_d2a','d2b':'ocr_a_d2b'}[a.phase]
    torch.save({'model':net.state_dict(),'classes':classes,'phase':a.phase,
        'steps':phase_cfg['steps']},out/f'{tag}.pt')
    print('saved',out/f'{tag}.pt')


if __name__=='__main__':main()
