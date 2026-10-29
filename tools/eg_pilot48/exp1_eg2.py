"""实验1.1: train E-G2, an independent writer encoder (ResNet18, own recipe),
char-grouped validation, same-script cross-char retrieval. Acceptance >=90 AUC.

Independent of E-G: fresh torchvision resnet18, gray-input, own augmentation
(affine+noise via torch), Adam. Validation reuses eval_style_v2 metric rules.
"""
import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .common import save_json,REAL_CSV,PILOT_ROOT
from tools.eval_unpaired.protocol import reference_bank


def load_rows():
    rows=[r for r in csv.DictReader(open(REAL_CSV,encoding='utf-8-sig'))]
    seen=set();out=[]
    for r in rows:
        k=r.get('pixel_sha256')
        if k and k in seen:continue
        if k:seen.add(k)
        out.append(r)
    return out


def char_folds(rows,n=5,salt='eg2-v1'):
    import hashlib
    folds={}
    for r in rows:
        c=r['character']
        folds.setdefault(c,int(hashlib.md5((salt+c).encode()).hexdigest(),16)%n)
    return folds


def build_model(device):
    import torchvision
    net=torchvision.models.resnet18(weights=None)
    net.conv1=torch.nn.Conv2d(1,64,7,stride=2,padding=3,bias=False)
    net.fc=torch.nn.Identity()
    return net.to(device)


def batch_tensors(rows,device,train=False):
    from PIL import Image
    import numpy as np
    imgs=[];labels=[]
    for r in rows:
        im=Image.open(r['image_path']).convert('L')
        arr=np.asarray(im,dtype=np.float32)/255.
        if train:
            if arr.shape!=(256,256):arr=np.asarray(im.resize((256,256),Image.Resampling.BICUBIC),dtype=np.float32)/255.
            # cheap augmentation: shift/rotate/scale via affine grid
            th=torch.tensor([[1,0,0],[0,1,0]],dtype=torch.float32).unsqueeze(0)
            arr_t=torch.from_numpy(arr)[None,None]
            ang=float(np.random.uniform(-0.06,0.06));sc=float(np.random.uniform(0.92,1.05))
            ca,sa=np.cos(ang)*sc,np.sin(ang)*sc
            th=torch.tensor([[ca,-sa,0],[sa,ca,0]],dtype=torch.float32).unsqueeze(0)
            grid=F.affine_grid(th,arr_t.size(),align_corners=False)
            arr=F.grid_sample(arr_t,grid,align_corners=False)[0,0].numpy()
            arr=np.clip(arr+np.random.normal(0,0.02,arr.shape),0,1).astype(np.float32)
        else:
            if arr.shape!=(256,256):arr=np.asarray(im.resize((256,256),Image.Resampling.BICUBIC),dtype=np.float32)/255.
        imgs.append(arr);labels.append(int(r['calligrapher_id']))
    x=torch.from_numpy(np.stack(imgs))[:,None].to(device)
    return x,torch.tensor(labels,device=device)


def embed(net,x):
    f=net(x)
    return F.normalize(f,dim=1)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--device',default='cuda')
    p.add_argument('--time-cap',type=float,default=5400.)
    p.add_argument('--out',default=PILOT_ROOT/'eg2_real_validation.json')
    a=p.parse_args()
    t0=time.monotonic()
    rows=load_rows()
    folds=char_folds(rows)
    train_rows=[r for r in rows if folds[r['character']]!=0]
    valid_rows=[r for r in rows if folds[r['character']]==0 and not (r.get('aug') or '').strip()]
    device=a.device
    net=build_model(device)
    opt=torch.optim.Adam(net.parameters(),lr=3e-4)
    rng=np.random.default_rng(7)
    step=0;epoch=0
    while time.monotonic()-t0<a.time_cap and epoch<30:
        net.train()
        rng.shuffle(train_rows)
        for start in range(0,len(train_rows)-256,256):
            x,y=batch_tensors(train_rows[start:start+256],device,train=True)
            logits=net(x)
            loss=F.cross_entropy(logits,y)
            opt.zero_grad();loss.backward();opt.step();step+=1
        epoch+=1
        print('epoch',epoch,'loss',round(float(loss),4),flush=True)
    net.eval()
    # validation: cross-char same-script retrieval on valid chars, refs from train chars
    with torch.no_grad():
        vq=[];vlab=[]
        for start in range(0,len(valid_rows),64):
            x,y=batch_tensors(valid_rows[start:start+64],device)
            vq.append(embed(net,x).cpu().numpy());vlab+= [int(r['calligrapher_id']) for r in valid_rows[start:start+64]]
        vq=np.concatenate(vq);vlab=np.array(vlab)
        vscr=np.array([r['script'] for r in valid_rows])
        vchar=np.array([r['character'] for r in valid_rows])
    pos=[];neg=[]
    for i in range(len(vq)):
        same_script=np.where((vscr==vscr[i])&(vchar!=vchar[i])&(np.arange(len(vq))!=i))[0]
        by={}
        for j in same_script:by.setdefault(vlab[j],[]).append(j)
        tgt=vlab[i]
        if tgt not in by:continue
        pos.append(vq[i]@vq[by[tgt][:64]].mean(0) if False else np.sort(vq[i]@np.array(vq[by[tgt][:64]]).T)[-5:].mean())
        neg+= [np.sort(vq[i]@np.array(vq[by[w][:64]]).T)[-5:].mean() for w in by if w!=tgt]
    pos=np.array(pos);neg=np.array(neg)
    diff=pos[:,None]-neg[None,:]
    auc=float((diff>0).mean()+0.5*(diff==0).mean())
    TMP=Path('assets/eg_pilot48_tmp');TMP.mkdir(parents=True,exist_ok=True)
    torch.save(dict(net=net.state_dict()),TMP/'eg2.pt')
    res=dict(auc=round(100*auc,2),n_queries=len(pos),n_valid_chars=len(set(vchar)),
        epochs=epoch,steps=step,train_rows=len(train_rows),valid_rows=len(valid_rows),
        accepted=bool(auc>=0.90),seconds=round(time.monotonic()-t0),weights=str(TMP/'eg2.pt'))
    save_json(a.out,res)
    print(json.dumps(res))


if __name__=='__main__':main()
