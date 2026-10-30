"""Exp E part 1: train E-G2, an INDEPENDENT writer verifier (different arch,
init and recipe from the frozen E-G), on Top10 real images with a strict
character-grouped split. Outputs eg2.pt + eg2_validation.json.
Passing bar: cross-char real AUC > 85 (E-G reference: 95.98).
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from PIL import Image

EG_MEAN=torch.tensor([.485,.456,.406]);EG_STD=torch.tensor([.229,.224,.225])


def load_img(path,size=128):
    img=Image.open(path).convert('RGB').resize((size,size),Image.Resampling.BICUBIC)
    a=np.asarray(img).copy()
    if np.random.random()<0.5:a=np.fliplr(a).copy()
    if np.random.random()<0.3:
        pad=np.random.randint(0,16)
        a=np.pad(a,((pad,pad),(pad,pad),(0,0)),mode='edge')[:size,:size] if pad else a
    return torch.from_numpy(a).permute(2,0,1).float()/255


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--csv',default='assets/u100_p1p2_20261025/p2_train.csv')
    p.add_argument('--epochs',type=int,default=4)
    p.add_argument('--out',default='assets/p125_20261030/eg2.pt')
    p.add_argument('--report',default='docs/1031/eg2_validation.json')
    a=p.parse_args()
    torch.manual_seed(20261031);np.random.seed(20261031)
    rows=[r for r in csv.DictReader(open(a.csv,encoding='utf-8-sig'))
          if not (r.get('aug') or '').strip()]
    # character-grouped split by hash
    def fold(c):return int(hashlib.md5(('eg2|'+c).encode()).hexdigest(),16)%5
    train=[r for r in rows if fold(r['character'])<4]
    valid=[r for r in rows if fold(r['character'])==4]
    print('train',len(train),'valid',len(valid),flush=True)
    import torchvision
    net=torchvision.models.resnet34(weights='IMAGENET1K_V1')
    net.fc=nn.Linear(net.fc.in_features,10)
    net=net.cuda()
    opt=torch.optim.AdamW(net.parameters(),lr=3e-4,weight_decay=1e-4)
    ls=lambda logits,y:F.cross_entropy(logits,y,label_smoothing=0.1)
    steps_per_epoch=len(train)//256
    g=torch.Generator().manual_seed(7)
    for ep in range(a.epochs):
        order=torch.randperm(len(train),generator=g).tolist()
        net.train();tot=0.
        for it in range(steps_per_epoch):
            idx=order[it*256:(it+1)*256]
            batch=torch.stack([load_img(train[i]['image_path']) for i in idx]).cuda()
            y=torch.tensor([int(train[i]['calligrapher_id']) for i in idx]).cuda()
            logits=net(batch)
            loss=ls(logits,y);opt.zero_grad();loss.backward();opt.step();tot+=float(loss)
        print('epoch',ep,'loss',round(tot/steps_per_epoch,4),flush=True)
    # cross-char validation: embeddings = penultimate
    net.eval();feats=nn.Sequential(*list(net.children())[:-1])
    emb_v=[];lab_v=[];ch_v=[]
    with torch.no_grad():
        for i in range(0,len(valid),256):
            batch=torch.stack([load_img(valid[j]['image_path']) for j in range(i,min(i+256,len(valid)))]).cuda()
            e=feats(batch).flatten(1)
            emb_v.append(e.cpu());lab_v+= [valid[j]['calligrapher_id'] for j in range(i,min(i+256,len(valid)))]
            ch_v+=[valid[j]['character'] for j in range(i,min(i+256,len(valid)))]
    E=torch.cat(emb_v);E=E/E.norm(dim=-1,keepdim=True)
    lab=np.array([int(x) for x in lab_v])
    pos=[];neg=[]
    for i in range(len(E)):
        same=np.where((lab==lab[i])&(np.array(ch_v)!=ch_v[i]))[0]
        diff=np.where(lab!=lab[i])[0]
        if not len(same):continue
        s=float(E[i]@E[same].T.max()) if len(same) else 0.
        s_top=float(np.sort(E[i]@E.T)[::-1][1:6].mean()) if len(E)>6 else 0.
        pos.append(s_top);neg+= [float(E[i]@E[j]) for j in diff[:64]]
    pos=np.array(pos);neg=np.array(neg[:len(pos)*8])
    diff=pos[:,None]-neg[None,:]
    auc=float((diff>0).mean()+0.5*(diff==0).mean())
    save=dict(real_crosschar_auc=round(100*auc,2),n_valid_queries=len(pos),
        arch='resnet34_imagenet_init_ls0.1_128px',passed=bool(auc>0.85))
    Path(a.report).parent.mkdir(parents=True,exist_ok=True)
    Path(a.report).write_text(json.dumps(save,indent=1))
    torch.save({'model':net.state_dict(),'validation':save},a.out)
    print(json.dumps(save))


if __name__=='__main__':main()
