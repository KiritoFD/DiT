"""OCR-A acceptance: Real-Test accuracy (by script/writer), Corruption-Test
detection recall, and the abstention threshold calibration.
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from torch import nn
from PIL import Image

from tools.p125.ocr_a_train import load_gray,SIZE
from tools.p125.ocr_acceptance import corruptions


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--ckpt',default='assets/p125_20261030/ocr/ocr_a_d2.pt')
    p.add_argument('--realtest',default='assets/p125_20261030/ocr/ocr_realtest.csv')
    p.add_argument('--out',default='docs/1031/ocr_a_acceptance.json')
    a=p.parse_args()
    import torchvision
    ck=torch.load(a.ckpt,map_location='cpu',weights_only=False)
    classes=ck['classes'];idx2c={v:k for k,v in classes.items()}
    net=torchvision.models.resnet34()
    net.fc=nn.Linear(net.fc.in_features,len(classes))
    net.load_state_dict(ck['model']);net=net.cuda().eval()
    rows=list(csv.DictReader(open(a.realtest,encoding='utf-8-sig')))
    rows=[r for r in rows if r['character'] in classes]
    probs=[];targets=[]
    with torch.no_grad():
        for i in range(0,len(rows),128):
            x=torch.stack([torch.from_numpy(load_gray(r['image_path'])).repeat(3,1,1)
                for r in rows[i:i+128]]).cuda()
            pr=F.softmax(net(x),dim=1).cpu().numpy()
            probs.append(pr);targets+=[classes[r['character']] for r in rows[i:i+128]]
    P=np.concatenate(probs);T=np.array(targets)
    top1=P.argmax(1)==T
    order=np.argsort(-P,axis=1)
    top5=np.array([T[i] in order[i,:5] for i in range(len(T))])
    margins=np.array([P[i,T[i]]-np.max(np.concatenate([P[i,:T[i]],P[i,T[i]+1:]])) for i in range(len(T))])
    by_script=defaultdict(list)
    for i,r in enumerate(rows):by_script[r['script']].append(int(top1[i]))
    scripts={s:round(100*np.mean(v),2) for s,v in by_script.items()}
    # corruption detection on a subset
    rng=np.random.default_rng(10311031)
    sub=rows[:300]
    det=defaultdict(list)
    for r in sub:
        g=np.asarray(Image.open(r['image_path']).convert('L').resize((SIZE,SIZE)),dtype=np.float32)/255
        for kind,img in corruptions(g,rng).items():
            x=torch.from_numpy(load_gray_arr(img)).repeat(3,1,1)[None].cuda()
            with torch.no_grad():pr=F.softmax(net(x),dim=1).cpu().numpy()[0]
            t=classes.get(r['character'])
            pred=int(pr.argmax())
            det[kind].append(int(pred!=t))
    det_res={k:round(float(np.mean(v)),3) for k,v in det.items()}
    # abstention: confidence threshold on Real-Test margins
    thr=float(np.percentile(margins,5))
    out=dict(n=len(T),real_top1=round(100*float(top1.mean()),2),
        real_top5=round(100*float(top5.mean()),2),
        real_margin_mean=round(float(margins.mean()),4),by_script=scripts,
        corruption_detection=det_res,abstain_margin_thr=round(thr,4),
        note='abstention: reject samples with target-margin < thr')
    Path(a.out).parent.mkdir(parents=True,exist_ok=True)
    Path(a.out).write_text(json.dumps(out,indent=1))
    print(json.dumps(out,ensure_ascii=False))


def load_gray_arr(img):
    import numpy as np
    return np.asarray(img,dtype=np.float32)/255


if __name__=='__main__':main()
