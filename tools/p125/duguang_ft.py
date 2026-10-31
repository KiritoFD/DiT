"""读光 OCR fine-tuning on real calligraphy (single-char images).

Baseline: 0% Top-1 on real calligraphy (predictions are valid Chinese but wrong).
Fine-tune the ViTSTR + head on 144k real calligraphy images with character labels.
CNN backbone: lower LR (1e-5). ViTSTR + head: normal LR (3e-4).
"""
import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from PIL import Image

sys.path.insert(0,'.')
os.chdir('/root/Workspace/xy/DiT')
from tools.p125.duguang_model import DuguangOCR

D="/root/.cache/modelscope/models/iic--cv_convnextTiny_ocr-recognition-handwritten_damo/snapshots/master"


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--steps',type=int,default=5000)
    p.add_argument('--batch',type=int,default=64)
    p.add_argument('--lr-vitstr',type=float,default=3e-4)
    p.add_argument('--lr-cnn',type=float,default=1e-5)
    p.add_argument('--out',default='assets/p125_20261030/ocr/duguang_ft.pt')
    a=p.parse_args()
    torch.manual_seed(10311031)
    model=DuguangOCR(vocab_size=7644)
    ck=torch.load(os.path.join(D,"pytorch_model.pt"),map_location="cpu",weights_only=False)
    model.load_state_dict(ck,strict=False)
    model=model.cuda()
    vocab=[l.strip() for l in open(os.path.join(D,"vocab.txt"),encoding="utf-8")]
    char2idx={c:i+1 for i,c in enumerate(vocab)}  # +1: index0=blank
    # targets use vocab.txt index space (model CTC head outputs over vocab.txt)
    rows=[r for r in csv.DictReader(open("assets/p125_20261030/ocr/ocr_train.csv",encoding="utf-8-sig"))
          if r['character'] in char2idx]
    print(f"training rows: {len(rows)} (chars in vocab)",flush=True)
    cnn_params=[];vit_params=[]
    for n,p_ in model.named_parameters():
        if n.startswith("cnn_model."):cnn_params.append(p_)
        else:vit_params.append(p_)
    opt=torch.optim.AdamW([
        dict(params=cnn_params,lr=a.lr_cnn),
        dict(params=vit_params,lr=a.lr_vitstr)],weight_decay=1e-4)
    rng=random.Random(10311031)
    model.train()
    for step in range(1,a.steps+1):
        idx=rng.sample(range(len(rows)),min(a.batch,len(rows)))
        images=[];targets=[]
        for i in idx:
            r=rows[i]
            img=Image.open(r['image_path']).convert('L')
            w,h=img.size
            tw=max(32,min(804,int(32*w/h)))
            img=img.resize((tw,32),Image.Resampling.BICUBIC)
            x=np.asarray(img,dtype=np.float32)[None,None]/255
            x=(x-0.5)/0.5
            images.append(torch.from_numpy(x))
            targets.append(char2idx[r['character']])  # char2idx maps to vocab.txt positions
        # pad to same width
        maxw=max(im.shape[-1] for im in images)
        padded=[]
        for im in images:
            diff=maxw-im.shape[-1]
            if diff>0:im=F.pad(im,(0,diff),mode='constant',value=-0.5)
            padded.append(im)
        x=torch.cat(padded).cuda()
        t=torch.tensor(targets).cuda()
        logits=model(x)  # (B,T,7644)
        # CTC: collapse to per-sample argmax timestep, use CE on that timestep
        B_=logits.shape[0]
        conf=logits.max(-1).values  # (B,T) max class prob per timestep
        peak=conf.argmax(-1)  # (B,) peak timestep
        peak_logits=logits[torch.arange(B_),peak]  # (B,7644)
        loss=F.cross_entropy(peak_logits,t)
        opt.zero_grad();loss.backward();opt.step()
        if step%500==0:
            acc=float((peak_logits.argmax(-1)==t).float().mean())
            print(f"step {step} loss {loss.item():.4f} batch_acc {acc:.3f}",flush=True)
    torch.save({'model':model.state_dict(),'vocab_size':7644,
        'steps':a.steps},a.out)
    print('saved',a.out,flush=True)


if __name__=='__main__':main()
