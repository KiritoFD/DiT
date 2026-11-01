"""Extended 读光 fine-tuning: continue from 5k checkpoint to 20k+ with cosine LR.
Evaluates Real-Test Top-1 at each checkpoint to track convergence."""
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
    p.add_argument('--init',default='assets/p125_20261030/ocr/duguang_ft.pt')
    p.add_argument('--steps',type=int,default=15000)
    p.add_argument('--batch',type=int,default=128)
    p.add_argument('--lr',type=float,default=1e-4)
    p.add_argument('--eval-every',type=int,default=5000)
    p.add_argument('--out',default='assets/p125_20261030/ocr/duguang_ft_ext.pt')
    a=p.parse_args()
    torch.manual_seed(10311032)
    model=DuguangOCR(vocab_size=7644)
    ck=torch.load(a.init,map_location="cpu",weights_only=False)
    model.load_state_dict(ck["model"])
    model=model.cuda()
    vocab=[l.strip() for l in open(os.path.join(D,"vocab.txt"),encoding="utf-8")]
    char2idx={c:i+1 for i,c in enumerate(vocab)}
    rows=[r for r in csv.DictReader(open("assets/p125_20261030/ocr/ocr_train.csv",encoding="utf-8-sig"))
          if r['character'] in char2idx]
    print(f"training rows: {len(rows)}",flush=True)
    # Real-Test
    rt=[r for r in csv.DictReader(open("assets/p125_20261030/ocr/ocr_realtest.csv",encoding="utf-8-sig"))
        if r['character'] in char2idx]
    rt_imgs=[];rt_targets=[]
    for r in rt:
        img=Image.open(r['image_path']).convert('L')
        w,h=img.size;tw=max(32,min(320,int(32*w/h)))
        img=img.resize((tw,32),Image.Resampling.BICUBIC)
        x=np.asarray(img,dtype=np.float32)[None,None]/255
        x=(x-0.5)/0.5
        rt_imgs.append(torch.from_numpy(x))
        rt_targets.append(char2idx[r['character']])
    rt_x=torch.cat(rt_imgs).cuda()
    rt_t=torch.tensor(rt_targets).cuda()
    cnn_p=[];vit_p=[]
    for n,p_ in model.named_parameters():
        if n.startswith("cnn_model."):cnn_p.append(p_)
        else:vit_p.append(p_)
    opt=torch.optim.AdamW([
        dict(params=cnn_p,lr=a.lr*0.3),
        dict(params=vit_p,lr=a.lr)],weight_decay=1e-4)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=a.steps,eta_min=a.lr*0.05)
    rng=random.Random(10311032)
    model.train()
    log=open("/tmp/duguang_ft2.log","a")
    def evaluate():
        model.eval();correct=0
        with torch.no_grad():
            for i in range(0,len(rt_x),128):
                logits=model(rt_x[i:i+128])
                conf=logits.max(-1).values
                peak=conf.argmax(-1)
                peak_logits=logits[torch.arange(len(peak)),peak]
                pred=peak_logits.argmax(-1)
                correct+=int((pred==rt_t[i:i+128]).sum())
        model.train()
        return round(100*correct/len(rt_t),2)
    base_acc=evaluate()
    print(f"baseline Real-Test Top-1: {base_acc}%",flush=True)
    for step in range(1,a.steps+1):
        idx=rng.sample(range(len(rows)),min(a.batch,len(rows)))
        images=[];targets=[]
        for i in idx:
            r=rows[i]
            img=Image.open(r['image_path']).convert('L')
            w,h=img.size;tw=max(32,min(320,int(32*w/h)))
            img=img.resize((tw,32),Image.Resampling.BICUBIC)
            x=np.asarray(img,dtype=np.float32)[None,None]/255
            x=(x-0.5)/0.5
            images.append(torch.from_numpy(x))
            targets.append(char2idx[r['character']])
        maxw=max(im.shape[-1] for im in images)
        padded=[]
        for im in images:
            diff=maxw-im.shape[-1]
            if diff>0:im=F.pad(im,(0,diff),mode='constant',value=-0.5)
            padded.append(im)
        x=torch.cat(padded).cuda()
        t=torch.tensor(targets).cuda()
        logits=model(x)
        B_=logits.shape[0]
        conf=logits.max(-1).values
        peak=conf.argmax(-1)
        peak_logits=logits[torch.arange(B_),peak]
        loss=F.cross_entropy(peak_logits,t)
        opt.zero_grad();loss.backward();opt.step();sched.step()
        if step%500==0:
            acc=evaluate()
            print(f"step {step} loss {loss.item():.4f} RealTest {acc}%  LR {sched.get_last_lr()[0]:.6f}",flush=True)
    torch.save({'model':model.state_dict(),'steps':a.steps},a.out)
    final=evaluate()
    print(f"FINAL Real-Test Top-1: {final}%",flush=True)


if __name__=='__main__':main()
