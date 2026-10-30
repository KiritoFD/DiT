"""P1 writer-table initialization: Top10 slots = mean of pretrained generic
writer rows + small Gaussian noise (seed recorded). Writes a derived P0
checkpoint used as P1's source; optimizer/EMA state preserved unchanged
(P1 rebuilds stage state anyway).
"""
import argparse
import hashlib
from pathlib import Path
import torch


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',required=True);p.add_argument('--out',required=True)
    p.add_argument('--n-top10',type=int,default=10)
    p.add_argument('--generic-lo',type=int,default=10)
    p.add_argument('--generic-hi',type=int,default=47)  # exclusive; null slot excluded
    p.add_argument('--seed',type=int,default=125125)
    p.add_argument('--noise-std',type=float,default=0.02)
    a=p.parse_args()
    ck=torch.load(a.source,map_location='cpu',weights_only=False)
    key='style_representation.table.weight'
    table=ck['ema'][key]
    generic=table[a.generic_lo:a.generic_hi]
    mean=generic.mean(0)
    g=torch.Generator().manual_seed(a.seed)
    noise=torch.randn(a.n_top10,table.shape[1],generator=g)*a.noise_std
    table=table.clone()
    table[:a.n_top10]=(mean+noise).to(table.dtype)
    ck['ema'][key]=table
    if 'delta' in ck and key in ck['delta']:
        ck['delta'][key]=table.clone()
    torch.save(ck,a.out)
    json_out(source=a.source,source_sha256=sha(a.source),out=a.out,
        table_rows=table.shape[0],generic_slots=list(range(a.generic_lo,a.generic_hi)),
        noise_std=a.noise_std,seed=a.seed)


def json_out(**kw):
    import json
    print(json.dumps(kw,ensure_ascii=False))


if __name__=='__main__':main()
