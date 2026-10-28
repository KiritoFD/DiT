"""Same-noise t=1 conditional velocity difference; training GT never conditions FM."""
import torch
from torch.nn import functional as F


def backward_pair(model,z,g,w,s,batch,rows,fraction,weight,micro):
    n=2*int(len(z)*fraction/2)
    if n<2:raise ValueError('Pair loss needs adjacent paired endpoints')
    ids=batch['img_id'][:n].tolist()
    for i in range(0,n,2):
        a,b=rows[int(ids[i])],rows[int(ids[i+1])]
        if (a['character'],a['script'])!=(b['character'],b['script']) or a['calligrapher']==b['calligrapher']:
            raise ValueError('Invalid same-char, same-script, different-writer pair')
    losses=[]
    # Isolate auxiliary RNG so FM and Pair branches retain identical FM draws.
    with torch.random.fork_rng(devices=[z.device.index or 0]):
        for begin in range(0,n,2*max(1,micro//2)):
            stop=min(n,begin+2*max(1,micro//2));a=slice(begin,stop,2);b=slice(begin+1,stop,2)
            noise=torch.randn_like(z[a]);tt=z.new_full((len(noise),),1000.)
            # Both endpoints must use exactly the same STD, Script and noise.
            with torch.autocast('cuda',dtype=torch.bfloat16):
                va=model(noise,tt,w[a],g=g[a],y_script=s[a],condition_dropped=True)
                vb=model(noise,tt,w[b],g=g[a],y_script=s[a],condition_dropped=True)
            target=(z[b]-z[a]).detach()
            scale=target.square().flatten(1).mean(1).sqrt().clamp_min(.05)[:,None,None,None]
            loss=F.smooth_l1_loss((va.float()-vb.float())/scale,target/scale)*(stop-begin)/n
            (weight*loss).backward();losses.append(loss.detach())
    return float(torch.stack(losses).sum())
