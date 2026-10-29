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
    return torch.stack(losses).sum()


class PairBudget:
    """Calibrate actual parameter-gradient ratio after FM backward, without replay."""
    def __init__(self,state=None,target=.25):
        self.state=state or {};self.target=target

    def backward(self,model,z,g,w,s,batch,rows,fraction,micro,step):
        parameters=[p for p in model.parameters() if p.requires_grad]
        probe='weight' not in self.state or step%500==0
        if not probe:
            loss=backward_pair(model,z,g,w,s,batch,rows,fraction,self.state['weight'],micro)
            return loss
        fm=[None if p.grad is None else p.grad.detach().clone() for p in parameters]
        fm_norm=torch.stack([v.float().square().sum() for v in fm if v is not None]).sum().sqrt()
        for p in parameters:p.grad=None
        loss=backward_pair(model,z,g,w,s,batch,rows,fraction,1.,micro)
        pair_norm=torch.stack([p.grad.float().square().sum() for p in parameters if p.grad is not None]).sum().sqrt()
        # Explicit low-frequency synchronization boundary, once per 500 steps.
        if not torch.isfinite(pair_norm) or pair_norm<1e-10:raise ValueError('Pair gradient is absent or nonfinite')
        weight=float(self.target*fm_norm/pair_norm)
        ratio=float(weight*pair_norm/fm_norm.clamp_min(1e-12))
        if not .20<=ratio<=.30:raise ValueError(f'Pair gradient budget not met: {ratio}')
        for p,grad in zip(parameters,fm):
            if p.grad is not None:p.grad.mul_(weight)
            if grad is not None:
                if p.grad is None:p.grad=grad
                else:p.grad.add_(grad)
        self.state=dict(weight=weight,ratio=ratio,fm_norm=float(fm_norm),pair_norm=float(pair_norm),step=step)
        return loss
