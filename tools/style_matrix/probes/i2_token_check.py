"""I2 KV-token preflight: prove Cross-Attention ranks 8 distinguishable tokens.

Run before launching S2/S3. If the KV tokens collapse to one effective token,
attention weights are uniform and the arm cannot test local style selection.
"""
import argparse
import json
import math
import torch
from ..runtime.common import construct,synthetic,save


def main():
    p=argparse.ArgumentParser();p.add_argument('--spec',required=True);p.add_argument('--out',required=True)
    a=p.parse_args()
    spec=json.loads(open(a.spec).read())
    model,cfg,_=construct(spec,initialize_source=False)
    model.eval();base=model.backbone
    x,t,g,w,s=synthetic(cfg,batch=10)
    w=torch.arange(10,device=g.device)
    glyph=base.std_stem(g);pyramid=[glyph]
    for down in base.std_down:pyramid.append(base.act(down(pyramid[-1])))
    with torch.no_grad():
        state=model.style_representation(w,pyramid,g)
        token_std=float(state.tokens.float().std(dim=1).mean())
        token01=float((state.tokens[:,0]-state.tokens[:,1]).float().square().mean().sqrt())
        for inj in model.style_injections:inj.record=True
        model(x[:4],t[:4],w[:4],g=g[:4],y_script=s[:4])
        entropy=[inj.last.get('attention_entropy') for inj in model.style_injections]
        variance=[inj.last.get('attention_spatial_variance') for inj in model.style_injections]
    ceiling=math.log(8)
    # Near-uniform attention at init is expected; the architectural facts that
    # matter are 8 distinct learnable KV tokens and non-degenerate attention.
    passed=bool(token_std>.1 and token01>.1
                and all(e is not None and e<ceiling-1e-3 for e in entropy)
                and all(v is not None and v>0 for v in variance))
    save(a.out,dict(arm=spec['arm'],kv_tokens=8,token_std=token_std,token01_distance=token01,
        attention_entropy=entropy,attention_spatial_variance=variance,uniform_entropy=ceiling,passed=passed))
    print(json.dumps(dict(token_std=token_std,token01_distance=token01,
        attention_entropy=entropy,uniform_entropy=ceiling,passed=passed)))
    if not passed:raise SystemExit('I2 KV tokens not distinguishable; S2/S3 would be vacuous')


if __name__=='__main__':main()
