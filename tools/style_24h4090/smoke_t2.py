"""One real P2 forward/backward, plus identical-input counterfactual check."""
import csv
import json
import torch
from src.unet.io import load
from src.loss.flow_matching import FlowMatching
from .t2_rank import WriterRankAdapter
from .run_ablation import P2_START


def main():
    torch.manual_seed(90)
    model,cfg,_=load(P2_START,device='cuda')
    rows=list(csv.DictReader(open(cfg.data_csv,encoding='utf-8-sig')))[:8]
    wm=json.load(open(cfg.callig_remap_json)); sm=json.load(open(cfg.font_remap_json))
    w=torch.tensor([wm[r['calligrapher_id']] for r in rows],device='cuda')
    s=torch.tensor([sm[r['script_id']] for r in rows],device='cuda')
    z=torch.randn(8,4,32,32,device='cuda'); t=torch.full((8,),.75,device='cuda')
    kw=dict(y_callig=w,y_script=s,g=torch.rand(8,1,64,64,device='cuda'),condition_dropped=True)
    fm=FlowMatching(); adapter=WriterRankAdapter(cfg); adapter.model=model
    with torch.autocast('cuda',dtype=torch.bfloat16):
        result=fm.training_losses(model,z,t,kw)
        xt=z+t[:,None,None,None]*result['v_target']
        repeat=model(xt,t*1000.,**kw)
        err=(repeat.float()-result['v_target']).square().mean((1,2,3))
        torch.testing.assert_close(err,result['loss'],atol=1e-5,rtol=1e-4)
        loss=adapter.objective(result,z,t,kw,dict(img_id=torch.tensor([int(r['img_id']) for r in rows])),0)
    loss.backward()
    grads=[p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert sum(float(g.square().sum()) for g in grads)>0
    print('T2_SMOKE_PASS same_interpolant same_time identical_writer_MSE finite_nonzero_backward',float(loss),flush=True)


if __name__=='__main__': main()
