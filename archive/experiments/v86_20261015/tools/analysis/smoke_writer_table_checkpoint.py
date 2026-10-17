"""One real-checkpoint forward/backward checks writer/null/script contracts."""
import argparse
import sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.eval.model_io import load_model_from_ckpt
from src.train.writer_table_tuning import configure_writer_table,split_writer_optimizer_group


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--ckpt',required=True)
    cli=parser.parse_args();torch.set_num_threads(4)
    model,args=load_model_from_ckpt(cli.ckpt,device='cuda',use_ema=True)
    for key in ('y_callig_embedder','y_script_embedder','y_char_embedder'):
        mod=getattr(model,key,None)
        if mod is not None:mod.requires_grad_(False)
    args.finetune_callig_table=True;args.freeze_triple_tables=True
    args.callig_table_lr=2e-6;args.callig_table_anchor_weight=.01
    args.callig_table_anchor_file='assets/v86A_v84A_writer_anchor.pt'
    st=configure_writer_table(model,args)
    before=st.weight.detach().clone()
    scripts={k:v.detach().clone() for k,v in model.y_script_embedder.state_dict().items()}
    assert torch.equal(before[:10],st.reference.to(before.device))
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-5,weight_decay=.02)
    split_writer_optimizer_group(opt,st,2e-6)
    x=torch.randn(2,4,32,32,device='cuda')
    # Use real nonblank std-soft masks rather than a g=0 out-of-distribution probe.
    from src.eval.in_mem_eval import _get_cache
    cache=_get_cache('exp-std/csv/eval200_fixed.csv',187,None,'',args)
    g=cache['skels_latent'][:1].repeat(2,1,1,1).cuda()
    with torch.autocast('cuda',dtype=torch.bfloat16):
        out=model(x,torch.tensor([500.,500.],device='cuda'),
                  torch.tensor([0,1],device='cuda'),torch.zeros(2,dtype=torch.long,device='cuda'),
                  g=g,y_script=torch.zeros(2,dtype=torch.long,device='cuda'))
        loss=out.float().square().mean()+.01*st.anchor_loss()
    loss.backward()
    assert st.weight.grad[:10].norm()>0
    assert torch.count_nonzero(st.weight.grad[10:])==0
    opt.step();st.restore_null()
    assert not torch.equal(before[:10],st.weight[:10])
    assert torch.equal(before[10:],st.weight[10:])
    assert all(torch.equal(v,model.y_script_embedder.state_dict()[k]) for k,v in scripts.items())
    print('REAL_CHECKPOINT_WRITER_STEP_OK grad=',st.weight.grad[:10].norm().item(),flush=True)

if __name__=='__main__':main()
