from types import SimpleNamespace
import torch
from torch import nn
from src.model.components import LabelEmbedder
from src.train.writer_table_tuning import configure_writer_table, split_writer_optimizer_group


def make_model(tmp_path):
    torch.manual_seed(17)
    model=nn.Module()
    model.y_callig_embedder=LabelEmbedder(10,32,0.1)
    model.y_script_embedder=LabelEmbedder(3,16,0.1)
    model.y_char_embedder=LabelEmbedder(5,8,0.1)
    model.proj=nn.Linear(32,2)
    for name in ('y_callig_embedder','y_script_embedder','y_char_embedder'):
        getattr(model,name).requires_grad_(False)
    ref=tmp_path/'anchor.pt'
    torch.save({'writer_rows':model.y_callig_embedder.embedding_table.weight[:10].detach()},ref)
    args=SimpleNamespace(finetune_callig_table=True,freeze_triple_tables=True,
        triple_table_lr_scale=0.,optimizer='adamw',callig_table_lr=2e-6,
        callig_table_anchor_weight=.01,callig_table_anchor_file=str(ref))
    return model,args


def test_only_real_writer_rows_update_and_null_survives_momentum(tmp_path):
    m,a=make_model(tmp_path)
    original={k:v.detach().clone() for k,v in m.state_dict().items()}
    st=configure_writer_table(m,a)
    opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=1e-5,weight_decay=.9)
    split_writer_optimizer_group(opt,st,a.callig_table_lr)
    for step in range(3):
        opt.zero_grad()
        z=m.y_callig_embedder(torch.tensor([1,3,10]),False)
        (m.proj(z).square().sum()+.01*st.anchor_loss()).backward()
        assert torch.equal(st.weight.grad[10],torch.zeros(32))
        opt.step();st.restore_null()
        if step==0:
            # Simulate null-row momentum inherited from an older checkpoint.
            opt.state[st.weight]['exp_avg'][10]=1.
            opt.state[st.weight]['exp_avg_sq'][10]=1.
        assert torch.equal(st.weight[10],original['y_callig_embedder.embedding_table.weight'][10])
    assert not torch.equal(st.weight[1],original['y_callig_embedder.embedding_table.weight'][1])
    for k in original:
        if 'y_script' in k or 'y_char' in k:
            assert torch.equal(m.state_dict()[k],original[k])
    assert opt.param_groups[1]['weight_decay']==0.
    assert set(m.state_dict())==set(original)


def test_anchor_normalizes_real_rows_and_excludes_null(tmp_path):
    m,a=make_model(tmp_path);st=configure_writer_table(m,a)
    with torch.no_grad():
        st.weight[:10]*=2
        st.weight[10]+=100
    assert st.anchor_loss().item()<1e-12
    with torch.no_grad():st.weight[0,0]+=1
    assert st.anchor_loss().item()>0
    st.anchor_loss().backward()
    assert st.weight.grad[10].count_nonzero()==0


def test_ema_null_is_restored_exactly(tmp_path):
    import copy
    m,a=make_model(tmp_path);st=configure_writer_table(m,a)
    ema=copy.deepcopy(m)
    with torch.no_grad():ema.y_callig_embedder.embedding_table.weight[10]+=1
    st.restore_null(ema)
    assert torch.equal(ema.y_callig_embedder.embedding_table.weight[10],st.fixed_tail[0])


def test_scheduler_and_optimizer_resume_preserve_two_speed_contract(tmp_path):
    m,a=make_model(tmp_path);st=configure_writer_table(m,a)
    opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=1e-5,weight_decay=.02)
    split_writer_optimizer_group(opt,st,a.callig_table_lr)
    sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda s:.5**s)
    for _ in range(3):
        opt.zero_grad();m.y_callig_embedder(torch.tensor([2]),False).sum().backward()
        opt.step();st.restore_null();sched.step()
        assert abs(opt.param_groups[1]['lr']/opt.param_groups[0]['lr']-.2)<1e-10
    state=opt.state_dict();anchor=st.reference.clone()
    opt.load_state_dict(state)
    assert torch.equal(st.reference,anchor)
    assert len(opt.param_groups)==2


def test_default_disabled_leaves_frozen_tables_unchanged(tmp_path):
    m,a=make_model(tmp_path);a.finetune_callig_table=False
    original={n:p.requires_grad for n,p in m.named_parameters()}
    assert configure_writer_table(m,a) is None
    assert original=={n:p.requires_grad for n,p in m.named_parameters()}
