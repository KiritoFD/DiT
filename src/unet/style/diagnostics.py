"""Bounded mechanism checks used before launch and at optimizer milestones."""
import torch


@torch.no_grad()
def inspect(model,inputs,calibrate=False):
    was_training=model.training;model.eval()
    x,t,g,w,s=inputs
    old_enabled=model.style_enabled;model.style_enabled=False
    plain=model(x,t,w,g=g,y_script=s)
    switched=model(x,t,(w+1)%10,g=g,y_script=s)
    assert torch.equal(plain,switched), 'Legacy Writer leakage'
    native=model.backbone(x,t,torch.full_like(w,model.num_writers),g=g,y_script=s)
    torch.testing.assert_close(plain,native,rtol=0,atol=0)
    model.style_enabled=True
    for layer in model.style_injections:layer.record=True
    conditional=model(x,t,w,g=g,y_script=s)
    if calibrate:
        for layer in model.style_injections:
            ratio=layer.last['delta_rms']/max(layer.last['base_rms'],1e-9)
            if ratio<1e-10:raise RuntimeError('Dead style initialization')
            layer.gain.mul_(.1/ratio)
        conditional=model(x,t,w,g=g,y_script=s)
    layer_reports=[dict(layer.last) for layer in model.style_injections]
    base=model.backbone;p=base.std_stem(g);pyramid=[p]
    for down in base.std_down:pyramid.append(base.act(down(pyramid[-1])))
    tokens=model.style_representation(w,pyramid,g).tokens.float()
    singular=torch.linalg.svdvals(tokens)
    probability=singular.square()/singular.square().sum(-1,keepdim=True).clamp_min(1e-9)
    effective_rank=float((-(probability*probability.clamp_min(1e-9).log()).sum(-1)).exp().mean())
    normalized=torch.nn.functional.normalize(tokens,dim=-1)
    correlation=normalized@normalized.transpose(-1,-2)
    swapped=model(x,t,(w+1)%10,g=g,y_script=s)
    null=model(x,t,torch.full_like(w,model.num_writers),g=g,y_script=s)
    report=dict(legacy_writer_leakage=False,disabled_matches_source_null=True,
        swap_rms=float((conditional-swapped).float().square().mean().sqrt()),
        null_gap=float((conditional-null).float().square().mean().sqrt()),
        output_rms=float(plain.float().square().mean().sqrt()),
        layers=layer_reports,
        tokens_effective_rank=effective_rank,
        tokens_off_diagonal_cosine=float((correlation.sum((-1,-2))-8).mean()/56))
    for layer in model.style_injections:layer.record=False
    model.style_enabled=old_enabled;model.train(was_training)
    return report


def gradient_report(model):
    report={}
    for prefix in ('style_representation','style_injections','backbone'):
        ps=[p for n,p in model.named_parameters() if n.startswith(prefix) and p.grad is not None]
        report[prefix]=float(torch.sqrt(sum(p.grad.detach().float().square().sum() for p in ps))) if ps else 0.
    if report['style_representation']==0 or report['style_injections']==0:
        raise RuntimeError('Style branch has no gradient')
    return report
