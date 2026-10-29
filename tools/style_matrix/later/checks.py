"""CPU-only tests of future LR histories and checkpoint identity ancestry."""
import copy
from .plan import matched
from tools.style_matrix.generation2.plan import continuation


def factor(step,schedule):
    rel=step-schedule['origin'];warm=schedule['warmup'];horizon=schedule['horizon']
    if 0<=rel<warm:return schedule['peak_scale']*max((rel+1)/warm,1e-8)
    import math
    progress=min(max((rel-warm)/max(horizon-warm,1),0),1)
    return schedule['peak_scale']*(.1+.9*.5*(1+math.cos(math.pi*progress)))


def main():
    source=dict(config=dict(max_steps=50000,lr_decay_steps=50000,lr=6e-5),identity=dict(arm='cold',seed=90))
    snapshot=copy.deepcopy(source)
    reference60=continuation(source,'S9FM',50000,60000)
    reference75=continuation(reference60,'S9FM',60000,75000)
    for arm in ('S10','S0','S4','S11'):
        middle,mature=matched(source,arm)
        assert middle['continues_identity']==source['identity']
        assert mature['continues_identity']==middle['identity']
        for point in (50000,50001,50199,55000,59999):
            assert factor(point,middle['schedule_restart'])==factor(point,reference60['schedule_restart'])
        for point in (60000,60001,60199,65000,74999,75000):
            assert factor(point,mature['schedule_restart'])==factor(point,reference75['schedule_restart'])
        assert middle['config']['max_steps']==60000 and mature['config']['max_steps']==75000
    assert source==snapshot
    print('PASS: first-stage spec unchanged; 4 mature controls match both LR segments and identity ancestry')


if __name__=='__main__':main()
