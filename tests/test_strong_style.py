import torch
import pytest
from src.train.strong_style import real_mask, writer_margin, calibrated_hinge, cf_loss
from src.train.strong_style import StrongStyle
from types import SimpleNamespace


SLOTS = ['A·kai', 'B·kai', 'C·xing', 'D·xing', 'std·kai']


def test_margin_excludes_std_and_other_script():
    logits = torch.tensor([[3., 1., 100., 200., 300.]], requires_grad=True)
    margin = writer_margin(logits, torch.tensor([0]), real_mask(SLOTS, 'cpu'))
    assert torch.allclose(margin, torch.tensor([2.]))
    margin.sum().backward()
    assert torch.equal(logits.grad[0, 2:], torch.zeros(3))


def test_sequential_backward_equals_combined_gradient_before_one_step():
    a = torch.tensor([.2, -.3], requires_grad=True)
    b = a.detach().clone().requires_grad_()
    ((a - 1).square().mean() + .7 * (a + 2).square().mean()).backward()
    (b - 1).square().mean().backward()
    (.7 * (b + 2).square().mean()).backward()
    assert torch.allclose(a.grad, b.grad)


def test_calibrated_target_has_zero_gradient_after_satisfied():
    margin = torch.tensor([2., 3.], requires_grad=True)
    loss = calibrated_hinge(margin, torch.ones(2))
    loss.backward()
    assert loss == 0 and torch.equal(margin.grad, torch.zeros(2))


def test_huber_outlier_gradient_is_bounded_and_not_zero():
    margin = torch.tensor([-100.], requires_grad=True)
    calibrated_hinge(margin, torch.tensor([1.])).backward()
    assert margin.grad.item() == -1.


def test_cf_rewards_correct_condition_and_gradients_both_branches():
    mask = real_mask(SLOTS, 'cpu')
    a = torch.tensor([[3., 0., 0., 0., 0.]], requires_grad=True)
    b = torch.tensor([[0., 3., 0., 0., 0.]], requires_grad=True)
    ya, yb = torch.tensor([0]), torch.tensor([1])
    correct = cf_loss(a, b, ya, yb, mask)
    assert correct < cf_loss(b, a, ya, yb, mask)
    correct.backward()
    assert a.grad.abs().sum() > 0 and b.grad.abs().sum() > 0


def test_cf_training_hook_shares_inputs_restores_mode_and_backpropagates():
    class Generator(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(.1))
            self.inputs = []

        def forward(self, x, t, **kw):
            assert not self.training
            self.inputs.append((x.detach().clone(), t.clone(), kw['g'].clone(),
                                kw['y_script'].clone(), kw['y_callig'].clone()))
            return x * self.weight + kw['y_callig'][:, None, None, None] * self.weight

    class Critic(torch.nn.Module):
        def forward(self, z, g, s):
            v = z.mean((1, 2, 3))
            return {'slot_logits': torch.stack((v, -v), 1), 'brush_logit': v}

    hook = StrongStyle.__new__(StrongStyle)
    hook.args = SimpleNamespace(strong_style_mode='C', strong_style_cf_every=4,
        strong_style_samples=0, strong_style_full_batch=True, critic_t_min=.05, critic_t_max=.3,
        strong_style_ratio=.6, strong_style_cf_ratio=.3,
        strong_style_probe_every=50, strong_style_max_weight=.1)
    hook.logger = SimpleNamespace(info=lambda *args: None)
    hook.critic = SimpleNamespace(slots=['A·kai', 'B·kai'], id2slot={(0, 0): 'A·kai', (1, 0): 'B·kai'},
                                 s2i={'kai': 0}, model=Critic())
    hook.mask = real_mask(hook.critic.slots, 'cpu')
    hook.inverse = {0: (0, 0), 1: (1, 0)}
    hook.targets = torch.tensor([10., 10.])
    hook.scale = {}
    model = Generator().train()
    n = 12
    x0 = torch.randn(n, 4, 4, 4)
    kwargs = {'y_callig': torch.full((n,), 10), 'y_script': torch.zeros(n, dtype=torch.long),
              'y_char': torch.zeros(n, dtype=torch.long), 'g': torch.zeros(n, 1, 8, 8)}
    batch = {'y_callig_raw': torch.tensor([0, 1] * 6), 'y_script': torch.zeros(n, dtype=torch.long),
             'std_soft': torch.ones(n, 1, 8, 8)}
    t = torch.linspace(0, 1, n)  # full batch includes rows outside the FM time window
    fm = (model.weight - 1).square()
    aux = hook(model, x0, t, kwargs, batch, fm, 0)
    assert len(model.inputs) == 2 and model.training
    assert model.inputs[0][0].shape[0] == n  # no silent eight-row cap
    assert (model.inputs[0][1] >= 50).all() and (model.inputs[0][1] <= 300).all()
    for a, b in zip(model.inputs[0][:4], model.inputs[1][:4]):
        assert torch.equal(a, b)
    assert torch.all(model.inputs[0][4] != model.inputs[1][4])
    assert torch.all(model.inputs[0][2] == 1)
    aux.backward()
    assert model.weight.grad.abs() > 0
    assert len(hook.scale) == 2
    # A deliberately tiny cap must fail loudly instead of running a weak arm.
    hook.args.strong_style_max_weight = 1e-12
    hook.weak_probes = {}
    with pytest.raises(RuntimeError, match='underpowered'):
        for _ in range(10):
            hook(model, x0, t, kwargs, batch, fm, 0)
