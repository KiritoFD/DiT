"""CPU checks for time-windowed corrected SDE; no checkpoint needed."""
import pytest
import torch

from src.loss import flow_kwargs_from
from src.loss.flow_matching import FlowMatching, create_flow_matching


@pytest.mark.parametrize("bounds", [(-.1, 1, 0), (0, 1.1, 0), (.6, .4, 0), (.2, .8, .4)])
def test_invalid_window(bounds):
    with pytest.raises(ValueError):
        FlowMatching(sde_tmin=bounds[0], sde_tmax=bounds[1], sde_ramp=bounds[2])


def test_envelope_and_factory():
    kw = dict(sde_tmin=.2, sde_tmax=.8, sde_ramp=.1, sde_gamma=.5)
    fm = create_flow_matching("50", **flow_kwargs_from(kw))
    t = torch.tensor([0, .2, .25, .4, .7, .75, .8, 1.])
    torch.testing.assert_close(fm._sde_weight(t), torch.tensor([0, 0, .5, 1, 1, .5, 0, 0.]))
    assert FlowMatching(sampler="euler", sde_gamma=.5).nfe == 100


@pytest.mark.parametrize("s0", [.2, 1.])
@pytest.mark.parametrize("window", [(0., 1., 0.), (.2, .8, .1)])
def test_oracle_marginal(s0, window):
    def velocity(x, time, **kwargs):
        t = time / 1000
        variance = (1 - t).square() * s0 ** 2 + t.square()
        return ((t - (1 - t) * s0 ** 2) / variance)[:, None, None, None] * x

    torch.manual_seed(51)
    x = torch.randn(20000, 1, 1, 1)
    fm = FlowMatching(num_steps=100, sde_gamma=.5,
                      sde_tmin=window[0], sde_tmax=window[1], sde_ramp=window[2])
    out = fm.ddim_sample_loop(velocity, x.shape, x, device=torch.device("cpu"))
    assert abs(out.mean().item()) < .025 * s0
    assert abs(out.var().item() / s0 ** 2 - 1) < .04


def test_gamma_zero_ignores_window():
    x = torch.randn(3, 1, 2, 2)
    velocity = lambda x, t, **kw: -.1 * x
    common = dict(num_steps=10, sde_gamma=0)
    full = FlowMatching(**common)
    window = FlowMatching(**common, sde_tmin=.2, sde_tmax=.8, sde_ramp=.1)
    assert torch.equal(full.ddim_sample_loop(velocity, x.shape, x, device="cpu"),
                       window.ddim_sample_loop(velocity, x.shape, x, device="cpu"))


@pytest.mark.parametrize("module", ["src.train.cli", "src.eval.cli"])
def test_cli_keeps_shard_shift(module, tmp_path):
    import importlib
    import json
    config = tmp_path / 'config.json'
    config.write_text(json.dumps({'vae_shift_factor': -.03341800242750449}), encoding='utf-8')
    parser = importlib.import_module(module).build_parser(
        ['--config', str(config)])
    assert parser.get_default('vae_shift_factor') == -.03341800242750449
    args, _ = parser.parse_known_args(['--config', str(config),
                                       '--sde-tmin', '.2', '--sde-tmax', '.8', '--sde-ramp', '.1'])
    assert args.sde_tmin == .2
    assert args.sde_tmax == .8
    assert args.sde_ramp == .1


def test_shift_wrapper_survives_eval_and_does_not_mutate_base():
    from src.eval.in_mem_eval import _ShiftedVAE

    class FakeVAE:
        def eval(self):
            return self

        def decode(self, z):
            return z

    base = FakeVAE()
    shifted = _ShiftedVAE(base, -.033418).eval()
    x = torch.ones(1)
    assert isinstance(shifted, _ShiftedVAE)
    torch.testing.assert_close(shifted.decode(x), x - .033418)
    assert torch.equal(base.decode(x), x)
