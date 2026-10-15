import pytest
import torch

from src.model.writer_glyph import WriterGlyphResidual


def test_zero_gate_then_internal_gradients_and_masking():
    torch.manual_seed(7)
    module = WriterGlyphResidual(32, 12, 4, 8, rank=8)
    x = torch.randn(3, 16, 32)
    g = torch.ones(3, 1, 16, 16)
    g[2].zero_()
    style = torch.randn(3, 12, requires_grad=True)
    delta = module.residual(x, style, g, torch.tensor([True, False, True]))
    assert not hasattr(module.film, '_last_styled')
    assert torch.count_nonzero(delta[1:]) == 0
    result = module.inject(x, delta, 3)
    assert torch.equal(result, x)
    result.square().mean().backward()
    assert module.gains['3'].grad.abs() > 0
    with torch.no_grad():
        module.gains['3'].fill_(.1)
    module.zero_grad(set_to_none=True)
    delta = module.residual(x, style, g, torch.tensor([True, False, True]))
    module.inject(x, delta, 3).square().mean().backward()
    assert module.film.P.weight.grad.norm() > 0
    assert module.film.Q.weight.grad.norm() > 0
    assert style.grad.norm() > 0


def test_local_residual_changes_with_writer_and_geometry():
    torch.manual_seed(9)
    module = WriterGlyphResidual(32, 12, 4, 8, rank=8)
    x, style = torch.randn(2, 16, 32), torch.randn(2, 12)
    g = torch.ones(2, 1, 16, 16)
    keep = torch.ones(2, dtype=torch.bool)
    original = module.residual(x, style, g, keep)
    assert not torch.equal(original, module.residual(x, style.flip(0), g, keep))
    assert not torch.equal(original, module.residual(x.flip(0), style, g, keep))
    assert torch.count_nonzero(module.residual(x, style, g, keep, torch.zeros(2))) == 0


def test_model_zero_gate_parity_and_strict_rebuild():
    from src.model.dit import DiT_2Cond
    kw = dict(input_size=8, patch_size=2, hidden_size=32, depth=8, num_heads=4,
        num_calligraphers=3, num_scripts=2, callig_embed_dim=8, script_embed_dim=4,
        use_script_cond=True, use_char_cond=False, content_cond='std', std_soft_cond=True,
        use_glyph_cond=True, glyph_in_channels=1, condition_fusion='factorized_cat',
        cond_fusion_norm='split', learn_sigma=False)
    base = DiT_2Cond(**kw).eval()
    model = DiT_2Cond(**kw, writer_glyph_rank=8).eval()
    missing, unexpected = model.load_state_dict(base.state_dict(), strict=False)
    assert missing and all(k.startswith('writer_glyph.') for k in missing)
    assert not unexpected
    with torch.no_grad():
        base.final_layer.linear.weight.normal_(std=.01)
        model.final_layer.linear.weight.copy_(base.final_layer.linear.weight)
    x = torch.randn(2, 4, 8, 8)
    cond = dict(y_callig=torch.tensor([0, 1]), y_script=torch.tensor([0, 1]),
                y_char=torch.zeros(2, dtype=torch.long), g=torch.rand(2, 1, 16, 16))
    t = torch.tensor([100., 500.])
    assert torch.equal(base(x, t, **cond), model(x, t, **cond))
    clone = DiT_2Cond(**kw, writer_glyph_rank=8).eval()
    clone.load_state_dict(model.state_dict(), strict=True)
    assert torch.equal(model(x, t, **cond), clone(x, t, **cond))


@pytest.mark.parametrize('at', ['3,3', '3,8', '-1,7'])
def test_invalid_layers_rejected(at):
    with pytest.raises(ValueError):
        WriterGlyphResidual(32, 12, 4, 8, at=at)


def test_training_and_eval_share_new_architecture_config():
    from src.train.cli import parse_args
    from src.model.build import dit_kwargs
    args = parse_args(['--config', 'src/train/configs/v88_writer_glyph_base.json'])
    kw = dit_kwargs(args)
    assert kw['writer_glyph_rank'] == 32
    assert kw['writer_glyph_at'] == '3,7'
    assert kw['lowrank_spatial_rank'] == 0
    assert not args.warm_start_from
    assert args.w_critic_writer == args.w_critic_away == 0
    assert args.strong_style_mode == 'off'


def test_uncached_film_is_bitwise_equal_and_keeps_checkpoint_keys():
    from src.model.components import LowRankSpatialStyleFiLM
    torch.manual_seed(11)
    film = LowRankSpatialStyleFiLM(12, 32, rank=8)
    original_keys = set(film.state_dict())
    x, style, pos = torch.randn(2, 16, 32), torch.randn(2, 12), torch.randn(1, 16, 32)
    uncached = film(x, style, pos, cache_styled=False)
    assert not hasattr(film, '_last_styled')
    cached = film(x, style, pos)
    assert torch.equal(cached, uncached)
    assert set(film.state_dict()) == original_keys
