"""The partial-transfer contract must refuse anything it cannot explain.

The migration's real risk is silence: a dropped tensor nobody noticed, or a typo'd prefix
that quietly leaves part of the model at random init. These tests pin the guards that make
that impossible, using tiny stand-in modules so they run in a second without a checkpoint.

The end-to-end proof against the real P0 EMA lives in tools/p0_s4_ft/migrate_dryrun.py;
what is tested here is the logic that decides.
"""
import torch
from torch import nn

from tools.p0_s4_ft.partial_transfer import (WRITER_TABLES, apply_transfer,
                                             init_writer_tables, load_backbone,
                                             plan_transfer)


def _r2_repr(writers):
    """Build an R2 repr module whose parameter names match the real one.

    Real names are `style_representation.local.N.M.weight`, so `local` must be a
    ModuleList of Sequentials -- a ModuleDict keyed `local.0` would be rejected by
    torch (keys cannot contain dots) and would not produce the names we detect on.
    """
    rep = nn.Module()
    rep.table = nn.Embedding(writers + 1, 4)
    rep.project = nn.Linear(4, 4)
    rep.local = nn.ModuleList([nn.Sequential(*[nn.Conv2d(1, 1, 1) for _ in range(3)])
                               for _ in range(3)])
    rep.pool = nn.Module()
    rep.pool.query = nn.Linear(4, 4)
    return rep


def _i1_injections(n=2):
    injs = nn.ModuleList()
    for _ in range(n):
        inj = nn.Module()
        inj.local = nn.Sequential(nn.Conv2d(1, 1, 1), nn.Conv2d(1, 1, 1))
        inj.register_buffer('gain', torch.tensor(1.0))
        injs.append(inj)
    return injs


def _i3_injections(n=2):
    injs = nn.ModuleList()
    for _ in range(n):
        inj = nn.Module()
        inj.down = nn.Conv2d(1, 1, 1, bias=False)
        inj.basis = nn.Conv2d(1, 1, 3, padding=1, bias=False)
        inj.output = nn.Conv2d(1, 1, 1, bias=False)
        inj.register_buffer('gain', torch.tensor(1.0))
        injs.append(inj)
    return injs


class FakeI1(nn.Module):
    """Target shape: R2(LocalCNN) repr + I1 injection + writer tables."""

    def __init__(self, writers=10):
        super().__init__()
        self.backbone = nn.Module()
        self.backbone.writer = nn.Embedding(writers + 1, 4)
        self.backbone.script = nn.Linear(3, 3)
        self.style_representation = _r2_repr(writers)
        self.style_injections = _i1_injections()


def fake_source(writers=47):
    """A source in the P0 shape: same R2, an I3-style injection, more writers."""
    m = nn.Module()
    m.backbone = nn.Module()
    m.backbone.writer = nn.Embedding(writers + 1, 4)
    m.backbone.script = nn.Linear(3, 3)
    m.style_representation = _r2_repr(writers)
    m.style_injections = _i3_injections()
    with torch.no_grad():
        for p in m.parameters():
            p.add_(0.5)
    return m.state_dict()


def test_plan_classifies_every_tensor_with_no_default():
    src = fake_source()
    model = FakeI1()
    _to_load, rep = plan_transfer(src, model, expect_repr=2, expect_injection=1)
    accounted = set(rep['transferred']) | set(rep['dropped']) | set(rep['fresh'])
    style = {k for k in list(src) + list(model.state_dict())
             if k.startswith(('style_representation.', 'style_injections.'))}
    assert accounted == style, sorted(style - accounted)


def test_i3_parameters_are_dropped_and_i1_are_fresh():
    src = fake_source()
    model = FakeI1()
    _to_load, rep = plan_transfer(src, model, expect_repr=2, expect_injection=1)
    assert any('down.weight' in k for k in rep['dropped'])
    assert any('basis.weight' in k for k in rep['dropped'])
    assert any('local.0.weight' in k for k in rep['fresh'])


def test_repr_mismatch_is_rejected_not_guessed():
    src = fake_source()
    model = FakeI1()
    try:
        plan_transfer(src, model, expect_repr=0, expect_injection=1)
    except ValueError as e:
        assert 'repr mismatch' in str(e)
    else:
        raise AssertionError('expected ValueError on repr mismatch')


def test_injection_mismatch_is_rejected_not_guessed():
    src = fake_source()
    model = FakeI1()
    try:
        plan_transfer(src, model, expect_repr=2, expect_injection=3)
    except ValueError as e:
        assert 'injection mismatch' in str(e)
    else:
        raise AssertionError('expected ValueError on injection mismatch')


def test_unexplained_source_tensor_raises():
    """A non-injection style tensor with no target counterpart must not be dropped."""
    src = fake_source()
    src = dict(src)
    src['style_representation.mystery.weight'] = torch.zeros(2, 2)
    model = FakeI1()
    try:
        plan_transfer(src, model, expect_repr=2, expect_injection=1)
    except ValueError as e:
        assert 'refusing to drop' in str(e)
    else:
        raise AssertionError('expected refusal to drop an unexplained tensor')


def test_apply_transfer_is_byte_exact():
    src = fake_source()
    model = FakeI1()
    to_load, rep = plan_transfer(src, model, expect_repr=2, expect_injection=1)
    proof = apply_transfer(model, src, rep, to_load)
    assert proof['max_diff'] == 0.0
    assert proof['verified'] == rep['counts']['transferred']


def test_load_backbone_is_byte_exact_and_skips_style():
    src = fake_source()
    model = FakeI1()
    _to_load, rep = plan_transfer(src, model, expect_repr=2, expect_injection=1)
    bb = load_backbone(model, src, rep)
    assert bb['max_diff'] == 0.0
    assert bb['identical'] == bb['loaded']
    # The writer table changes width (48 -> 11 rows), so it must be excluded rather than
    # copied; a copy would either raise or silently truncate.
    after = model.state_dict()
    assert after['backbone.writer.weight'].shape[0] == 11


def test_writer_tables_are_reported_separately_from_style_counts():
    src = fake_source()
    model = FakeI1()
    _to_load, rep = plan_transfer(src, model, expect_repr=2, expect_injection=1)
    for k in WRITER_TABLES:
        assert k in rep['writer_tables']
        assert rep['writer_tables'][k]['action'] == 'reinit'
        assert rep['writer_tables'][k]['source_rows'] == 48
        assert rep['writer_tables'][k]['target_rows'] == 11
    # The style-prefixed writer table is listed under `dropped` (it genuinely is not
    # copied) AND under `writer_tables` (which records how it is re-seeded instead).
    # What matters is that the re-seed path is recorded, not that the lists are disjoint.
    assert 'style_representation.table.weight' in rep['dropped']
    assert 'style_representation.table.weight' in rep['writer_tables']
    # The backbone-side table is not style-prefixed, so it must NOT appear in the style
    # audit at all -- if it did, the style counts would be misleading.
    assert 'backbone.writer.weight' not in rep['dropped']
    assert 'backbone.writer.weight' not in rep['transferred']


def test_writer_seeding_preserves_mean_and_adds_spread():
    src = fake_source()
    model = FakeI1()
    out = init_writer_tables(model, src, jitter=0.01, seed=0)
    for k, v in out.items():
        assert v['present']
        assert abs(v['new_mean'] - v['source_mean']) < 0.05
        assert v['row_spread'] > 0.0, 'rows must not be exactly identical'
    # two identical rows would give identical gradients on step 1
    t = model.state_dict()['backbone.writer.weight']
    assert not torch.allclose(t[0], t[1])
