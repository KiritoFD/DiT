import numpy as np
import pytest
import torch

from tools.analysis.v90_p0 import coverage, fixed_input_forward, legal_wrong_slots, pair_cells, summarize


def row(w, c, s='kai', aug=''):
    return dict(calligrapher=w, character=c, script=s, source='book', aug=aug)


def test_pair_cells_are_same_script_same_char_and_deduplicate_augmentation():
    rows = [row('A', 'X'), row('B', 'X'), row('B', 'X', aug='flip'), row('C', 'X', 'xing'), row('B', 'Y')]
    cells = pair_cells(rows)
    assert len(cells) == 1
    assert cells[0] == ('kai', 'X', 'A', 'B', [0], [1])


def test_coverage_exact_distribution_mixes_a_quarter_not_all_rows():
    stats = coverage([row('A', 'X'), row('B', 'X'), row('B', 'Y')])
    assert stats['pair_eligible_row_fraction'] == pytest.approx(2/3)
    dist = stats['expected_25pct_pair_distribution']['uniform_cell']['character']
    records = {r['key']: r for r in dist['largest_upweight']}
    assert records['X']['mixed_25pct'] == pytest.approx(.75*2/3+.25)
    assert records['Y']['mixed_ratio'] == pytest.approx(.75)


def test_coverage_uses_real_augmented_row_distribution_when_supplied():
    rows = [row('A', 'X'), row('B', 'X'), row('B', 'Y')]
    stats = coverage(rows, rows + [row('B', 'Y', aug='flip')]*3)
    dist = stats['expected_25pct_pair_distribution']['uniform_cell']['character']
    records = {r['key']: r for r in dist['largest_upweight']}
    assert stats['baseline_sampling_rows'] == 6
    assert records['X']['original'] == pytest.approx(1/3)
    assert records['X']['mixed_25pct'] == pytest.approx(.5)


def test_legal_swap_rejects_cross_script_and_same_writer():
    train = [row('A', 'X'), row('B', 'Y'), row('C', 'X', 'xing')]
    _, wrong = legal_wrong_slots(train[:1], train)
    assert wrong == [[('B', 'kai')]]
    with pytest.raises(ValueError, match='No legal'):
        legal_wrong_slots([row('C', 'Z', 'xing')], train)


def test_fixed_input_forward_holds_input_and_delta_sign():
    calls = []
    z = torch.ones(2, 4, 2, 2)
    eps = torch.zeros_like(z)
    t = torch.tensor([.1, .8])
    def model(x, times, **kw):
        calls.append((x.clone(), times.clone(), kw))
        return -torch.ones_like(x) + kw['y_callig'][:, None, None, None]
    kw = dict(y_callig=torch.zeros(2, dtype=torch.long), y_char=torch.tensor([2, 3]),
              y_script=torch.tensor([0, 0]), g=torch.ones(2, 1, 4, 4))
    bad = {**kw, 'y_callig': torch.ones(2, dtype=torch.long)}
    metrics = fixed_input_forward(model, z, eps, t, kw, bad)
    assert torch.equal(calls[0][0], calls[1][0])
    assert torch.equal(calls[0][1], calls[1][1])
    assert torch.equal(calls[0][1], t*1000)
    assert torch.all(metrics['correct_mse'] == 0)
    assert torch.all(metrics['delta_fm'] == 1)
    with pytest.raises(ValueError, match='Non-writer'):
        fixed_input_forward(model, z, eps, t, kw, {**bad, 'g': torch.zeros_like(kw['g'])})
    with pytest.raises(ValueError, match='not changed'):
        fixed_input_forward(model, z, eps, t, kw, kw)


def test_bootstrap_clusters_repeated_noise_trials():
    records = [dict(character=c, delta_fm=v, relative_gap=v, relative_velocity_change=v)
               for c, v in [('X', 1.), ('X', 1.), ('Y', 3.), ('Y', 3.)]]
    result = summarize(records)
    assert result['n_clusters'] == 2
    assert result['delta_fm']['mean'] == 2
    assert np.allclose(result['delta_fm']['mean_ci95'], [1., 3.])
