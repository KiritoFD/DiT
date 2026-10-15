import pytest
import torch

from tools.analysis.probe_B25_gradients import gradient_stats


@pytest.mark.parametrize('direction,expected', [(1., 1.), (-1., -1.)])
def test_aligned_and_opposed_gradients(direction, expected):
    value = torch.tensor([3., 4.])
    result = gradient_stats(['blocks.0.weight'], [value], [direction*value], 12)
    assert result['global']['cosine'] == pytest.approx(expected)
    assert result['early']['fm_norm'] == 5.
    assert result['middle']['cosine'] is None


def test_unused_parameters_contribute_to_their_own_norm():
    result = gradient_stats(['blocks.4.a', 'blocks.8.b', 'head.c'],
                            [torch.ones(1), None, torch.ones(1)],
                            [None, torch.ones(1), torch.ones(1)], 12)
    assert result['global']['cosine'] == pytest.approx(.5)
    assert result['middle']['cosine'] is None
    assert result['late']['writer_norm'] == 1.
    assert result['other']['cosine'] == 1.
