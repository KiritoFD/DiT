import csv
import hashlib
import importlib
import json
import subprocess
from types import SimpleNamespace

import pytest
import torch

from tools import v85_compare
from src.train.gradient_budget import GradientBudget
from src.train.strong_style import build_strong_style


def test_prepare_v88_import_has_no_writes(monkeypatch):
    from pathlib import Path
    import tools.analysis.prepare_v88 as prepare
    def forbidden(*args, **kwargs):
        raise AssertionError('Import must not write files')
    with monkeypatch.context() as patcher:
        patcher.setattr(Path, 'write_text', forbidden)
        importlib.reload(prepare)
    baseline = {'data_csv': 'original.csv', 'vae_scaling_factor': .18215}
    config = prepare.build_config(baseline)
    assert baseline == {'data_csv': 'original.csv', 'vae_scaling_factor': .18215}
    assert config['data_csv'] == 'original.csv'
    assert config['strong_style_mode'] == 'off'
    assert config['max_steps'] == 180000


@pytest.mark.parametrize('mode', ['', 'OFF', None, 'E'])
def test_invalid_posttraining_modes_fail_before_loading_assets(mode):
    with pytest.raises(ValueError, match='strong_style_mode'):
        build_strong_style(SimpleNamespace(strong_style_mode=mode), 'cpu', None)


def test_disabled_posttraining_needs_no_assets():
    assert build_strong_style(SimpleNamespace(strong_style_mode='off'), 'cpu', None) is None


def test_gradient_budget_matches_formula_and_does_not_accumulate_grads():
    value = torch.tensor([.3, -.5], requires_grad=True)
    loss = (value-2).square().mean()
    reference = torch.tensor(3.)
    scales, counts = {}, {}
    budget = GradientBudget(5., scales, counts, SimpleNamespace(info=lambda *args: None))
    gradient = torch.autograd.grad(loss, value, retain_graph=True)[0]
    expected = min(5., float(.8*reference/gradient.norm()))
    total, _ = budget.balance({'writer': (loss, .8)}, [value], reference, 0)
    assert scales['writer'] == pytest.approx(expected)
    assert value.grad is None
    assert torch.allclose(torch.autograd.grad(total, value)[0], expected*gradient)


def test_content_metrics_require_exact_checkpoint_and_set(tmp_path):
    path = tmp_path/'metrics.csv'
    with path.open('w', newline='') as output:
        writer = csv.DictWriter(output, fieldnames=['step', 'set', 'ssim_mean'])
        writer.writeheader()
        writer.writerows([{'step': 5000, 'set': 'eval200fix', 'ssim_mean': .6},
                          {'step': 10000, 'set': 'eval200fix', 'ssim_mean': .5},
                          {'step': 10000, 'set': 'seen', 'ssim_mean': .9}])
    assert v85_compare.read_csv_metrics(str(path), step=5000)['ssim'] == .6
    with pytest.raises(ValueError, match='Expected one row'):
        v85_compare.read_csv_metrics(str(path))
    with pytest.raises(ValueError):
        v85_compare.read_csv_metrics(str(path), 'missing', 5000)
    with pytest.raises(ValueError):
        v85_compare.read_csv_metrics(str(tmp_path/'*.absent'), step=5000)
    assert v85_compare.sample_step('run/step0005000/eval200fix') == 5000
    assert v85_compare.sample_step('run/eval200fix') is None


def test_failed_evaluator_cannot_reuse_old_report(tmp_path, monkeypatch):
    path = tmp_path/'style_report.json'
    path.write_text(json.dumps({'sets': {'a': {}}}))
    monkeypatch.setattr(subprocess, 'run', lambda *a, **k: SimpleNamespace(returncode=1, stderr='failed', stdout=''))
    with pytest.raises(RuntimeError, match='failed'):
        v85_compare.run_style_report(['dummy'], path, ['a'])
    monkeypatch.setattr(subprocess, 'run', lambda *a, **k: SimpleNamespace(returncode=0, stderr='', stdout=''))
    with pytest.raises(RuntimeError, match='fresh report'):
        v85_compare.run_style_report(['dummy'], path, ['a'])


def test_partial_report_is_rejected(tmp_path, monkeypatch):
    path = tmp_path/'style_report.json'
    def run(*args, **kwargs):
        path.write_text(json.dumps({'sets': {'a': {}}}))
        return SimpleNamespace(returncode=0, stderr='', stdout='')
    monkeypatch.setattr(subprocess, 'run', run)
    with pytest.raises(ValueError, match='missing requested sets'):
        v85_compare.run_style_report(['dummy'], path, ['a', 'b'])
    path.unlink()
    assert v85_compare.run_style_report(['dummy'], path, ['a'])['sets'] == {'a': {}}


def test_archive_manifest_preserves_every_original_byte():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    entries = json.loads((root/'archive/experiments/v86_20261015/manifest.json').read_text())['files']
    assert len(entries) == 36
    for entry in entries:
        assert not (root/entry['original']).exists()
        archived = root/entry['archived']
        assert archived.is_file()
        assert hashlib.sha256(archived.read_bytes()).hexdigest() == entry['sha256']


def test_diagnostic_helper_is_read_only():
    from src.model.writer_glyph import WriterGlyphResidual
    from src.train.diagnostics import log_writer_glyph
    branch = WriterGlyphResidual(32, 12, 4, 8, rank=8)
    branch.film.P.weight.grad = torch.ones_like(branch.film.P.weight)
    before = {key: value.clone() for key, value in branch.state_dict().items()}
    logs = []
    wrapper = SimpleNamespace(module=SimpleNamespace(_orig_mod=SimpleNamespace(writer_glyph=branch)))
    log_writer_glyph(wrapper, SimpleNamespace(info=lambda *args: logs.append(args)), 50)
    assert logs and logs[0][1] == 50
    assert all(torch.equal(before[key], value) for key, value in branch.state_dict().items())
