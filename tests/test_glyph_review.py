"""Small CPU contract tests; no models, downloads or training."""
import csv
import json

import numpy as np
from PIL import Image
import pytest

from tools.glyph_review import geometry as geo
from tools.glyph_review.calibrate import main
from tools.glyph_review.poster import render


def bar(width=12):
    image = np.full((64, 64, 3), 255, dtype=np.uint8)
    image[10:54, 32-width//2:32+width//2] = 25
    return image


def write_csv(path, keys):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['image_path','calligrapher','script','character'])
        writer.writeheader()
        writer.writerows(dict(image_path=k, calligrapher='王', script='楷', character=k) for k in keys)


def test_measurement_is_repeatable_and_zero_is_identity():
    image = bar()
    image[20:30, 28:36] = np.arange(80).reshape(10, 8, 1)+40
    a, *arrays_a = geo.measure(image)
    b, *arrays_b = geo.measure(image)
    assert a == b
    assert all(np.array_equal(x,y) for x,y in zip(arrays_a, arrays_b))
    assert np.array_equal(geo.displace(image, 0), image)
    adjusted, record = geo.calibrate(image, delta=2, settings=geo.Settings(strength=0))
    assert np.array_equal(adjusted, image)
    assert record['status'] == 'zero_strength'


@pytest.mark.parametrize('delta', [-1., 1.])
def test_manual_width_direction(delta):
    _, record = geo.calibrate(bar(), delta=delta)
    assert record['changed']
    assert (record['after']['p50']-record['before']['p50'])*delta > 0
    assert record['before']['components'] == record['after']['components'] == 1


def test_topology_rejects_merge_split_and_destroyed_hole():
    mask = np.zeros((32,32), bool)
    mask[5:25, 5:10] = True
    mask[5:25, 15:20] = True
    merged = mask.copy()
    merged[12:17, 10:15] = True
    assert not geo.topology_preserved(mask, merged)
    assert not geo.topology_preserved(merged, mask)
    ring = np.zeros((32,32), bool)
    ring[4:28,4:28] = True
    ring[10:22,10:22] = False
    filled = ring.copy()
    filled[10:22,10:22] = True
    assert not geo.topology_preserved(ring, filled)


def test_unsafe_adjustment_returns_original(monkeypatch):
    monkeypatch.setattr(geo, 'topology_preserved', lambda *args: False)
    image = bar()
    adjusted, record = geo.calibrate(image, delta=2)
    assert np.array_equal(adjusted, image)
    assert not record['changed'] and record['topology_rejected'] == 4


def test_target_improves_or_safely_keeps_image():
    image = bar()
    adjusted, r = geo.calibrate(image, target={'p50':16})
    assert r['changed'] and r['objective_after'] < r['objective_before']
    assert geo.topology_preserved(geo.measure(image)[1], geo.measure(adjusted)[1])
    with pytest.raises(ValueError, match='three target'):
        geo.calibrate(image, target={'p50':16}, method='profile')


def test_reference_leakage_guard_and_byte_identity(tmp_path):
    source = tmp_path/'source'
    source.mkdir()
    Image.fromarray(bar()).save(source/'g0.png')
    table = tmp_path/'train.csv'
    write_csv(table, ['source/g0.png'])
    reference = tmp_path/'reference.json'
    main(['reference','--csv',str(table),'--image-root',str(tmp_path),'--role','train',
          '--output',str(reference),'--size','64'])
    common = ['apply','--csv',str(table),'--input',str(source),'--size','64']
    with pytest.raises(ValueError, match='overlaps'):
        main(common+['--mode','reference','--reference',str(reference),'--gt-dir',str(source),
                     '--output',str(tmp_path/'leaked')])
    assert not (tmp_path/'leaked').exists()
    out = tmp_path/'identity'
    main(common+['--mode','manual','--delta','1','--strength','0','--output',str(out)])
    assert (out/'g0.png').read_bytes() == (source/'g0.png').read_bytes()
    assert json.loads((out/'run.json').read_text())['status'] == 'complete'
    oracle = tmp_path/'oracle'
    main(common+['--mode','oracle','--gt-dir',str(source),'--output',str(oracle)])
    report = json.loads((oracle/'run.json').read_text())
    assert report['oracle'] and report['summary']['before']['ssim_gray'] == 1
    assert report['summary']['after']['p50_mean_absolute_relative_error'] == 0


def test_poster_matches_csv_keys_and_preserves_canvas(tmp_path):
    anchor, other = tmp_path/'a.csv', tmp_path/'b.csv'
    write_csv(anchor, ['one','two'])
    write_csv(other, ['two','one'])
    source = tmp_path/'images'
    source.mkdir()
    Image.fromarray(bar(8)).save(source/'g0.png')
    im = bar(16)
    im[2:6, 2:6] = [200,0,0]  # Off-center marker must not be recropped away.
    Image.fromarray(im).save(source/'g1.png')
    original = (source/'g1.png').read_bytes()
    config = tmp_path/'poster.json'
    config.write_text(json.dumps(dict(index_csv=str(anchor), image_size=64,
        selection=dict(indices=[0]),columns=[dict(label='GT',kind='gt',csv=str(other),dir=str(source))])), encoding='utf-8')
    out = tmp_path/'poster'
    render(config,out,cell=128)
    manifest = json.loads((out/'manifest.json').read_text(encoding='utf-8'))
    assert manifest['images'][0]['mapped_idx'] == 1
    assert manifest['display'] == 'entire_canvas_uniform_scale_no_recrop'
    page = np.asarray(Image.open(out/'page01.png'))
    assert page[194,136,0] > page[194,136,1]+100
    assert original == (source/'g1.png').read_bytes()
