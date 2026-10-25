"""Small CPU tests for reversible filtering and grouped split contracts."""
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from tools.data_clean.common import read_json,rows,sha,write_csv
from tools.data_clean.scan import extract,link_duplicates,main as scan
from tools.data_clean.build import main as build,conflicts


def glyph(kind=0):
    x = np.full((256,256,3),255,np.uint8)
    x[100:120,40:175] = 0
    x[60:200,100:115] = 0
    if kind == 1:x[160:180,50:190] = 0
    if kind == 2:x[10:240,205:218] = 0
    return x


def metadata(path,char='青',writer='王',script='楷',source=''):
    return dict(image_path=path,character=char,calligrapher=writer,script=script,
        calligrapher_id='0' if writer=='王' else '1',script_id='0',img_id=Path(path).stem,
        character_id='1',glyph_id='1',aug='',source='test',src_image_path=source)


def setup(tmp_path):
    train = [metadata('a.png'),metadata('b.png','尚'),metadata('c.png','靑','苏','行')]
    evaluation = [metadata('e.png','靑')]
    for name,kind in [('a',0),('b',1),('c',2),('e',0)]:
        Image.fromarray(glyph(kind)).save(tmp_path/f'{name}.png',compress_level=0 if name=='e' else 9)
    for role,items in [('train',train),('eval',evaluation),('top5',train)]:write_csv(tmp_path/f'{role}.csv',items)
    scan(['--train-csv',str(tmp_path/'train.csv'),'--eval-csv',str(tmp_path/'eval.csv'),
        '--top5-csv',str(tmp_path/'top5.csv'),'--root',str(tmp_path),'--output',str(tmp_path/'scan'),'--workers','2'])
    return tmp_path/'scan'


def test_reencoded_exact_pixels_group_without_silent_alias_rewrite(tmp_path):
    s = setup(tmp_path)
    images = {r['image_path']:r for r in read_json(s/'images.json')}
    assert images['a.png']['file_sha256'] != images['e.png']['file_sha256']
    assert images['a.png']['duplicate_group'] == images['e.png']['duplicate_group']
    assert images['a.png']['metadata'][0]['character'] == '青'
    assert images['e.png']['metadata'][0]['character'] == '靑'


def test_build_removes_heldout_group_and_quarantines_uncertain_quality(tmp_path):
    s = setup(tmp_path)
    out = tmp_path/'build'
    build(['--scan',str(s),'--output',str(out)])
    assert [r['image_path'] for r in rows(out/'train_top10_candidate.csv')] == ['b.png']
    assert [r['image_path'] for r in rows(out/'eval_top10_candidate.csv')] == ['e.png']
    report = read_json(out/'report.json')
    assert report['status'] == 'candidate_not_training_approved'
    assert report['audit_counts']['original_train_eval_linked_groups'] == 1
    assert report['pending_quality_images'] == 1
    assert (tmp_path/'c.png').exists()


def test_bad_manual_hash_fails_and_does_not_create_output(tmp_path):
    s = setup(tmp_path)
    decisions = tmp_path/'decisions.json'
    decisions.write_text(json.dumps({'c.png':dict(action='keep',file_sha256='wrong',reason='reviewed')}))
    with pytest.raises(ValueError,match='hash mismatch'):
        build(['--scan',str(s),'--output',str(tmp_path/'bad'),'--decisions',str(decisions)])
    assert not (tmp_path/'bad').exists()


def test_reviewed_legal_long_stroke_can_be_restored(tmp_path):
    s = setup(tmp_path)
    d = tmp_path/'decisions.json'
    d.write_text(json.dumps({'c.png':dict(action='keep',file_sha256=sha(tmp_path/'c.png'),reason='synthetic legal long vertical')}))
    out = tmp_path/'build'
    build(['--scan',str(s),'--output',str(out),'--decisions',str(d)])
    assert len(rows(out/'train_top10_candidate.csv')) == 2
    assert read_json(out/'report.json')['pending_quality_images'] == 0


def test_conflicting_writer_labels_are_not_auto_reassigned(tmp_path):
    r1 = dict(image_path='a',metadata=[metadata('a')])
    r2 = dict(image_path='b',metadata=[metadata('b',writer='苏')])
    for r in (r1,r2):r.update(pixel_sha256='same',phash='1')
    link_duplicates([r1,r2],[glyph()[...,0],glyph()[...,0]])
    assert r1['duplicate_group'] in conflicts([r1,r2])


def test_flags_are_not_hard_quality_verdicts_and_measurement_is_repeatable(tmp_path):
    path = tmp_path/'stroke.png';Image.fromarray(glyph(2)).save(path)
    a,x = extract(path);b,y = extract(path)
    assert a == b and np.array_equal(x,y)
    assert a['flags'] == ['peripheral_bar'] and a['hard_issues'] == []


def test_source_identity_reserved_even_if_eval_itself_is_rejected(tmp_path):
    for name,kind in [('a',0),('b',1),('e',2)]:Image.fromarray(glyph(kind)).save(tmp_path/f'{name}.png')
    train = [metadata('a.png',source='page/crop.png'),metadata('b.png','尚')]
    ev = [metadata('e.png','靑',source='page/crop.png')]
    for role,items in [('train',train),('eval',ev),('top5',train)]:write_csv(tmp_path/f'{role}.csv',items)
    s = tmp_path/'scan'
    scan(['--train-csv',str(tmp_path/'train.csv'),'--eval-csv',str(tmp_path/'eval.csv'),
        '--top5-csv',str(tmp_path/'top5.csv'),'--root',str(tmp_path),'--output',str(s)])
    out = tmp_path/'build';build(['--scan',str(s),'--output',str(out)])
    assert [r['image_path'] for r in rows(out/'train_top10_candidate.csv')] == ['b.png']


def test_thin_edge_frame_is_flagged_but_center_stroke_is_not(tmp_path):
    a=glyph();a[3,:]=0
    p=tmp_path/'frame.png';Image.fromarray(a).save(p)
    stats,_=extract(p)
    assert 'thin_edge_frame' in stats['flags'] and not stats['hard_issues']
    a=glyph();a[128,:]=0
    p=tmp_path/'stroke.png';Image.fromarray(a).save(p)
    assert 'thin_edge_frame' not in extract(p)[0]['flags']
