import csv
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

from src.eval.dataset import fixed_eval_noise,make_eval_cache
from tools.data_clean.common import read_json,write_json,write_csv,sha,rows
from tools.data_cohorts.build import normalized_row,select_test,main as build
from tools.data_cohorts.prepare import FIRST_TEN


def test_noise_is_paired_when_subset_or_order_changes_and_old_csv_is_unchanged():
    rs = [dict(eval_noise_seed=str(s)) for s in (12,56,89)]
    all_noise = fixed_eval_noise(rs,4,32)
    assert torch.equal(fixed_eval_noise([rs[2],rs[0]],4,32),all_noise[[2,0]])
    assert torch.equal(fixed_eval_noise([{},{}],4,32),torch.randn(2,4,32,32,generator=torch.Generator().manual_seed(0)))
    with pytest.raises(ValueError,match='every row'):
        fixed_eval_noise([rs[0],{}],4,32)


def test_bom_csv_is_readable_by_real_eval_cache(tmp_path):
    path = tmp_path/'x.png';Image.new('RGB',(32,32),'white').save(path)
    p = tmp_path/'eval.csv'
    write_csv(p,[dict(image_path=str(path),calligrapher_id='1',character_id='2',script_id='0',img_id='1',eval_noise_seed='51')])
    cache = make_eval_cache(str(p),'','',32,1,8,4,.18215)
    assert cache['n'] == 1 and torch.equal(cache['noise'],fixed_eval_noise([{'eval_noise_seed':'51'}],4,4))


def test_unicode_ids_are_collision_free_and_original_ids_are_retained():
    a = normalized_row(dict(character='青',character_id='42',glyph_id='51',script_id='0'))
    b = normalized_row(dict(character='靑',character_id='42',glyph_id='51',script_id='3'))
    assert a['character_id'] != b['character_id'] and a['glyph_id'] != b['glyph_id']
    assert a['original_character_id'] == b['original_character_id'] == '42'


def test_selection_balances_scripts_and_never_repeats_character_or_source_group():
    rs = [dict(image_path=str(i),script='楷' if i%2 else '行',character=chr(0x4e00+i),split_group=str(i)) for i in range(30)]
    a = select_test('writer',rs,set(),20,90)
    assert len(a) == len({x['character'] for x in a}) == 20
    assert {x['script'] for x in a} == {'楷','行'}
    assert a == select_test('writer',rs,set(),20,90)


def test_twenty_writer_end_to_end_nested_manifests_and_global_holdout(tmp_path):
    writers = FIRST_TEN+[f'新{i}' for i in range(10)]
    images=[];source=[]
    # Frozen synthetic audit: unique pixel content and independent groups, no GPU.
    for w,writer in enumerate(writers):
        for j in range(24):
            idx=w*24+j;path=tmp_path/f'{idx}.png'
            image=np.full((32,32,3),255,np.uint8)
            image[w%30+1,j%30+1]=[0,idx%255,(idx//255)%255]
            Image.fromarray(image).save(path)
            row=dict(image_path=path.name,calligrapher=writer,calligrapher_id=str(w),script='楷',script_id='0',
                character=chr(0x4e00+j),character_id=str(j),glyph_id=str(j),img_id=str(idx),aug='',src_image_path=path.name)
            source.append(row)
            images.append(dict(image_path=path.name,metadata=[row],roles=['train'],flags=[],hard_issues=[],
                duplicate_group='d'+str(idx),split_group='s'+str(idx),pixel_sha256=sha(path),file_sha256=sha(path)))
    scan=tmp_path/'scan';scan.mkdir();(scan/'inputs').mkdir()
    for role in ('train','eval','top5'):
        write_csv(scan/'inputs'/f'{role}.csv',source if role=='train' else [source[0]])
    images[0]['roles'].append('eval')
    write_json(scan/'images.json',images)
    write_json(scan/'manifest.json',dict(status='complete',root=str(tmp_path),near_scope='synthetic',
        images_sha256=sha(scan/'images.json'),inputs={role:dict(sha256=sha(scan/'inputs'/f'{role}.csv')) for role in ('train','eval','top5')}))
    d=tmp_path/'decisions.json';write_json(d,{})
    out=tmp_path/'cohorts'
    build(['--scan',str(scan),'--output',str(out),'--decisions',str(d),'--min-writer','1','--per-writer','2','--per-writer-val','1'])
    previous_train,previous_test=set(),set()
    for n in (2,5,10,20):
        tr=rows(out/f'top{n}/train.csv');ev=rows(out/f'top{n}/test.csv')
        ts={r['image_path'] for r in tr};es={r['image_path'] for r in ev}
        assert previous_train <= ts and previous_test <= es
        assert not ts & {r['image_path'] for r in rows(out/'all_test.csv')}
        assert not ts & {r['image_path'] for r in rows(out/'all_validation.csv')}
        assert not es & {r['image_path'] for r in rows(out/'all_validation.csv')}
        assert len(ev)==n*2
        assert len(rows(out/f'top{n}/validation.csv'))==n
        previous_train,previous_test=ts,es
    assert len(previous_test)==40
    assert len(read_json(out/'top20/tables/callig_remap.json'))==20
