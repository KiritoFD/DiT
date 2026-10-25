from collections import Counter
import json

import pytest

from tools.cohort_eval.datasets import (
    build_probes, read_rows, write_rows, seeded_rows, image_key, select_scripts, resolve_collection,
)


def test_noise_is_image_stable_and_frozen_seeds_are_preserved():
    items=[dict(image_path='a.png'),dict(image_path='b.png',eval_noise_seed='123')]
    a=seeded_rows(items)
    assert a[1]['eval_noise_seed']=='123'
    assert seeded_rows(items[::-1])[1]['eval_noise_seed']==a[0]['eval_noise_seed']
    with pytest.raises(ValueError):seeded_rows([dict(image_path='x',eval_noise_seed='-1')])


def test_script_matched_half_is_nested_and_sampling_is_order_independent():
    pool=[dict(image_path=f'{i}.png',script_id='0' if i<24 else '3') for i in range(40)]
    ref=[dict(script_id='0' if i<12 else '3') for i in range(20)]
    equal=select_scripts(pool,ref,20,90)
    half=select_scripts(equal,ref,10,90)
    assert Counter(r['script_id'] for r in equal)=={'0':12,'3':8}
    assert Counter(r['script_id'] for r in half)=={'0':6,'3':4}
    assert {image_key(r) for r in half}<={image_key(r) for r in equal}
    assert equal==select_scripts(pool[::-1],ref,20,90)


def test_four_levels_are_fixed_nested_and_do_not_change_frozen_splits(tmp_path):
    data=tmp_path/'datasets_final';data.mkdir()
    (data/'manifest.json').write_text('{}')
    trains=[];tests=[];vals=[]
    for w in range(20):
        for j in range(61):
            r=dict(image_path=f'w{w}_{j}.png',calligrapher_id=str(w),script_id='0',character='字')
            (trains if j<40 else tests if j<60 else vals).append(r)
    for n in (2,5,10,20):
        p=data/f'top{n}';p.mkdir()
        for name,items in [('train',trains),('test',tests),('validation',vals)]:
            write_rows(p/f'{name}.csv',[r for r in items if int(r['calligrapher_id'])<n])
    build_probes(tmp_path)
    previous=set()
    heldout={image_key(r) for r in tests+vals}
    for n in (2,5,10,20):
        equal=read_rows(resolve_collection(f'top{n}/train_equal',tmp_path))
        half=read_rows(resolve_collection(f'top{n}/train_half',tmp_path))
        keys={image_key(r) for r in equal}
        assert len(equal)==20*n and len(half)==10*n
        assert previous<=keys and not keys&heldout
        assert {image_key(r) for r in half}<=keys
        assert set(Counter(r['calligrapher_id'] for r in equal).values())=={20}
        previous=keys
    with pytest.raises(FileExistsError):build_probes(tmp_path)


def test_nested_metric_aggregation_joins_identity_and_checks_noise():
    from tools.cohort_eval.batch import subset_rows,summarize
    source=[dict(image_path=f'{i}.png',img_id=str(i),eval_noise_seed=str(i+10)) for i in range(4)]
    metrics=[dict(idx=str(i),img_id=str(i),ssim=str(i/10),lpips='.2',ink_iou='.3',
        frag_ratio='2',soft_ink_ratio='1',mse='.4',skel_iou='.1',ink_ssim='.5',
        hole_pred='.04',hole_gt='.03') for i in range(4)]
    selected=subset_rows(source,metrics[::-1],[source[3],source[1]])
    summary=summarize(selected)
    assert summary['n']==2 and summary['ssim_mean']==pytest.approx(.2)
    assert [r['image_path'] for r in selected]==['3.png','1.png']
    wrong=dict(source[1],eval_noise_seed='999')
    with pytest.raises(ValueError,match='noise'):subset_rows(source,metrics,[wrong])
