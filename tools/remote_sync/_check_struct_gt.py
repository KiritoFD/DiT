# -*- coding: utf-8 -*-
"""检查 5script train 图的 canny/skeleton GT 覆盖率。"""
import csv, os, glob

rows = list(csv.DictReader(open('5script/train.csv', encoding='utf-8')))
ids = set(os.path.basename(r['image_path'])[:-4] for r in rows)
canny = set(os.path.splitext(os.path.basename(p))[0] for p in glob.glob('final_canny/*.png'))
skel = set(os.path.splitext(os.path.basename(p))[0] for p in glob.glob('final_skeleton/*.png'))
print('train rows:', len(rows), 'unique img ids:', len(ids))
print('final_canny files:', len(canny), 'final_skeleton files:', len(skel))
print('canny missing:', len(ids - canny))
print('skel missing:', len(ids - skel))
# eval 集覆盖率
for csvf in ['5script/eval_strata/clean_unseen_triple_100.csv']:
    erows = list(csv.DictReader(open(csvf, encoding='utf-8')))
    eids = set(os.path.basename(r['image_path'])[:-4] for r in erows)
    print(csvf, 'n=', len(eids), 'canny missing:', len(eids - canny), 'skel missing:', len(eids - skel))
