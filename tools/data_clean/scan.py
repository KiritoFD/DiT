"""Scan base (not augmented) train/eval CSVs; never change original pixels."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import itertools
from pathlib import Path
import shutil

import numpy as np
from PIL import Image, UnidentifiedImageError
import scipy
from scipy.fft import dctn
from scipy.ndimage import label, find_objects
import skimage
from skimage.metrics import structural_similarity

from .common import Groups, key, rows, sha, write_csv, write_json


# Proposal families only. Never rewrites character/glyph IDs or STD conditions.
ALIASES = {'靑': '青'}


def extract(path):
    try:
        with Image.open(path) as im:
            rgb = np.asarray(im.convert('RGB'))
            gray = np.asarray(im.convert('L'))
    except (OSError, UnidentifiedImageError) as exc:
        return dict(error=str(exc), hard_issues=['unreadable'], flags=[]), None
    h,w = gray.shape
    mask = gray < 128
    area = int(mask.sum())
    hard = []
    if (h,w) != (256,256):
        hard.append('wrong_canvas')
    if area < 25:
        hard.append('near_empty')
    labels,n = label(mask, structure=np.ones((3,3)))
    sizes = np.bincount(labels.ravel());sizes[0] = 0
    tiny = int(sizes[sizes < 25].sum())
    boxes = find_objects(labels)
    bars = []
    for i,box in enumerate(boxes,1):
        if box is None or sizes[i] < .015*h*w:
            continue
        ys,xs = box
        height,width = ys.stop-ys.start,xs.stop-xs.start
        cx,cy = (xs.stop+xs.start)/2/w,(ys.stop+ys.start)/2/h
        if ((height > .72*h and width < .28*w and (cx < .23 or cx > .77)) or
                (width > .72*w and height < .20*h and (cy < .15 or cy > .85))):
            bars.append([xs.start,ys.start,xs.stop,ys.stop])
    border = np.ones_like(mask)
    border[8:-8,8:-8] = False
    edge_fraction = float((mask & border).sum()/max(1,area))
    flags = []
    if bars: flags.append('peripheral_bar')
    # Thin scan frames can be below the large-component area threshold.
    # Long edge projections only propose quarantine; legitimate strokes can be restored.
    edge_rows = mask.mean(1) >= .80
    edge_cols = mask.mean(0) >= .80
    thin_frame = False
    for projection,length in ((edge_rows,h),(edge_cols,w)):
        segments,count = label(projection)
        for segment in range(1,count+1):
            positions = np.flatnonzero(segments==segment)
            if len(positions)<=3 and (positions[0]<.065*length or positions[-1]>.935*length):
                thin_frame = True
    if thin_frame:flags.append('thin_edge_frame')
    if area/(h*w) > .55: flags.append('very_dark_or_inverted')
    if area/(h*w) < .015: flags.append('very_low_ink')
    if tiny/max(1,area) > .12: flags.append('many_small_fragments')
    if edge_fraction > .28: flags.append('edge_heavy')
    if mask.any():
        y,x = np.where(mask)
        crop = gray[y.min():y.max()+1,x.min():x.max()+1]
    else:
        crop = gray
    thumb = np.asarray(Image.fromarray(crop).resize((64,64),Image.Resampling.BILINEAR))
    small = np.asarray(Image.fromarray(thumb).resize((32,32),Image.Resampling.BILINEAR),dtype=float)
    coefficients = dctn(small,type=2,norm='ortho')[:8,:8].ravel()[1:]
    bits = coefficients > np.median(coefficients)
    fingerprint = sum(int(bit)<<j for j,bit in enumerate(bits))
    return dict(file_sha256=sha(path), rgb_sha256=hashlib.sha256(rgb.tobytes()+str(rgb.shape).encode()).hexdigest(),
        pixel_sha256=hashlib.sha256(gray.tobytes()+str(gray.shape).encode()).hexdigest(),
        width=w,height=h,ink_area=area,ink_ratio=area/(h*w),components=int(n),
        tiny_ink_fraction=tiny/max(1,area),edge_fraction=edge_fraction,bar_boxes=bars,
        phash=str(fingerprint),hard_issues=hard,flags=flags),thumb


def link_duplicates(records, thumbnails):
    duplicate = Groups(len(records))
    exact = {}
    families = defaultdict(list)
    for i,r in enumerate(records):
        pixel = r.get('pixel_sha256')
        if not pixel: continue
        if pixel in exact: duplicate.join(i,exact[pixel])
        else: exact[pixel] = i
        for char in {ALIASES.get(x['character'],x['character']) for x in r['metadata']}:
            families[char].append(i)
    proposals = []
    checked = set()
    for family,indices in sorted(families.items()):
        for a,b in itertools.combinations(indices,2):
            if (a,b) in checked: continue
            checked.add((a,b))
            ra,rb = records[a],records[b]
            if ra['pixel_sha256'] == rb['pixel_sha256']: continue
            hamming = (int(ra['phash'])^int(rb['phash'])).bit_count()
            if hamming > 8: continue
            x,y = thumbnails[a],thumbnails[b]
            mae = float(np.abs(x.astype(float)-y.astype(float)).mean()/255)
            if mae > .08: continue
            mx,my = x < 128,y < 128
            iou = float((mx & my).sum()/max(1,(mx | my).sum()))
            if iou < .75: continue
            ssim = float(structural_similarity(x,y,data_range=255))
            accepted = ssim >= .985 and iou >= .95 and mae <= .02
            # Lower threshold is a review proposal, never an automatic merge.
            if ssim < .90: continue
            if accepted: duplicate.join(a,b)
            proposals.append(dict(a=ra['image_path'],b=rb['image_path'],family=family,
                hamming=hamming,mae=mae,iou=iou,ssim=ssim,accepted=accepted))
    labels = duplicate.labels()
    split = Groups(len(records))
    sources = {}
    for i,r in enumerate(records):
        split.join(i,labels[i])
        for meta in r['metadata']:
            source = key(meta.get('src_image_path',''))
            if not source: continue
            if source in sources: split.join(i,sources[source])
            else: sources[source] = i
    split_labels = split.labels()
    for i,r in enumerate(records):
        r['duplicate_group'] = 'd'+str(labels[i])
        r['split_group'] = 's'+str(split_labels[i])
    return proposals


def run(args):
    out = Path(args.output)
    out.mkdir(parents=True,exist_ok=False)
    (out/'inputs').mkdir()
    inputs = {'train':args.train_csv,'eval':args.eval_csv,'top5':args.top5_csv}
    manifest = dict(schema='calligraphy-clean-scan-v1',status='incomplete',root=str(Path(args.root).resolve()),
        inputs={},aliases_for_proposals=ALIASES,near_scope='same_character_or_explicit_alias_family',
        near_accept=dict(ssim=.985,iou=.95,mae=.02,hamming=8),
        versions=dict(numpy=np.__version__,scipy=scipy.__version__,skimage=skimage.__version__))
    combined = {}
    for role,path in inputs.items():
        original = Path(path)
        shutil.copy2(original,out/'inputs'/f'{role}.csv')
        manifest['inputs'][role] = dict(path=str(original.resolve()),sha256=sha(original))
        if role == 'top5': continue
        for row in rows(original):
            if row.get('aug','').strip():
                raise ValueError('Scan base/noaug rows, not an augmented CSV')
            k = key(row['image_path'])
            obj = combined.setdefault(k,dict(image_path=k,roles=[],metadata=[]))
            if role not in obj['roles']:obj['roles'].append(role)
            if row not in obj['metadata']:obj['metadata'].append(row)
    write_json(out/'manifest.json',manifest)
    records = [combined[k] for k in sorted(combined)]
    paths = [Path(args.root)/r['image_path'] for r in records]
    thumbnails = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i,(record,(stats,thumb)) in enumerate(zip(records,pool.map(extract,paths)),1):
            record.update(stats);thumbnails.append(thumb)
            if i%2000 == 0:print(f'scan {i}/{len(records)}',flush=True)
    print('linking duplicate groups',flush=True)
    proposals = link_duplicates(records,thumbnails)
    write_json(out/'images.json',records)
    write_csv(out/'near_pairs.csv',proposals,['a','b','family','hamming','mae','iou','ssim','accepted'])
    counts = Counter(r['duplicate_group'] for r in records)
    manifest.update(status='complete',n_images=len(records),
        duplicate_groups=sum(n>1 for n in counts.values()),duplicate_excess=sum(n-1 for n in counts.values()),
        near_proposals=len(proposals),near_accepted=sum(p['accepted'] for p in proposals),
        flags=dict(Counter(f for r in records for f in r['flags'])),
        hard_issues=dict(Counter(f for r in records for f in r['hard_issues'])),images_sha256=sha(out/'images.json'))
    write_json(out/'manifest.json',manifest)
    print(manifest,flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('train-csv','eval-csv','top5-csv','output'):p.add_argument('--'+name,required=True)
    p.add_argument('--root',default='.')
    p.add_argument('--workers',type=int,default=8)
    a = p.parse_args(argv)
    if a.workers < 1:p.error('workers must be positive')
    run(a)


if __name__ == '__main__':main()
