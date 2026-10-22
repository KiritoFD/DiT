"""Small data contracts shared by the accepted generic curriculum runner and tests."""
from collections import defaultdict
import csv
import hashlib
from pathlib import Path

import numpy as np
from skimage.metrics import structural_similarity


def read_rows(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as handle:
        rows=list(csv.DictReader(handle))
    if not rows:raise ValueError('Empty collection: '+str(path))
    return rows


def write_rows(path,rows):
    fields=list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('w',encoding='utf-8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields)
        writer.writeheader();writer.writerows(rows)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class NearIndex:
    """Nine 7-bit bands find all candidates within eight bits of a63-bit fingerprint."""
    def __init__(self):self.bands=defaultdict(set);self.values=[]

    def add(self,record,thumb):
        index=len(self.values);self.values.append((record,thumb));code=int(record['phash'])
        for band in range(9):self.bands[band,(code>>(7*band))&127].add(index)

    def matches(self,record,thumb,threshold=.90):
        code=int(record['phash']);indices=set()
        for band in range(9):indices.update(self.bands.get((band,(code>>(7*band))&127),()))
        for index in indices:
            other,image=self.values[index]
            if (code^int(other['phash'])).bit_count()>8:continue
            if np.abs(thumb.astype(float)-image.astype(float)).mean()>255*.08:continue
            if structural_similarity(thumb,image,data_range=255)>=threshold:return True
        return False


def normalized(row,generic):
    row=dict(row);row['character_id']=str(ord(row['character']))
    row['glyph_id']=str(int(row['script_id'])*0x110000+ord(row['character']))
    row['generic']='1' if generic else '0'
    if generic:
        row['original_calligrapher_id']=row.get('calligrapher_id','')
        row['calligrapher_id']='-1';row['calligrapher']='UNKNOWN'
    return row
