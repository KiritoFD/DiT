"""B2 weights from training coverage, clean GT and STD geometry only; cap 1.5."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import label
from skimage.morphology import skeletonize
from tools.three_day.contracts import read,save,sha


def build(config,out):
    from src.utils.std_soft_glyph import StdSoftGlyphStore
    rows=read(config['data_csv']);counts=Counter(r['character'] for r in rows)
    store=StdSoftGlyphStore(config['std_soft_root'],is_train=False,eval_sigma=0.)
    complexity={}
    for r in rows:
        key=(int(r['script_id']),r['character'])
        if key not in complexity:
            mask=store.get(*key,sigma=0.).numpy().squeeze()>.5
            complexity[key]=float(skeletonize(mask).sum())
    values=np.array([complexity[int(r['script_id']),r['character']] for r in rows])
    ranks=np.argsort(np.argsort(values))/max(len(values)-1,1)
    rare=np.array([1/np.sqrt(counts[r['character']]) for r in rows]);rare/=rare.max()
    def reliable(r):
        with Image.open(r['image_path']) as im:a=np.asarray(im.convert('L').resize((128,128)))<128
        mass=a.mean();components,n=label(a);sizes=np.bincount(components.ravel())[1:]
        tiny=sizes[sizes<3].sum()/max(a.sum(),1)
        return bool(.01<mass<.65 and tiny<.1 and sizes.max(initial=0)>20)
    with ThreadPoolExecutor(max_workers=8) as pool:ok=np.array(list(pool.map(reliable,rows)))
    weights=1+.5*(ranks*rare)*ok
    out=Path(out);np.savez(out,img_ids=np.array([int(r['img_id']) for r in rows]),weights=weights)
    save(out.with_suffix('.json'),dict(csv_sha256=sha(config['data_csv']),rows=len(rows),reliable=int(ok.sum()),
        weights_min=float(weights.min()),weights_max=float(weights.max()),
        rule='1+.5 * STD-skeleton-length rank * inverse-sqrt train character frequency, only quality-passing GT',
        heldout_input_used=False,no_failure_character_list=True,pair_endpoint_distribution='unchanged; ordinary share reweighted'))
    return str(out)
