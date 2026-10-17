"""Same-PNG metrics and character-clustered paired bootstrap, no inference."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def main():
    import numpy as np
    import torch
    import lpips
    from PIL import Image, ImageDraw, ImageFont
    from src.eval.metrics import ssim, frag_ratio
    from src.eval.metrics_ink import ink_iou
    from src.eval.protocol import EvalRequest, is_complete
    from dataclasses import fields
    from tools.analysis.unet_compare.topology import analyze_image_pair

    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    cli = parser.parse_args()
    root = Path(cli.root)
    rows = list(csv.DictReader(open('exp-std/csv/eval200_fixed.csv', encoding='utf-8-sig')))
    train = list(csv.DictReader(open('exp-std/csv/train_top10_aug_sym.csv', encoding='utf-8-sig')))
    seen = {r['character'] for r in train}
    unseen = np.array([r['character'] not in seen for r in rows])
    names = ['unet15', 'dit40', 'unet20', 'dit50', 'dit60']
    dirs = dict(unet15=root/'unet_raw15000/eval_samples_ctrl/step0015000/eval200fix',
                unet20=root/'unet_raw20000/eval_samples_ctrl/step0020000/eval200fix',
                **{f'dit{k}': root/f'dit/step{k*1000:07d}/eval200fix' for k in (40, 50, 60)})
    writer = json.loads((root/'writer_per_image.json').read_text())
    gt = [np.asarray(Image.open(dirs['dit40']/f'gt{i}.png').convert('RGB')).copy() for i in range(187)]
    assert len(rows) == len(gt) == 187
    result = dict(n=187, unseen_n=int(unseen.sum()), unseen_characters=sorted({r['character'] for i,r in enumerate(rows) if unseen[i]}),
                  states={'unet15':'raw', 'unet20':'raw', 'dit40':'ema', 'dit50':'ema', 'dit60':'ema'},
                  images_seen=dict(unet15=7680000, dit40=7680000, unet20=10240000, dit50=9600000, dit60=11520000),
                  models={}, paired={}, limitations=['U-Net raw vs valid historical DiT EMA; not an isolated architecture ablation',
                    'One generation seed; bootstrap measures evaluation-character variation, not training-seed uncertainty',
                    'LR, batch, parameter count, REPA and conditional architectures may differ; exposure matching alone is not full control'])
    torch.set_num_threads(4)
    perceptual = lpips.LPIPS(net='vgg', verbose=False).cuda().eval()
    truth = torch.from_numpy(np.stack(gt)).permute(0,3,1,2).float()/255
    metrics = ['ssim', 'lpips_vgg', 'ink_iou', 'frag', 'thin_recall', 'missing_components', 'merged_components', 'writer_top1', 'writer_margin']
    for name in names:
        directory = dirs[name]
        protocol = json.loads((directory/'protocol.json').read_text())
        request = EvalRequest(**{f.name:protocol[f.name] for f in fields(EvalRequest) if f.name in protocol})
        complete, why = is_complete(request, str(directory))
        assert complete, (name, why)
        assert protocol['ema'] == name.startswith('dit')
        assert protocol['cfg'] == 1. and protocol['num_steps'] == 50 and protocol['gamma'] == 0.
        assert protocol['dataset_sha256'] == 'f2205eb2fbd34b03915d4dd26b379a107183606683cf3389b5264bcdd6d45a27'
        pred = []
        for i in range(187):
            check = np.asarray(Image.open(directory/f'gt{i}.png').convert('RGB'))
            assert np.array_equal(check, gt[i]), (name, i, 'GT mismatch')
            pred.append(np.asarray(Image.open(directory/f'g{i}.png').convert('RGB')).copy())
        w = {r['row']:r for r in writer[name]}
        assert set(w) == set(range(187)), (name, len(w))
        def measure(i):
            p = pred[i].astype(np.float32)/255
            g = gt[i].astype(np.float32)/255
            topo = analyze_image_pair(directory/f'g{i}.png', directory/f'gt{i}.png')
            return dict(idx=i, character=rows[i]['character'], ssim=ssim(p,g), ink_iou=ink_iou(p,g),
                        frag=frag_ratio(p,g), thin_recall=topo['thin_skeleton_recall'],
                        missing_components=topo['missing_components'], merged_components=topo['merged_components'],
                        writer_top1=float(w[i]['top1']), writer_margin=w[i]['writer_margin'])
        with ThreadPoolExecutor(max_workers=8) as pool:
            records = list(pool.map(measure, range(187)))
        tensor = torch.from_numpy(np.stack(pred)).permute(0,3,1,2).float()/255
        with torch.no_grad():
            for start in range(0,187,8):
                values = perceptual(tensor[start:start+8].cuda()*2-1, truth[start:start+8].cuda()*2-1).flatten().cpu().tolist()
                for j,v in enumerate(values): records[start+j]['lpips_vgg'] = v
        aggregate = {}
        for subset,mask in [('all',np.ones(187,dtype=bool)),('unseen',unseen),('seen',~unseen)]:
            aggregate[subset] = dict(n=int(mask.sum()), **{key:float(np.mean([r[key] for i,r in enumerate(records) if mask[i]])) for key in metrics})
        result['models'][name] = dict(directory=str(directory), protocol=protocol, done_verified=True, aggregate=aggregate, per_image=records,
            png_sha256={str(i):hashlib.sha256((directory/f'g{i}.png').read_bytes()).hexdigest() for i in range(187)})
        print('MEASURED', name, aggregate['all'], flush=True)
    for a,b in [('unet15','dit40'),('unet20','dit50'),('unet20','dit60')]:
        pair = {}
        for subset,mask in [('all',np.ones(187,dtype=bool)),('unseen',unseen)]:
            indices = np.where(mask)[0]
            groups = sorted({rows[i]['character'] for i in indices})
            membership = np.array([[i for i in indices if rows[i]['character']==c] for c in groups], dtype=object)
            counts = np.array([len(g) for g in membership])
            rng = np.random.default_rng(20261018)
            draws = rng.integers(len(groups), size=(5000,len(groups)))
            data = {}
            for key in metrics:
                delta = np.array([result['models'][a]['per_image'][i][key]-result['models'][b]['per_image'][i][key] for i in range(187)])
                sums = np.array([delta[list(g)].sum() for g in membership])
                boot = sums[draws].sum(1)/counts[draws].sum(1)
                data[key] = dict(delta=float(delta[indices].mean()), ci95=np.quantile(boot,[.025,.975]).tolist())
            pair[subset] = dict(n=len(indices), characters=len(groups), metrics=data)
        result['paired'][a+'-'+b] = pair
    (root/'comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    sample_ids = [0,10,30,50,70,100,130,180]
    canvas = Image.new('RGB',(260+160*len(sample_ids),60+180*6),'white')
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
    for col,i in enumerate(sample_ids):draw.text((270+160*col,10),f'idx {i}',fill='black',font=font)
    for row,name in enumerate(['GT',*names]):
        title=name
        if name!='GT':
            m=result['models'][name]['aggregate']['all']
            title+=f"\nSSIM {m['ssim']:.4f}\nLPIPS {m['lpips_vgg']:.4f}\nWriter {m['writer_top1']:.1%}"
        draw.text((10,65+180*row),title,fill='black',font=font)
        for col,i in enumerate(sample_ids):
            path=dirs['dit40']/f'gt{i}.png' if name=='GT' else dirs[name]/f'g{i}.png'
            canvas.paste(Image.open(path).convert('RGB').resize((155,155)),(260+160*col,60+180*row))
    canvas.save(root/'poster.png')
    print('DONE',root/'comparison.json',flush=True)


if __name__=='__main__': main()
