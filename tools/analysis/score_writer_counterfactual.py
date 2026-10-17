"""Independent writer judgment, latent Critic logits and paired brush diagnostics."""
import csv
import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from collections import defaultdict
from types import SimpleNamespace
import numpy as np
import torch
from PIL import Image

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from tools.eval_gen_with_disc import Enc, ImgSet
from src.eval.style.bank import resolve
from src.eval.style.brush import features_from_image,aggregate,bsd
from src.eval.style.encoder import StyleEncoder
from src.eval.in_mem_eval import _get_cache
from src.loss.critic_loss import CriticLoss


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--device',default='cpu')
    opt=parser.parse_args()
    torch.set_num_threads(4);torch.set_grad_enabled(False)
    dev=opt.device;out=ROOT/'assets/writer_counterfactual_20261014'
    meta=list(csv.DictReader(open(out/'eval_map.csv',encoding='utf-8-sig')))
    manifest=json.loads((out/'manifest.json').read_text(encoding='utf-8'))
    ck=torch.load(ROOT/'assets/style_disc_ce_holdout.pt',map_location='cpu',weights_only=False)
    enc=Enc(d=ck['d'],n_cls=ck['n_cls']).to(dev);enc.load_state_dict(ck['model']);enc.eval()
    rows=list(csv.DictReader(open(ROOT/'exp-std/csv/train.csv',encoding='utf-8-sig')))
    random.Random(0).shuffle(rows);rows=rows[:6000]
    roots=[str(ROOT),str(ROOT/'data'),str(ROOT/'exp-std')]
    paths=[];slots=defaultdict(list);brush=defaultdict(list)
    for r in rows:
        if r.get('aug','').strip():continue
        p=resolve(r['image_path'],roots)
        assert p,r['image_path']
        key=r['calligrapher']+'·'+r['script']
        slots[key].append(len(paths));paths.append(p)
    bank=ImgSet(paths,tag='independent bank').emb_all(enc,dev)
    protos={k:bank[ix].mean(0) for k,ix in slots.items()}
    stdkeys=defaultdict(list);stdpaths=[]
    for i,r in enumerate(meta):
        if i%10:continue
        stdkeys['std·'+r['script']].append(len(stdpaths))
        stdpaths.append(resolve(r['std_path'],roots))
    embstd=ImgSet(stdpaths,tag='std bank').emb_all(enc,dev)
    for k,ix in stdkeys.items():protos[k]=embstd[ix].mean(0)
    protos={k:v/np.linalg.norm(v) for k,v in protos.items()}
    keys=sorted(protos);P=np.stack([protos[k] for k in keys])
    gp=[str(out/f'g{i}.png') for i in range(len(meta))]
    E=ImgSet(gp,tag='counterfactual').emb_all(enc,dev);sims=E@P.T
    # Fixed, per-writer/script brush references; balance up to100 real examples.
    for k,ix in slots.items():brush[k]=aggregate([features_from_image(paths[j]) for j in ix[:100]])
    per=[]
    for i,r in enumerate(meta):
        key=r['calligrapher']+'·'+r['script'];ci=keys.index(key)
        other=[j for j,k in enumerate(keys) if k!=key and not k.startswith('std·') and k.endswith('·'+r['script'])]
        predicted=keys[int(sims[i].argmax())]
        per.append(dict(manifest['per_image'][i],prototype_pred=predicted,
            prototype_top1=predicted==key,
            writer_margin=float(sims[i,ci]-max(sims[i,j] for j in other)),
            std_margin=float(sims[i,ci]-sims[i,keys.index('std·'+r['script'])]),
            bsd_width=bsd(features_from_image(gp[i]),brush[key])['width']))
    # Score the exact sampled z rather than re-encoding a PNG into a new latent.
    audit_path=ROOT/'assets/v85_diagnostics_20261014_audit.json'
    if not audit_path.exists():audit_path=ROOT/'assets/v85_diagnostics_20261014/audit.json'
    conf=json.loads(audit_path.read_text(encoding='utf-8'))['runs']['v84A']['config']
    args=SimpleNamespace(**conf);args.std_soft_root='exp-std/data/std_soft_glyphs_v1'
    cache=_get_cache('exp-std/csv/eval200_fixed.csv',187,None,'',args)
    critic=CriticLoss(str(ROOT/'assets/writer_brush_critic.pt'),device=dev)
    for group,idx in enumerate(manifest['selected_eval_indices']):
        z=torch.load(out/f'latent_{idx}.pt',map_location=dev,weights_only=True)
        g=cache['skels_latent'][idx:idx+1].repeat(10,1,1,1).to(dev)
        assert hashlib.sha256(g[0].cpu().numpy().tobytes()).hexdigest()==per[group*10]['g_sha256']
        si=torch.tensor([critic.s2i[meta[group*10]['script']]]*10,device=dev)
        scores=critic.model(z,g,si)
        for w in range(10):
            rec=per[group*10+w];target=critic.slots.index(rec['writer']+'·'+rec['script'])
            lg=scores['slot_logits'][w].cpu().numpy()
            rec.update(critic_writer_target_logit=float(lg[target]),critic_writer_logits=lg.tolist(),
                critic_brush_logit=float(scores['brush_logit'][w]),critic_slot_pred=critic.slots[int(lg.argmax())])
    (out/'partial_metrics.json').write_text(json.dumps(per,ensure_ascii=False,indent=2),encoding='utf-8')
    # Character-head agreement is a proxy, not ground-truth OCR accuracy.
    charenc=StyleEncoder.load('pretrained_models/eval_classifier.pt',device=dev)
    def char_preds(pp):
        ans=[]
        for i in range(0,len(pp),16):
            aa=[np.asarray(Image.open(p).convert('RGB')) for p in pp[i:i+16]]
            lg,_,_=charenc.net(charenc.to_tensor(aa).to(dev));ans+=lg.argmax(-1).cpu().tolist()
        return ans
    gtps=[resolve(meta[i]['image_path'],roots) for i in range(0,len(meta),10)]
    gtids=char_preds(gtps);stdids=char_preds(stdpaths);predids=char_preds(gp)
    for i,rec in enumerate(per):
        rec.update(char_head_id=predids[i],gt_char_head_id=gtids[i//10],
            char_head_agrees_gt=predids[i]==gtids[i//10],char_head_agrees_std=predids[i]==stdids[i//10])
    summary={}
    for w in range(10):
        rr=[r for r in per if r['writer_id']==w]
        summary[str(w)]={k:float(np.mean([r[k] for r in rr])) for k in
            ['prototype_top1','writer_margin','std_margin','bsd_width','ink','p50','char_head_agrees_gt']}
        summary[str(w)]['writer']=rr[0]['writer']
    result={'n':len(per),'per_writer':summary,'per_image':per,
        'independent_metric':'CE-holdout resnet18 embeddings; 6000 shuffled bank; 23 writer-script +2 STD prototypes',
        'bsd_reference':'per writer-script; first up to100 real images from shuffled6000 bank; unbalanced characters',
        'ocr_proxy':'7765-class char-head agrees with selected GT class; not OCR accuracy',
        'checkpoint_sha256':manifest['checkpoint_sha256'],
        'judge_sha256':hashlib.sha256((ROOT/'assets/style_disc_ce_holdout.pt').read_bytes()).hexdigest(),
        'critic_sha256':hashlib.sha256((ROOT/'assets/writer_brush_critic.pt').read_bytes()).hexdigest()}
    (out/'metrics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print('WRITER_COUNTERFACTUAL_SCORED',json.dumps(summary,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
