"""F1 strict heldout audit; F3 matched nuisance controls use the same triplets."""
from pathlib import Path
import numpy as np
import torch
from tools.three_day.contracts import save,sha
from .teachers import Teacher,payloads
from .metrics import retrieval


def main(root):
    root=Path(root);pixel,latent=payloads();out=root/'F1';out.mkdir(exist_ok=True)
    report={'protocol':'existing fixed 847/196 matched triplets, train-only cross-character gallery',
        'limitations':['Source proxy and finite image-disjoint subset remain limited.',
                      'Heldout audit only; never use test metrics to tune a generator.'], 'teachers':{}}
    if (root/'style_teacher_audit.json').exists():
        import json
        report=json.loads((root/'style_teacher_audit.json').read_text())
    for name in ('DINO','pixel','D1','D2','D3','pixel_patch'):
        if name in report['teachers'] and (out/(name+'_features.npy')).exists():continue
        teacher=Teacher(name,root,len(pixel['slots'])).cuda();features=[]
        with torch.inference_mode():
            for start in range(0,len(pixel['gray']),128):
                x=(latent['z'][start:start+128].cuda().float() if teacher.domain=='latent'
                   else pixel['gray'][start:start+128].cuda().float()/255)
                with torch.autocast('cuda',dtype=torch.bfloat16):e=teacher(x)
                features.append(e.float().cpu().numpy())
        f=np.concatenate(features);np.save(out/(name+'_features.npy'),f)
        report['teachers'][name]={s:retrieval(f,pixel,s) for s in ('val_char','test_char','test_source')}
        report['teachers'][name]['feature_sha256']=sha(out/(name+'_features.npy'))
        save(root/'style_teacher_audit.json',report);print('F1_TEACHER_DONE',name,flush=True)
        del teacher


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();torch.set_num_threads(8);main(a.root)
