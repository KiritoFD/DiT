"""Paired D1-D0 comparison, created only after both final acceptances exist."""
import json
from pathlib import Path
import numpy as np
import torch
from tools.analysis.run_style_fingerprint_audit import interval


def compare(root,data_root):
    root=Path(root);reports=[json.loads((root/arm/'acceptance.json').read_text()) for arm in ('D0','D1')]
    configs=[json.loads((root/arm/'config.json').read_text()) for arm in ('D0','D1')]
    for key in ('seed','batch','steps','lr','data_manifest_sha256','initial_parameter_sha256'):
        assert configs[0][key]==configs[1][key],key
    traces=[[json.loads(line) for line in (root/arm/'paired_input_trace.jsonl').read_text().splitlines()] for arm in ('D0','D1')]
    assert traces[0]==traces[1],'D0/D1 input/augmentation traces diverged'
    data=torch.load(Path(data_root)/'data.pt',map_location='cpu',weights_only=False)
    features=[np.load(root/arm/'clean_features.npz')['embedding'] for arm in ('D0','D1')]
    output={'paired_control_checks_passed':True,'trace_steps':[x['step'] for x in traces[0]],
            'best_steps':[r['step'] for r in reports],'paired_matched_improvement':{},
            'arms':dict(zip(('D0','D1'),reports)),
            'next_action':'Research review only. No automatic DiT training.'}
    for key,pairs in data['matched_pairs'].items():
        if key=='val_char':continue
        scores=[]
        for f in features:
            margin=(f[pairs[:,0]]*f[pairs[:,1]]).sum(1)-(f[pairs[:,0]]*f[pairs[:,2]]).sum(1)
            scores.append((margin>0).astype(float)+.5*(margin==0))
        diff=scores[1]-scores[0]
        output['paired_matched_improvement'][key]={'definition':'mean D1-D0 accuracy difference; interval accuracy field is a difference, not a probability',
            'by_char':interval(diff,data['char'][pairs[:,0]]),'by_source':interval(diff,data['source'][pairs[:,0]])}
    supplement=root/'source_disjoint_expanded.npz'
    if supplement.exists():
        pairs=np.load(supplement)['pairs'];output['expanded_source_image_disjoint']={}
        pixel=Path('/root/Workspace/xy/DiT/scratch/writer_pixel_d0d1_20261016_01/assets')
        models={'latent_D0':features[0],'latent_D1':features[1],
                'pixel_D0':np.load(pixel/'D0/clean_features.npz')['embedding'],
                'pixel_D1':np.load(pixel/'D1/clean_features.npz')['embedding']}
        for name,f in models.items():
            m=(f[pairs[:,0]]*f[pairs[:,1]]).sum(1)-(f[pairs[:,0]]*f[pairs[:,2]]).sum(1)
            correct=(m>0).astype(float)+.5*(m==0)
            output['expanded_source_image_disjoint'][name]={'by_char':interval(correct,data['char'][pairs[:,0]]),
                'by_source':interval(correct,data['source'][pairs[:,0]])}
    (root/'comparison.json').write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf-8')
    print('PIXEL_D0_D1_COMPARISON_COMPLETE',flush=True)
