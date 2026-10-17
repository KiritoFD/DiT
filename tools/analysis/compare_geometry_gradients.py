"""Independent transfer is the endpoint; all candidates remain review-only."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from tools.analysis.run_style_fingerprint_audit import interval


def main():
    p=argparse.ArgumentParser();p.add_argument('--work',required=True);a=p.parse_args();work=Path(a.work)
    reports={arm:json.loads((work/f'assets/gradient_{arm}/results.json').read_text(encoding='utf-8')) for arm in ('D1','D2','D3')}
    first=reports['D1'];metadata=first['metadata'];chars=np.array([r['character'] for r in metadata]);sources=np.array([r['gt_source_proxy'] for r in metadata]);seeds=np.array([r['seed'] for r in metadata])
    assert len(set(chars))>=48 and len(set(seeds))>=3
    for arm,r in reports.items():
        assert r['metadata']==metadata and r['source']['source_latent_sha256']==first['source']['source_latent_sha256']
        assert r['source']['independent_ce_sha256']==first['source']['independent_ce_sha256']
        for name in ('baseline','random_0.01'):
            for key in ('pixel_D1_margin','independent_CE_margin'):
                assert np.allclose(r['variants'][name]['per_image'][key],first['variants'][name]['per_image'][key],atol=1e-6),f'Nonpaired external baseline: {arm}/{name}/{key}'
    output={'n':len(chars),'distinct_characters':len(set(chars)),'seeds':sorted(map(int,set(seeds))),
        'paired_generation_and_judges_verified':True,'arms':{},'paired_transfer_changes':{},
        'limitations':['Source clusters refer to GT evaluation references, not known generation provenance.',
                      'Roundtrip error is not a true latent manifold distance.','No DiT training follows automatically.']}
    vectors={}
    for arm,r in reports.items():
        vv=r['variants'];result={'independent_transfer':{},'roundtrip':{name:value.get('roundtrip') for name,value in vv.items()}}
        vectors[arm]={}
        for key in ('pixel_D1_margin','independent_CE_margin'):
            delta=np.array(vv['style_0.01']['per_image'][key])-vv['random_0.01']['per_image'][key]
            vectors[arm][key]=delta
            result['independent_transfer'][key]={'by_character':interval(delta,chars),'by_reference_source_proxy':interval(delta,sources),
                'per_seed':{str(seed):{'mean_delta':float(delta[seeds==seed].mean()),'by_character':interval(delta[seeds==seed],chars[seeds==seed])} for seed in sorted(set(seeds))}}
        result['means']={name:entry['means'] for name,entry in vv.items()}
        output['arms'][arm]=result
    for hi,lo in (('D2','D1'),('D3','D2'),('D3','D1')):
        output['paired_transfer_changes'][hi+'-'+lo]={key:{'by_character':interval(vectors[hi][key]-vectors[lo][key],chars),
            'by_reference_source_proxy':interval(vectors[hi][key]-vectors[lo][key],sources)} for key in vectors[hi]}
    output['next_action']='Review representation and independent-transfer results together. Even positive outcomes do not authorize automatic DiT post-training.'
    (work/'assets/gradient_comparison.json').write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf-8')
    print('GEOMETRY_GRADIENT_COMPARISON_COMPLETE',flush=True)


if __name__=='__main__':main()
