"""Paired heldout D1/D2/D3 acceptance; no automatic generator training."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from src.model.writer_style_latent import build_writer_style_latent
from tools.analysis.run_style_fingerprint_audit import interval
from tools.analysis.run_swd_style_audit import plans


def outcomes(features,pairs):
    a,p,n=features[pairs[:,0]],features[pairs[:,1]],features[pairs[:,2]]
    m=(a*p).sum(1)-(a*n).sum(1)
    return (m>0).astype(float)+.5*(m==0)


def retrieval16(features,trials):
    correct=[];rr=[];counts=[];chars=[];sources=[]
    for t in trials:
        for anchor,writer in ((t['ids'][0],t['w']),(t['ids'][2],t['v'])):
            a=features[anchor].mean(0);a=a/max(np.linalg.norm(a),1e-12);scores=[]
            for slot,ids in sorted(t['candidate_ids'].items()):
                b=features[ids].mean(0);b=b/max(np.linalg.norm(b),1e-12);scores.append((float(a@b),int(slot)))
            scores.sort(reverse=True);rank=next(i+1 for i,(_,s) in enumerate(scores) if s==writer)
            correct.append(float(rank==1));rr.append(1/rank);counts.append(len(scores))
            chars.append('|'.join(t['test_chars']));sources.append(t['source_block'])
    return {'n_queries':len(correct),'top1_by_character_set':interval(correct,chars),'top1_by_source_proxy':interval(correct,sources),
            'mean_reciprocal_rank':float(np.mean(rr)),'candidate_count_quantiles':np.quantile(counts,[0,.5,1]).tolist(),
            'note':'One relevant candidate bundle per writer; report MRR, not multi-relevant mAP. Repeated overlapping sets are not independent.'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--work',required=True);a=p.parse_args();root=Path(a.root);work=Path(a.work)
    torch.set_num_threads(4);data=torch.load(work/'assets/data/data.pt',map_location='cpu',weights_only=False)
    runs={'D1':root/'scratch/writer_latent_d0d1_20261016_01/assets/D1','D2':work/'assets/D2','D3':work/'assets/D3'}
    configs={k:json.loads((r/'config.json').read_text()) for k,r in runs.items()}
    for key in ('seed','batch','steps','lr','data_manifest_sha256','initial_parameter_sha256','architecture','sdf_manifest_sha256'):
        assert configs['D2'][key]==configs['D3'][key],key
    traces={arm:[json.loads(x) for x in (runs[arm]/'paired_input_trace.jsonl').read_text().splitlines()] for arm in ('D2','D3')}
    assert traces['D2']==traces['D3'],'Paired sampling/augmentation/SDF targets diverged'
    d1_trace=[json.loads(x) for x in (runs['D1']/'paired_input_trace.jsonl').read_text().splitlines()]
    for old,new in zip(d1_trace,traces['D2']):
        for key in ('step','sample_ids_sha256','augmented_gray_sha256'):assert old[key]==new[key],f'D1/D2 input trace mismatch: {key}'
    assert len(d1_trace)==len(traces['D2'])
    for key in ('seed','batch','steps','lr','data_manifest_sha256'):assert configs['D1'][key]==configs['D2'][key],key
    features={k:np.load(r/'clean_features.npz')['embedding'] for k,r in runs.items()}
    sets,_=plans(data,5)
    report={'paired_checks_passed':True,'D1_D2_input_trace_checks_passed':True,'trace_steps':[r['step'] for r in traces['D2']],
        'parameters':{k:v['parameters'] for k,v in configs.items()},'arms':{},'paired_differences':{},
        'limitations':['Source directories are proxies, not verified works.','D1 differs in architecture; D2/D3 are exactly paired at initialization and input trace.'],
        'next_action':'Gradient transfer evaluation only. Never automatically start DiT.'}
    sdf=torch.load(work/'assets/sdf32/unclipped_sdf.pt',map_location='cpu',weights_only=True).float().clamp(-1,1)
    supplement=np.load(root/'scratch/writer_latent_d0d1_20261016_01/assets/source_disjoint_expanded.npz')['pairs']
    for name,run in runs.items():
        entry=json.loads((run/'acceptance.json').read_text());entry['set16_retrieval']={key:retrieval16(features[name],sets[key,16]) for key in ('test_char','test_source')}
        scores=outcomes(features[name],supplement)
        entry['fixed_supplemental_image_disjoint']={'by_char':interval(scores,data['char'][supplement[:,0]]),'by_source':interval(scores,data['source'][supplement[:,0]])}
        if name!='D1':
            ck=torch.load(run/'best.pt',map_location='cpu',weights_only=False);m=build_writer_style_latent(len(data['slots']),'geometry').cuda().eval().requires_grad_(False);m.load_state_dict(ck['model'],strict=True)
            entry['clean_geometry_sdf_l1']={}
            with torch.no_grad():
                for key in ('val_char','test_char','test_source'):
                    ix=data['splits'][key];loss=[]
                    for s in range(0,len(ix),128):
                        ids=ix[s:s+128];pred=m(data['z'][ids].cuda().float())['geometry_sdf'];loss.extend((pred-sdf[ids].cuda()).abs().mean((1,2,3)).cpu().tolist())
                    entry['clean_geometry_sdf_l1'][key]=float(np.mean(loss))
            del m;torch.cuda.empty_cache()
        report['arms'][name]=entry
    for hi,lo in (('D2','D1'),('D3','D2'),('D3','D1')):
        d={}
        for key,pairs in data['matched_pairs'].items():
            if key=='val_char':continue
            delta=outcomes(features[hi],pairs)-outcomes(features[lo],pairs)
            d[key]={'by_char':interval(delta,data['char'][pairs[:,0]]),'by_source':interval(delta,data['source'][pairs[:,0]])}
        supplemental=root/'scratch/writer_latent_d0d1_20261016_01/assets/source_disjoint_expanded.npz'
        pairs=np.load(supplemental)['pairs'];delta=outcomes(features[hi],pairs)-outcomes(features[lo],pairs)
        d['fixed_supplemental_image_disjoint']={'n':len(pairs),'by_char':interval(delta,data['char'][pairs[:,0]]),'by_source':interval(delta,data['source'][pairs[:,0]]),
            'note':'Fixed alternative maximal matching inherited from D1, NOT more independent images or replacement of original protocol.'}
        report['paired_differences'][hi+'-'+lo]=d
    (work/'assets/comparison.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');print('GEOMETRY_COMPARISON_COMPLETE',flush=True)


if __name__=='__main__':main()
