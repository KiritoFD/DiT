"""Paired-control review of a frozen-latent pilot; never starts training."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from tools.analysis.run_style_fingerprint_audit import interval


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();out=Path(a.out)
    report=json.loads((out/'results.json').read_text(encoding='utf-8'));v=report['variants'];base=v['baseline']['means']
    characters=np.array([r['character'] for r in report['metadata']]);comparisons={}
    for control in ('random_0.01','inverse_0.01'):
        comparisons[control]={}
        for key in ('latent_D1_margin','pixel_D1_margin','independent_CE_margin','ocr_head_agrees_gt','ink_fraction'):
            delta=np.array(v['style_0.01']['per_image'][key])-v[control]['per_image'][key]
            comparisons[control][key]=interval(delta,characters)
    passes=[];checks={}
    for name in ('style_0.001','style_0.003','style_0.01'):
        means=v[name]['means'];deltas=v[name]['paired_delta_vs_baseline']
        c={'pixel_margin_character_ci_positive':deltas['pixel_D1_margin']['by_character']['ci95'][0]>0,
           'external_margin_character_ci_positive':deltas['independent_CE_margin']['by_character']['ci95'][0]>0,
           'ink_iou_ge98pct':means['ink_iou_to_baseline']>=.98,
           'lpips_to_baseline_le001':means['lpips_to_baseline']<=.01,
           'ocr_proxy_not_worse':means['ocr_head_agrees_gt']>=base['ocr_head_agrees_gt']}
        checks[name]=c
        if all(c.values()):passes.append(name)
    tensors=torch.load(out/'latent_steps.pt',map_location='cpu',weights_only=True);g=tensors['gradient'].float();z=tensors['z'].float()
    gray=torch.stack([torch.from_numpy(np.asarray(Image.open(out/'baseline'/f'g{i}.png').convert('L')).copy()).float()/255 for i in range(len(g))])[:,None]
    ink=F.adaptive_avg_pool2d((gray<.55).float(),32)>=.05
    neighborhood=F.max_pool2d(ink.float(),3,1,1)>0
    energy=g.square().sum(1,keepdim=True);total=energy.flatten(1).sum(1).clamp_min(1e-20)
    outside=(energy*~neighborhood).flatten(1).sum(1)/total
    constant=g.mean((2,3),keepdim=True).square().sum((1,2,3))*32*32/total
    gain=(g*z).flatten(1).sum(1).square()/(z.square().flatten(1).sum(1)*total).clamp_min(1e-20)
    report['gradient_concentration']={'outside_ink_neighborhood_energy_fraction_mean':float(outside.mean()),
        'outside_ink_neighborhood_area_fraction_mean':float((~neighborhood).float().mean()),
        'spatially_constant_channel_mode_energy_fraction_mean':float(constant.mean()),
        'global_gain_mode_energy_fraction_mean':float(gain.mean()),
        'note':'Descriptive gradient projections, not causal proof of shortcut use.'}
    report['paired_control_comparisons']=comparisons
    report['pilot_gate']={'definition':'Engineering pilot only, not formal generator-supervision acceptance; <=1% latent RMS; independent margins positive by-character intervals plus content preservation.',
        'checks':checks,'passing_steps':passes,'ready_for_dit_training':False,
        'next_action':'Expand heldout character/seed coverage and audit trainable DiT/FM gradients before any training; no automatic D2/D3.' if passes else 'No DiT training. Investigate representation/gradient mismatch; geometry-aware D2/D3 remain untested hypotheses, not presumed fixes.'}
    report['practical_transfer_review']={
        'numeric_sign_gate_is_not_semantic_success':True,
        'one_pct_style_exceeds_random_pixel_margin_ci':comparisons['random_0.01']['pixel_D1_margin']['ci95'][0]>0,
        'one_pct_style_exceeds_random_external_margin_ci':comparisons['random_0.01']['independent_CE_margin']['ci95'][0]>0,
        'accepted_for_dit_training':False,
        'interpretation':'Weak positive cross-encoder margin transfer exists, but large self-score improvement does not translate to robust independent writer identification. Single-step pilot only; do not infer D2 architecture will fix it.'}
    (out/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('GRADIENT_PILOT_REVIEW',json.dumps(report['pilot_gate']),flush=True)


if __name__=='__main__':main()
