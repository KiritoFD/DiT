"""Check Pixel D1 on EXACT DiT target latents, fixed matched triplets only."""
import json
from pathlib import Path
import sys
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from diffusers import AutoencoderKL
from src.model.writer_style_pixel import WriterStylePixel
from tools.analysis.audit_d1_vae_bridge import pixel_input
from tools.analysis.accept_writer_style_pixel import grouped
from tools.train_writer_style_pixel import sha


def main():
    torch.set_num_threads(4);base=Path('/root/Workspace/xy/DiT')
    pixel=base/'scratch/writer_pixel_d0d1_20261016_01/assets/D1'
    data=torch.load('assets/data/data.pt',map_location='cpu',weights_only=False)
    ck=torch.load(pixel/'best.pt',map_location='cpu',weights_only=False);assert ck['step']==9500
    d1=WriterStylePixel(len(data['slots'])).cuda().eval().requires_grad_(False);d1.load_state_dict(ck['model'],strict=True)
    vae=AutoencoderKL.from_pretrained('data/pretrained/sd-vae-ft-ema').cuda().eval().requires_grad_(False)
    indices=np.unique(np.concatenate([data['matched_pairs'][k].ravel() for k in ('test_char','test_source')]))
    raw=np.load(pixel/'clean_features.npz')['embedding'];recon=raw.copy()
    with torch.no_grad():
        for start in range(0,len(indices),16):
            ii=indices[start:start+16];z=data['z'][ii].cuda().float()
            with torch.autocast('cuda',dtype=torch.bfloat16):
                rgb=(vae.decode(z/.18215).sample.float().clamp(-1,1)+1)/2
                e=d1(pixel_input(rgb))['embedding']
            recon[ii]=e.float().cpu().numpy()
    tr=data['splits']['train'];mixed=recon.copy();mixed[tr]=raw[tr]
    report={'pixel_d1_sha256':sha(pixel/'best.pt'),'latent_data_manifest_sha256':sha('assets/data/manifest.json'),
            'n_decoded':len(indices),'protocols':{}}
    for name,features in [('raw',raw),('recon_clean_bank',mixed),('recon_recon_triplets',recon)]:
        report['protocols'][name]={k:grouped(features,data['matched_pairs'][k],data) for k in ('test_char','test_source')}
    result=report['protocols']['recon_clean_bank']
    report['pass']=bool(result['test_char']['by_char']['accuracy']>=.57 and result['test_source']['by_char']['accuracy']>=.55)
    report['note']='Fixed matched union only, not a full reconstructed retrieval gallery; full mode-roundtrip audit is separate.'
    Path('assets/exact_shard_bridge.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('EXACT_SHARD_BRIDGE',json.dumps(report),flush=True)
    if not report['pass']:raise SystemExit('Exact target-latent bridge failed; no automatic encoder training')


if __name__=='__main__':main()
