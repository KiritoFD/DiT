"""Fixed-target correct/wrong/null FM audit, with explicit seen-character strata."""
import argparse
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from .common import setup,save,read,paired_ci


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--csv',required=True)
    p.add_argument('--out',required=True);p.add_argument('--batch',type=int,default=32);a=p.parse_args()
    torch.set_num_threads(8)
    model,c,ck,cache,rows,legal,process=setup(a.checkpoint,a.csv,a.out)
    from src.eval.decoder import load_eval_vae
    vae=load_eval_vae('cuda',c.vae_path);targets=[]
    with torch.inference_mode():
        for start in range(0,len(rows),a.batch):
            x=cache['gts'][start:start+a.batch].cuda().float()
            z=vae.encode(x).latent_dist.mode()
            targets.append(((z-c.vae_shift_factor)*c.vae_scaling_factor).cpu())
    targets=torch.cat(targets);del vae
    # Check membership against every upstream stage, not only the latest CSV.
    csvs=[c.data_csv];ancestor=c.initialize_from
    from src.unet.io import checkpoint_config
    while ancestor:
        saved=torch.load(ancestor,map_location='cpu',weights_only=False);old=checkpoint_config(saved)
        csvs.append(old.data_csv);ancestor=old.initialize_from;del saved
    known=set();slots=set()
    for f in set(csvs):
        for row in read(f):
            known.add(row['character']);slots.add((int(row['calligrapher_id']),row['script'],row['character']))
    records=[]
    with torch.inference_mode():
        for u in (.05,.15,.25,.35,.45,.55,.65,.75,.85,.95):
            for start in range(0,len(rows),a.batch):
                rr=rows[start:start+a.batch];b=len(rr);z0=targets[start:start+b].cuda()
                eps=cache['noise'][start:start+b].cuda();t=torch.full((b,),u,device='cuda')
                writer=torch.tensor([int(r['calligrapher_id']) for r in rr],device='cuda')
                script=torch.tensor(cache['hier_conds'][2][start:start+b],device='cuda')
                g=cache['skels_latent'][start:start+b].cuda()
                def losses(w):
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        return process.training_losses(model,z0,t,model_kwargs=dict(y_callig=w,y_script=script,g=g),noise=eps)['loss'].float().cpu().numpy()
                correct=losses(writer);null=losses(torch.full_like(writer,c.num_writers));wrong=[]
                for w in range(10):wrong.append(losses(torch.full_like(writer,w)))
                wrong=np.stack(wrong,1)
                for j,r in enumerate(rr):
                    candidates=[w for w in legal[r['script']] if w!=int(r['calligrapher_id'])]
                    loss_wrong=float(wrong[j,candidates].mean())
                    records.append(dict(idx=start+j,char=r['character'],script=r['script'],t=u,
                        correct=float(correct[j]),wrong=loss_wrong,null=float(null[j]),
                        delta_wrong=loss_wrong-float(correct[j]),delta_null=float(null[j]-correct[j]),
                        seen_character=r['character'] in known,
                        seen_writer_script_character=(int(r['calligrapher_id']),r['script'],r['character']) in slots))
            print('FM_AUDIT_T',u,flush=True)
    summary=[]
    for u in sorted({r['t'] for r in records}):
        for group in ['all','unseen_character','unseen_writer_script_character','seen_writer_script_character']:
            v=[r for r in records if r['t']==u and (group=='all' or
                (r[group] if group.startswith('seen_') else not r[group.replace('unseen','seen')]))]
            if v:summary.append(dict(t=u,group=group,wrong=paired_ci([r['delta_wrong'] for r in v],[r['char'] for r in v]),
                                     null=paired_ci([r['delta_null'] for r in v],[r['char'] for r in v]),
                                     wrong_relative=float(np.mean([r['delta_wrong']/max(r['correct'],1e-12) for r in v])),
                                     null_relative=float(np.mean([r['delta_null']/max(r['correct'],1e-12) for r in v])),
                                     correct_beats_wrong=float(np.mean([r['delta_wrong']>0 for r in v])),
                                     correct_beats_null=float(np.mean([r['delta_null']>0 for r in v]))))
    save(Path(a.out)/'per_image.json',records);save(Path(a.out)/'summary.json',dict(groups=summary,upstream_csvs=sorted(set(csvs)),
        target='deterministic posterior mode; fixed per-image eps across all writer conditions and t',
        caveat='Positive FM benefit is not proof of visual style or CFG benefit'))


if __name__=='__main__':main()
