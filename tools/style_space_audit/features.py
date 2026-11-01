"""Frozen CE encoder; matched native generators, GT-only VAE roundtrip, nuisance controls."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from tools.three_day.contracts import read,save,sha


class StyleDisc(nn.Module):
    def __init__(self):
        super().__init__()
        from torchvision.models import resnet18
        ck=torch.load('assets/style_disc_ce_holdout.pt',map_location='cpu',weights_only=False)
        self.body=nn.Sequential(*list(resnet18(weights=None).children())[:-1])
        self.head=nn.Sequential(nn.Flatten(),nn.Linear(512,ck['d']));self.cls=nn.Linear(ck['d'],ck['n_cls'])
        self.load_state_dict(ck['model'],strict=True);self.slots=ck['all_keys'];self.eval().requires_grad_(False)

    def forward(self,rgb):
        x=F.interpolate(rgb,(160,160),mode='bicubic',align_corners=False,antialias=True).clamp(0,1)
        x=(x-x.new_tensor([.485,.456,.406])[None,:,None,None])/x.new_tensor([.229,.224,.225])[None,:,None,None]
        raw=self.head(self.body(x));return F.normalize(raw,dim=-1),self.cls(raw)


def pixels(paths,size=256):
    return torch.from_numpy(np.stack([np.asarray(Image.open(p).convert('RGB').resize((size,size),Image.Resampling.BICUBIC)).copy() for p in paths])).permute(0,3,1,2).float()/255


def variants(x,mode,targets=None):
    if mode=='raw':return x
    gray=(x*x.new_tensor([.299,.587,.114])[None,:,None,None]).sum(1,keepdim=True)
    # White-paper control removes color and near-white low-contrast scanning background.
    ink=((1-gray)-.08).clamp_min(0)/.92
    if mode=='white':return (1-ink).expand(-1,3,-1,-1)
    if mode!='ink_match':raise ValueError(mode)
    if targets is None:raise ValueError('Missing paired GT ink target')
    from scipy.ndimage import minimum_filter,maximum_filter
    matched=[]
    for value,target in zip(ink[:,0].cpu().numpy(),targets.cpu().numpy()):
        operation=minimum_filter if value.mean()>target else maximum_filter;previous=value
        for radius in range(1,65):
            current=operation(value,size=2*radius+1,mode='constant',cval=0)
            a,b=float(previous.mean()),float(current.mean())
            if min(a,b)<=target<=max(a,b):
                fraction=(target-a)/(b-a) if b!=a else 0.;matched.append(previous+fraction*(current-previous));break
            previous=current
        else:raise ValueError('Ink mass not reachable within diagnostic radius')
    ink=torch.from_numpy(np.stack(matched)[:,None]).to(x.device)
    if not torch.allclose(ink.mean((1,2,3)),targets,atol=1e-5):raise ValueError('Ink matching residual')
    return (1-ink).expand(-1,3,-1,-1)


def deep(root,native_std=False):
    from tools.u100_pretrain.eval_offline import load_legacy_ema_verified
    from src.unet.io import diffusion
    from src.eval.sampler import sample_latents
    from src.eval.decoder import load_eval_vae
    root=Path(root);rows=json.loads((root/'rows.json').read_text());inputs=torch.load(root/'common_inputs.pt',weights_only=True)
    ckpt='assets/results/unet_deep_sym4_114m_60m/20261019-011647-Unet-flow/checkpoints/0085000.pt'
    model,c,ck,record=load_legacy_ema_verified(ckpt);model.eval().requires_grad_(False)
    mapping={int(k):int(v) for k,v in json.loads(Path(c.callig_remap_json).read_text()).items()}
    fonts={int(k):int(v) for k,v in json.loads(Path(c.font_remap_json).read_text()).items()}
    writers=[mapping[int(r['raw_writer_id'])] for r in rows];scripts=[fonts[int(r['script_id'])] for r in rows]
    conds=[(w,0) for w in writers];hier=(writers,writers,scripts)
    g=inputs['g']
    if native_std:
        from src.utils.std_soft_glyph import StdSoftGlyphStore
        store=StdSoftGlyphStore(c.std_soft_root,is_train=False,eval_sigma=c.std_soft_eval_sigma)
        g=torch.stack([store.get(int(r['script_id']),r['char'],sigma=c.std_soft_eval_sigma) for r in rows])
    with torch.inference_mode():
        z=sample_latents(model,diffusion(c),inputs['noise'],conds,1.,8,'cuda',skel=g,seed=0,hier_conds=hier)
    name='Deep114_native' if native_std else 'Deep114'
    np.save(root/(name+'_latents.npy'),z.numpy());del model,ck
    vae=load_eval_vae('cuda',c.vae_path).eval().requires_grad_(False)
    out=root/name;out.mkdir(exist_ok=True);recon=root/'VAE_GT';recon.mkdir(exist_ok=True)
    with torch.inference_mode():
        for start in range(0,len(rows),4):
            decoded=(vae.decode(z[start:start+4].cuda()/c.vae_scaling_factor+c.vae_shift_factor).sample.float().clamp(-1,1)+1)/2
            gt=pixels([root/'GT'/f'g{i}.png' for i in range(start,min(start+4,len(rows)))]).cuda()
            zr=(vae.encode(2*gt-1).latent_dist.mean-c.vae_shift_factor)*c.vae_scaling_factor
            rr=(vae.decode(zr/c.vae_scaling_factor+c.vae_shift_factor).sample.float().clamp(-1,1)+1)/2
            for j,(x,y) in enumerate(zip(decoded,rr)):
                for value,path in [(x,out),(y,recon)]:Image.fromarray((value.permute(1,2,0).cpu().numpy()*255).round().astype('uint8')).save(path/f'g{start+j}.png')
    trained={str(Path(r['image_path'])) for r in read(c.data_csv)}
    save(root/(name+'_identity.json'),dict(checkpoint=ckpt,sha256=sha(ckpt),ema=record,writer_rows=mapping,
        cfg=1,steps=50,seed='exact exported P2 per-image noise',
        STD=dict(mode='native' if native_std else 'shared_P2',root=c.std_soft_root if native_std else 'exported P2 bbox64',sigma=c.std_soft_eval_sigma if native_std else 0.)))
    save(root/(name+'_training_overlap.json'),dict(exact_image_path_overlap=sum(str(Path(r['image_path'])) in trained for r in rows),n=len(rows),
        std_mode='checkpoint_native' if native_std else 'shared_P2_STD',source_csv=c.data_csv,
        caveat='Path-based overlap, not a guarantee of pixel/near-duplicate disjointness'))


def extract(root):
    root=Path(root);real=json.loads((root/'real_rows.json').read_text());rows=json.loads((root/'rows.json').read_text())
    model=StyleDisc().cuda();features=root/'features';features.mkdir(exist_ok=True)
    qroot=Path('/home/ds/Workspace/xy/qwen-lora-ref-20261025')
    qrows=json.loads((qroot/'data.json').read_text())['datasets']['validation']
    lookup={(r['calligrapher'],r['script'],r['character'],r['pixel_sha256']):i for i,r in enumerate(qrows)}
    qpaths=[]
    for r in rows:
        key=(r['writer'],r['script'],r['char'],r['pixel_sha256'])
        if key not in lookup:raise ValueError('No exact paired Qwen target: '+str(key))
        qpaths.append(str(next((qroot/'evaluations/0004000').glob(f'{lookup[key]:03d}_*.png'))))
    save(root/'qwen_pairing.json',dict(n=len(qpaths),key='writer/script/character/original-pixel-sha256',paths=qpaths))
    std_dir=root/'STD';std_dir.mkdir(exist_ok=True)
    g=torch.load(root/'common_inputs.pt',map_location='cpu',weights_only=True)['g']
    standard=F.interpolate(1-g.float(),(256,256),mode='bicubic',align_corners=False).clamp(0,1)
    for i,x in enumerate(standard):Image.fromarray((x[0].numpy()*255).round().astype('uint8')).convert('RGB').save(std_dir/f'g{i}.png')
    datasets={'REAL':[r['path'] for r in real],**{name:[str(root/name/f'g{i}.png') for i in range(len(rows))]
        for name in ('STD','GT','VAE_GT','Deep114','Deep114_native','P2_CFG1','P2_CFG1.6')},'Qwen':qpaths,
        'P2_swap':[str(p) for p in sorted((root/'P2_swap').glob('g*.png'),key=lambda p:int(p.stem[1:]))]}
    for name,paths in datasets.items():
        targets=[]
        if name not in ('REAL','P2_swap'):
            for start in range(0,len(rows),32):targets.extend((1-variants(pixels(datasets['GT'][start:start+32],160),'white')).mean((1,2,3)).tolist())
        for mode in (('raw','white') if name in ('REAL','P2_swap') else ('raw','white','ink_match')):
            existing=features/(name+'_'+mode+'.npz')
            if existing.exists():
                if len(np.load(existing)['embedding'])!=len(paths):raise ValueError('Cached feature sample count mismatch')
                continue
            ee=[];ll=[]
            with torch.inference_mode():
                for start in range(0,len(paths),32):
                    x=pixels(paths[start:start+32],160).cuda()
                    t=torch.tensor(targets[start:start+32],device='cuda') if targets else None
                    e,l=model(variants(x,mode,t));ee.append(e.cpu().numpy());ll.append(l.cpu().numpy())
            np.savez(features/(name+'_'+mode+'.npz'),embedding=np.concatenate(ee),logits=np.concatenate(ll))
            print('STYLE_FEATURES',name,mode,len(paths),flush=True)


def heldout_roundtrip(root):
    from src.eval.decoder import load_eval_vae
    from .analysis import clustering
    root=Path(root);all_rows=json.loads((root/'real_rows.json').read_text())
    rows=[r for r in all_rows if r['dedup_source_heldout'] and r['character_heldout']]
    if not rows:raise ValueError('No source-and-character-heldout GT for VAE audit')
    vae=load_eval_vae('cuda','pretrained_models/sd-vae-ft-ema').eval().requires_grad_(False);teacher=StyleDisc().cuda()
    es=[];ers=[];correct=[];rcorrect=[];gallery=[];keys=[tuple(k) for k in teacher.slots]
    with torch.inference_mode():
        for start in range(0,len(rows),4):
            rr=rows[start:start+4];x=pixels([r['path'] for r in rr]).cuda()
            z=vae.encode(2*x-1).latent_dist.mean;y=(vae.decode(z).sample.float().clamp(-1,1)+1)/2
            # Match the PNG/PIL160 evaluation pipeline for BOTH domains; differentiable
            # Torch resizing is reserved for the separate gradient pilot.
            tx=pixels([r['path'] for r in rr],160).cuda()
            array=(y.permute(0,2,3,1).cpu().numpy()*255).round().astype('uint8')
            resized=np.stack([np.asarray(Image.fromarray(im).resize((160,160),Image.Resampling.BICUBIC)).copy() for im in array])
            ty=torch.from_numpy(resized).permute(0,3,1,2).cuda().float()/255
            e,l=teacher(tx);re,rl=teacher(ty);truth=torch.tensor([keys.index((r['writer'],r['script'])) for r in rr],device='cuda')
            correct.extend((l.argmax(1)==truth).cpu().tolist());rcorrect.extend((rl.argmax(1)==truth).cpu().tolist())
            es.append(e.cpu().numpy());ers.append(re.cpu().numpy());gallery.extend(zip(x.cpu(),y.cpu()))
    a=np.concatenate(es);b=np.concatenate(ers)
    save(root/'heldout_vae_roundtrip.json',dict(n=len(rows),source_and_character_heldout=True,
        source_caveat='Saved directory-level holdout proxy, exact-pixel duplicates removed; not verified independent artwork IDs.',
        preprocessing='Both GT and reconstruction: original256 -> uint8 PNG-equivalent -> PIL bicubic160 -> ImageNet normalization',
        gt_slot_top1=float(np.mean(correct)),reconstruction_slot_top1=float(np.mean(rcorrect)),
        feature_l2_mean=float(np.linalg.norm(a-b,axis=1).mean()),feature_cosine_mean=float((a*b).sum(1).mean()),
        GT_ARI=clustering(a,rows),VAE_ARI=clustering(b,rows),
        vae='pretrained_models/sd-vae-ft-ema',posterior='deterministic mean; native unscaled encode/decode, equivalent sf=.18215 shift0 inverse'))
    selected=[];seen=set()
    for i,r in enumerate(rows):
        key=r['writer'],r['script']
        if key not in seen:selected.append(i);seen.add(key)
    from PIL import ImageDraw
    canvas=Image.new('RGB',(256,24+len(selected)*128),'white');d=ImageDraw.Draw(canvas);d.text((2,4),'Heldout GT',fill='black');d.text((130,4),'VAE reconstruction',fill='black')
    for j,i in enumerate(selected):
        for k,x in enumerate(gallery[i]):canvas.paste(Image.fromarray((x.permute(1,2,0).numpy()*255).round().astype('uint8')).resize((128,128)),(k*128,24+j*128))
    canvas.save(root/'heldout_vae_roundtrip.png');print('HELDOUT_VAE_COMPLETE',len(rows),flush=True)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--deep',action='store_true');p.add_argument('--native-std',action='store_true');p.add_argument('--heldout-roundtrip',action='store_true');a=p.parse_args();torch.set_num_threads(8)
    if a.deep:deep(a.root,a.native_std)
    elif a.heldout_roundtrip:heldout_roundtrip(a.root)
    else:extract(a.root)
