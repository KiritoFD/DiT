"""Full-batch training-only teacher, preserving the v84A inference signature."""
import hashlib
from pathlib import Path
import torch
import numpy as np
from src.model.style_encoder import build_style_encoder
from src.loss.crosschar_style import CrossCharacterFeatureSampler,cross_character_nce
from src.train.gradient_budget import GradientBudget,format_metrics
from src.loss.flow_matching import TIME_SCALE


class CrossCharacterPostTraining:
    def __init__(self,args,device,logger):
        if args.diffusion_type!='flow' or not args.crosschar_bank or not args.crosschar_encoder_ckpt:
            raise ValueError('Cross-character training requires flow, feature bank and teacher checkpoint')
        if getattr(args,'w_critic_writer',0) or getattr(args,'w_critic_away',0):
            raise ValueError('Cross-character pilot must not combine legacy Critic/AwaySTD losses')
        bank=torch.load(args.crosschar_bank,map_location='cpu',weights_only=False)
        digest=hashlib.sha256(Path(args.crosschar_encoder_ckpt).read_bytes()).hexdigest()
        if digest!=bank['encoder_sha256'] or bank['partition']!='encoder_train_only' or not bank.get('teacher_accepted',False):
            raise ValueError('Teacher identity or reference split provenance mismatch')
        parent=Path(args.crosschar_encoder_ckpt).parent
        if hashlib.sha256((parent/'matched_teacher.json').read_bytes()).hexdigest()!=bank['matched_teacher_sha256']:
            raise ValueError('Matched teacher acceptance report changed')
        split_file=parent/'split_indices.npz'
        if hashlib.sha256(split_file.read_bytes()).hexdigest()!=bank['split_sha256']:
            raise ValueError('Teacher reference split changed')
        split=np.load(split_file)
        references=set(map(int,bank['reference_indices']))
        if not references.issubset(set(map(int,split['train']))) or any(references & set(map(int,split[k])) for k in ('val_char','test_char','test_source')):
            raise ValueError('Held-out GT found in the supervision feature bank')
        if len(bank['features'])!=len(bank['reference_indices']):
            raise ValueError('Feature rows and provenance indices disagree')
        ck=torch.load(args.crosschar_encoder_ckpt,map_location='cpu',weights_only=False)
        self.teacher=build_style_encoder(len(ck['slots']),len(ck['scripts']),ck.get('architecture','legacy'))
        self.teacher.load_state_dict(ck['model'],strict=True)
        self.teacher.to(device).eval().requires_grad_(False)
        self.features=bank['features'].to(device).detach()
        self.script_lookup=bank['encoder_script_ids'].to(device)
        self.target_metadata=bank['target_metadata']
        self.sampler=CrossCharacterFeatureSampler(bank['writers'],bank['scripts'],bank['chars'],bank['sources'])
        self.args,self.logger=args,logger
        self.scale={};self.weak={};self.g_scale=None
        logger.info('[crosschar] full-batch; ratio=%.3f; train-only reference count=%d',args.crosschar_gradient_ratio,len(self.features))

    def __call__(self,model,x0,t,kwargs,batch,fm,step,fm_norm=None):
        device=x0.device
        writer=batch['y_callig_raw'].to(device).long()
        script=batch['y_script'].to(device).long()
        ids=batch['img_id'].reshape(-1).tolist()
        metadata=[self.target_metadata[int(i)] for i in ids]
        w_cpu,s_cpu=writer.tolist(),script.tolist()
        if any((w,s)!=(m[0],m[1]) for w,s,m in zip(w_cpu,s_cpu,metadata)):
            raise ValueError('Batch image IDs disagree with explicit writer/script metadata')
        pi,ni,sampling=self.sampler.sample(w_cpu,s_cpu,[m[2] for m in metadata],[m[3] for m in metadata])
        pi=torch.as_tensor(pi,device=device);ni=torch.as_tensor(ni,device=device)
        g=batch['std_soft'].to(device).float()
        if g.ndim==3:
            g=g[:,None]
        if self.g_scale is None:
            self.g_scale=1. if float(g.amax())<=1.5 else 1/255.
        g=g*self.g_scale
        ka={k:v for k,v in kwargs.items() if k not in ('return_intermediate_layer','return_intermediate_layers','y_pair','y_callig_raw')}
        ka.update(y_callig=writer,y_script=script,g=g)
        # Ordinary same-writer denoising, NOT a writer-swapped counterfactual.
        tt=.05+.25*torch.rand(len(x0),device=device)
        xt=(1-tt[:,None,None,None])*x0+tt[:,None,None,None]*torch.randn_like(x0)
        modes=[(m,m.training) for m in model.modules()]
        model.eval()
        try:
            with torch.autocast(device.type,dtype=torch.bfloat16,enabled=x0.is_cuda):
                velocity=model(xt,tt*TIME_SCALE,**ka)
                if isinstance(velocity,tuple):
                    velocity=velocity[0]
                pred=xt-tt[:,None,None,None]*velocity[:,:x0.shape[1]].float()
                # Frozen teacher parameters, but an intact gradient to pred and DiT.
                query=self.teacher(pred,g,self.script_lookup[script])['embedding']
                contrast=cross_character_nce(query,self.features,pi,ni)
        finally:
            for module,mode in modes:
                module.training=mode
        budget=GradientBudget(self.args.strong_style_max_weight,self.scale,self.weak,self.logger)
        value,metrics=budget.balance({'crosschar':(contrast,self.args.crosschar_gradient_ratio)},
            [p for p in model.parameters() if p.requires_grad],fm_norm,step)
        if fm_norm is not None or step%50==0:
            self.logger.info('[crosschar-step] step=%d n=%d hard=%.3f cross_source=%.3f %s',step,len(x0),
                sampling['hard_negative_fraction'],sampling['positive_cross_source_fraction'],format_metrics(metrics))
        return value
