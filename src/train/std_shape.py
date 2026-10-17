"""Pinned-RAM STD bank sampling and low-frequency encoder health telemetry."""
import torch
from contextlib import nullcontext
from torch.nn import functional as F
from src.utils.std_shape_aug import mixture


class StdShapeTraining:
    def __init__(self,args,logger):
        data=torch.load(args.std_shape_cache,map_location='cpu',weights_only=True,mmap=True)
        if data['version']!=2:raise ValueError('Offline STD bank required')
        self.bank=data['variants'].pin_memory()
        self.mode=args.std_shape_aug
        self.profile = getattr(args, 'profile_steps', 0) > 0
        self.logger=logger
        logger.info('[std-shape] offline bank preloaded in pinned RAM: %.2f GiB, %d variants/tier',
                    self.bank.numel()*self.bank.element_size()/2**30,self.bank.shape[2])
        self.logger=logger;self.interval=args.std_shape_log_every
        if self.interval<=0:raise ValueError('STD health logging interval must be positive')
        for key in ('glyph_deform_prob','glyph_noise_prob','glyph_patch_drop','glyph_mask_prob'):
            if getattr(args,key,0):raise ValueError('STD shape must not overlap legacy corruptions: '+key)
        if args.pair_fraction or args.rank_gradient_ratio or args.strong_style_mode!='off' or args.w_critic_writer or args.w_critic_away:
            raise ValueError('First20L run is FM+REPA only, not P1/Critic training')
        for key in ('crosschar_gradient_ratio','w_style_rank','w_latent_skel','w_std_mid','w_deform_skel','deform_skel'):
            if getattr(args,key,0):raise ValueError('First20L run must not mix additional supervision/structure: '+key)

    def __call__(self,model,batch,kwargs,x0,step):
        clean=kwargs['g']  # Already copied by build_train_model_kwargs.
        indices=batch['std_shape_index'].cpu()
        tiers=torch.multinomial(torch.tensor(mixture(step)),len(indices),replacement=True)
        effective=torch.where(tiers>0,torch.ones_like(tiers),tiers) if self.mode=='mild' else tiers
        variants=torch.randint(self.bank.shape[2],(len(indices),))
        region = torch.profiler.record_function if self.profile else lambda name: nullcontext()
        with region('std_bank.select'):
            selected=self.bank[indices,(effective-1).clamp_min(0),variants]
        with region('std_bank.pin'):
            selected=selected.pin_memory()
        with region('std_bank.h2d'):
            sampled=selected.to(x0.device,non_blocking=True).float()
        g=torch.where((effective==0).to(x0.device)[:,None,None,None],clean,sampled)
        kwargs['g']=g
        if step%self.interval==0:
            original=getattr(model,'module',model);original=getattr(original,'_orig_mod',original)
            health=encoder_health(original,g,x0)
            values={f'tier_{i}':float((tiers==i).float().mean()) for i in range(4)}
            self.logger.info('[std-shape] step=%d aug=%s health=%s',step,values,health)


@torch.no_grad()
def encoder_health(model,g,x0):
    with torch.autocast(g.device.type,dtype=torch.bfloat16,enabled=g.is_cuda):
        f0=model.glyph_embedder(torch.zeros_like(g[:1])).float()
        fg=model.glyph_embedder(g).float();delta=fg-f0
        mask=F.adaptive_avg_pool2d(g,delta.shape[-2:])>.1
        energy=delta.square().mean(1,keepdim=True)
        inside=(energy*mask).sum()/mask.sum().clamp_min(1)
        outside=(energy*~mask).sum()/(~mask).sum().clamp_min(1)
        scale=model.effective_glyph_scale().float()
        stem=model.x_embedder(x0).float()
        ratio=(scale*delta).square().mean().sqrt()/stem.square().mean().sqrt().clamp_min(1e-8)
    return {'f0_norm':float(f0.norm()),'fg_rms':float(fg.square().mean().sqrt()),
            'delta_rms':float(delta.square().mean().sqrt()),'content_background_ratio':float(inside/outside.clamp_min(1e-8)),
            'glyph_scale':float(scale),'injection_over_GT_stem_rms':float(ratio)}
