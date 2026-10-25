"""FM + optional REPA objective. Legacy experiments are not dispatched here.

pair_fraction is deliberately allowed here: it only shapes batch composition
(same-char cross-writer pairs via PairMixtureSampler); the objective stays
plain FM with no pair-ranking auxiliary.
"""
import torch


class LatentStdObjective:
    def __init__(self, state):
        args = state.args
        # Fail at selection, not halfway through training or via silent fallback.
        unsupported = ('w_std_mid', 'w_latent_skel', 'w_style_rank', 'w_critic_writer',
                       'w_critic_away', 'w_deform_skel', 'style_anchor_weight',
                       'w_style_ortho', 'rank_gradient_ratio',
                       'crosschar_gradient_ratio', 'glyph_deform_prob',
                       'glyph_noise_scale', 'glyph_patch_drop', 'glyph_mask_prob',
                       'w_tv', 'w_fold', 'w_tv_out', 'w_tv_res', 'w_tv_stroke',
                       'w_prune_l1', 'w_lig_l1', 'deform_skel')
        enabled = [key for key in unsupported if getattr(args, key, 0)]
        if (enabled or not state._std_soft_cond or state._train_vae_enc or state._aux_dirs
                or state._strong_style is not None or state._writer_table_tuning is not None):
            raise ValueError(f'latent-std path supports cached 4ch FM+REPA only: {enabled}')
        if 'flow' not in type(state.diffusion).__name__.lower():
            raise ValueError('latent-std requires Flow Matching')
        if float(args.w_repa) > 0 and state.repa_loss_fn is None:
            raise ValueError('REPA enabled but module missing')
        self.diffusion = state.diffusion
        self.repa = state.repa_loss_fn if args.w_repa > 0 else None
        self.channel_weights = state._ch_w

    def __call__(self, model, z, t, kwargs, batch, image, step):
        if 'latent' not in batch or z.shape[1] != 4 or 'g' not in kwargs:
            raise ValueError('latent-std requires cached 4ch latent and STD condition')
        if self.repa is not None:
            layers = self.repa.layers
            if len(layers) > 1:
                kwargs['return_intermediate_layers'] = layers
            else:
                kwargs['return_intermediate_layer'] = layers[0]
        with torch.autocast(z.device.type, dtype=torch.bfloat16, enabled=z.is_cuda):
            result = self.diffusion.training_losses(
                model, z, t, kwargs, channel_weights=self.channel_weights)
            fm = result['loss'].mean()
        repa = fm.new_zeros(())
        if self.repa is not None:
            feats = result.get('intermediate_feats')
            if feats is None:
                raise RuntimeError('REPA enabled but student features missing')
            if isinstance(feats, dict) and any(k not in feats for k in self.repa.layers):
                raise RuntimeError('REPA enabled but some requested layers are missing')
            if image is None and (self.repa.feature_cache is None or 'img_id' not in batch):
                raise RuntimeError('REPA has neither image nor indexed cache')
            repa = self.repa(feats, image, step=step, img_ids=batch.get('img_id'))
            if step >= self.repa.warmup_steps and not repa.requires_grad:
                raise RuntimeError('REPA enabled but result has no gradient graph')
        return result, fm, repa
