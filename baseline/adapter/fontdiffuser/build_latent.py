# -*- coding: utf-8 -*-
"""Build FontDiffuser components in LATENT space (4x32x32, project VAE).

Adaptations (method-faithful otherwise):
  - UNet: in/out channels 4, sample_size 32 (latent); MCA/StyleRSI structure,
    channels, schedulers unchanged. StyleRSI style-content pairing uses the
    size-matching fix (adapter patch_fontdiffuser_256.py) which is
    resolution-agnostic.
  - Style/Content encoders: consume 4ch latents directly (input_nc=4) at
    resolution 32 via a 32-entry in their resolution->arch tables (mirror of
    the 96 ladder: 5 DBlocks ending at ch*16=1024). The CG-GAN encoders are
    DBlock chains, so this is just a deeper ladder on the same design.
"""
import sys

import torch

sys.path.insert(0, "/root/Workspace/xy/DiT/baseline/FontDiffuser")

from src.modules import style_encoder as _se  # noqa: E402
from src.modules import content_encoder as _ce  # noqa: E402
from src.modules.content_encoder import ContentEncoder as _ContentEncoder  # noqa: E402
from src.modules.style_encoder import StyleEncoder as _StyleEncoder  # noqa: E402
from src import UNet  # noqa: E402


def _ladder32(ch, nc):
    """4-block DBlock ladder 32->2: out [ch,2ch,4ch,8ch].

    Channel positions must line up with the DOWN-block content-feature
    indexing (`encoder_hidden_states[1][index]`, index=1..3 -> ch*2^(index-1)),
    which is what makes MCA's ChannelAttn GroupNorms match. Final = 8*ch, so
    the UNet cross_attention_dim must be style_start_channel*8 at 32px.
    """
    return {
        "in_channels": [nc] + [ch * i for i in (1, 2, 4)],
        "out_channels": [ch * i for i in (1, 2, 4, 8)],
        "resolution": [16, 8, 4, 2],
    }


def install_latent_arch_tables(input_nc=4):
    """Add resolution-32 entries to both encoders' arch tables (idempotent)."""
    if not hasattr(_se, "_latent32_installed"):
        _orig_se = _se.style_encoder_textedit_addskip_arch

        def se_arch(ch=64, out_channel_multiplier=1, input_nc=3):
            d = _orig_se(ch, out_channel_multiplier, input_nc)
            d[32] = _ladder32(ch, input_nc)
            return d

        _se.style_encoder_textedit_addskip_arch = se_arch

        _orig_ce = _ce.content_encoder_arch

        def ce_arch(ch=64, out_channel_multiplier=1, input_nc=3):
            d = _orig_ce(ch, out_channel_multiplier, input_nc)
            d[32] = _ladder32(ch, input_nc)
            return d

        _ce.content_encoder_arch = ce_arch
        _se._latent32_installed = True


class _SaveFeatures32Mixin:
    """Both encoder classes branch save_featrues on resolution 80/96/128/256;
    resolution 32 falls through, so set it explicitly (same depth as 96)."""

    def _fix_save_features(self):
        if not hasattr(self, "save_featrues"):
            self.save_featrues = [0, 1, 2, 3, 4]


class StyleEncoderLatent(_StyleEncoder, _SaveFeatures32Mixin):
    def __init__(self, G_ch=64, **kw):
        install_latent_arch_tables()
        kw.update(resolution=32, input_nc=4)
        super().__init__(G_ch=G_ch, **kw)
        self._fix_save_features()


class ContentEncoderLatent(_ContentEncoder, _SaveFeatures32Mixin):
    def __init__(self, G_ch=64, **kw):
        install_latent_arch_tables()
        kw.update(resolution=32, input_nc=4)
        super().__init__(G_ch=G_ch, **kw)
        self._fix_save_features()


def build_unet_latent(args):
    unet = UNet(
        sample_size=args.resolution,                 # 32 (latent grid)
        in_channels=4,
        out_channels=4,
        flip_sin_to_cos=True,
        freq_shift=0,
        down_block_types=("DownBlock2D",
                          "MCADownBlock2D",
                          "MCADownBlock2D",
                          "DownBlock2D"),
        up_block_types=("UpBlock2D",
                        "StyleRSIUpBlock2D",
                        "StyleRSIUpBlock2D",
                        "UpBlock2D"),
        block_out_channels=args.unet_channels,
        layers_per_block=2,
        downsample_padding=1,
        mid_block_scale_factor=1,
        act_fn="silu",
        norm_num_groups=32,
        norm_eps=1e-05,
        cross_attention_dim=args.style_start_channel * 8,   # 32px ladder ends at 8*ch
        attention_head_dim=1,
        channel_attn=args.channel_attn,
        content_encoder_downsample_size=args.content_encoder_downsample_size,
        content_start_channel=args.content_start_channel,
        reduction=32,
    )
    return unet


def build_style_encoder_latent(args):
    return StyleEncoderLatent(G_ch=args.style_start_channel)


def build_content_encoder_latent(args):
    return ContentEncoderLatent(G_ch=args.content_start_channel)


class FontDiffuserModelLatent(torch.nn.Module):
    """Same forward contract as FontDiffuserModel, on latents."""

    def __init__(self, unet, style_encoder, content_encoder):
        super().__init__()
        self.unet = unet
        self.style_encoder = style_encoder
        self.content_encoder = content_encoder

    def forward(self, x_t, timesteps, style_images, content_images,
                content_encoder_downsample_size):
        style_img_feature, _, _ = self.style_encoder(style_images)
        b, c, h, w = style_img_feature.shape
        style_hidden_states = style_img_feature.permute(0, 2, 3, 1).reshape(b, h * w, c)

        content_img_feature, content_residual = self.content_encoder(content_images)
        content_residual.append(content_img_feature)
        style_content_feature, style_content_res = self.content_encoder(style_images)
        style_content_res.append(style_content_feature)

        out = self.unet(
            x_t, timesteps,
            encoder_hidden_states=[style_img_feature, content_residual,
                                   style_hidden_states, style_content_res],
            content_encoder_downsample_size=content_encoder_downsample_size)
        return out[0], out[1]
