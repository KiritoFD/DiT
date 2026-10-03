"""FLUX AE(16ch) 的统一装载与编解码 —— 供 encode_flux_latents / latent_signal_arena 共用。

三个已踩过的坑(全部焊在这里, 别在别处重写):
 ① diffusers 0.27.2 的 AutoencoderKL **不认** use_quant_conv/use_post_quant_conv
    (Flux AE 的 config 是 false) -> from_pretrained 报缺 key; 必须手工建 + load(strict=False)
    + 把 quant_conv/post_quant_conv 灌成**恒等**(真实 Flux AE 就没有这两个 conv)。
 ② 原始的 ae.safetensors 是 sgm/CompVis 老命名(encoder.down.N.block.M), 需要重映射,
    且 decoder.up.N 的顺序与 up_blocks **相反**; attention 子名也不同(q/k/v/proj_out)。
    -> 优先用官方 diffusers 转换件(本文件默认路径), 就没有 ② 的问题。
 ③ 归一化: z = (raw - shift) * scale,  scale=0.3611, shift=0.1159 (读 config, 不写死)。
"""
import json
import os
import re

import torch as th
from diffusers import AutoencoderKL
from safetensors import safe_open

DEFAULT_DIR = "data/pretrained/flux_vae_diffusers"
SD_SF = 0.18215
_CFG_KEYS = ("in_channels", "out_channels", "latent_channels", "down_block_types",
             "up_block_types", "block_out_channels", "layers_per_block",
             "norm_num_groups", "sample_size", "scaling_factor")
_SKIP = ("quant_conv.weight", "quant_conv.bias",
         "post_quant_conv.weight", "post_quant_conv.bias")
_NDOWN = 4


def remap_legacy_key(k):
    """sgm/CompVis 老命名 -> diffusers 命名(只在用原始 ae.safetensors 时才需要)。"""
    k = (k.replace("mid.attn_1.q.", "mid_block.attentions.0.to_q.")
          .replace("mid.attn_1.k.", "mid_block.attentions.0.to_k.")
          .replace("mid.attn_1.v.", "mid_block.attentions.0.to_v.")
          .replace("mid.attn_1.proj_out.", "mid_block.attentions.0.to_out.0.")
          .replace("mid.attn_1.norm.", "mid_block.attentions.0.group_norm.")
          .replace(".mid.block_1.", ".mid_block.resnets.0.")
          .replace(".mid.block_2.", ".mid_block.resnets.1."))
    m = re.match(r"(encoder|decoder)\.down\.(\d+)\.block\.(\d+)\.(.*)", k)
    if m:
        return f"{m.group(1)}.down_blocks.{m.group(2)}.resnets.{m.group(3)}.{m.group(4)}"
    m = re.match(r"(encoder|decoder)\.down\.(\d+)\.downsample\.(.*)", k)
    if m:
        return f"{m.group(1)}.down_blocks.{m.group(2)}.downsamplers.0.{m.group(3)}"
    m = re.match(r"decoder\.up\.(\d+)\.block\.(\d+)\.(.*)", k)
    if m:
        return (f"decoder.up_blocks.{_NDOWN - 1 - int(m.group(1))}"
                f".resnets.{m.group(2)}.{m.group(3)}")
    m = re.match(r"decoder\.up\.(\d+)\.upsample\.(.*)", k)
    if m:
        return f"decoder.up_blocks.{_NDOWN - 1 - int(m.group(1))}.upsamplers.0.{m.group(2)}"
    return k


def _identity_convs(vae):
    with th.no_grad():
        for mod in (vae.quant_conv, vae.post_quant_conv):
            w = mod.weight
            w.zero_()
            for i in range(w.shape[0]):
                w[i, i, 0, 0] = 1.0
            mod.bias.zero_()


def load_flux_ae(dev="cuda", diff_dir=DEFAULT_DIR, legacy_ckpt="",
                 verbose=True):
    """返回 (vae, scale, shift)。scale/shift 从 config.json 读。"""
    scale, shift = 0.3611, 0.1159
    if diff_dir and os.path.isdir(diff_dir):
        cfgp = os.path.join(diff_dir, "config.json")
        c = json.load(open(cfgp, encoding="utf-8")) if os.path.exists(cfgp) else {}
        scale = float(c.get("scaling_factor", scale))
        shift = float(c.get("shift_factor", shift))
        kw = {k: v for k, v in c.items() if k in _CFG_KEYS}
        kw["down_block_types"] = tuple(kw.get("down_block_types", ("DownEncoderBlock2D",) * 4))
        kw["up_block_types"] = tuple(kw.get("up_block_types", ("UpDecoderBlock2D",) * 4))
        kw["block_out_channels"] = tuple(kw.get("block_out_channels", (128, 256, 512, 512)))
        vae = AutoencoderKL(**kw)
        sdf = {}
        with safe_open(os.path.join(diff_dir, "diffusion_pytorch_model.safetensors"),
                       framework="pt") as f:
            for k in f.keys():
                sdf[k] = f.get_tensor(k)
        miss, unexp = vae.load_state_dict(sdf, strict=False)
        _identity_convs(vae)
        if verbose:
            print(f"[flux-ae] diffusers 目录 {diff_dir}: latent_ch={c.get('latent_channels')} "
                  f"scale={scale} shift={shift} "
                  f"missing(除两 conv)={[k for k in miss if k not in _SKIP][:3]} "
                  f"unexpected={unexp[:3]}")
    elif legacy_ckpt and os.path.exists(legacy_ckpt):
        kw = dict(in_channels=3, out_channels=3, latent_channels=16,
                  down_block_types=("DownEncoderBlock2D",) * 4,
                  up_block_types=("UpDecoderBlock2D",) * 4,
                  block_out_channels=(128, 256, 512, 512), layers_per_block=2,
                  norm_num_groups=32, sample_size=1024)
        vae = AutoencoderKL(**kw)
        sdf = {}
        with safe_open(legacy_ckpt, framework="pt") as f:
            for k in f.keys():
                sdf[remap_legacy_key(k)] = f.get_tensor(k)
        miss, unexp = vae.load_state_dict(sdf, strict=False)
        _identity_convs(vae)
        if verbose:
            print(f"[flux-ae] 老命名件 {legacy_ckpt}: 重映射后 "
                  f"missing(除两 conv)={[k for k in miss if k not in _SKIP][:3]} "
                  f"unexpected={unexp[:3]}")
    else:
        raise FileNotFoundError(f"找不到 FLUX VAE: {diff_dir} / {legacy_ckpt}")
    return vae.to(dev).eval(), scale, shift


@th.no_grad()
def encode_flux(vae, x, scale=0.3611, shift=0.1159):
    """x: (B,3,H,W) in [0,1] -> (B,16,H/8,W/8) 归一化 latent。"""
    with th.autocast("cuda", dtype=th.bfloat16):
        z = vae.encode((x * 2 - 1).to(th.bfloat16)).latent_dist.mode()
    return (z.float() - shift) * scale


@th.no_grad()
def decode_flux(vae, z, scale=0.3611, shift=0.1159):
    with th.autocast("cuda", dtype=th.bfloat16):
        d = vae.decode((z / scale + shift).to(th.bfloat16)).sample
    return ((d.clamp(-1, 1) + 1) / 2).float()
