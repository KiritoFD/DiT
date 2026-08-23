# -*- coding: utf-8 -*-
"""
convert_grayscale_vae.py — 把 3ch VAE 改造成 1ch 黑白 VAE (外科手术).

第一刀: encoder.conv_in  [C_out, 3, K, K] -> sum(dim=1, keepdim) -> [C_out, 1, K, K]
         完美保留对全色彩融合边缘的检测能力
第二刀: decoder.conv_out [3, C_in, K, K] -> mean(dim=0, keepdim) -> [1, C_in, K, K]
         取 3 通道均值

做完后 VAE 变成纯正单通道黑白压缩器, 且吃满预训练特征.

用法:
  python tools/vae/convert_grayscale_vae.py --src pretrained_models/sd-vae-ft-ema --dst pretrained_models/sd-vae-ft-ema-gray
  python tools/vae/convert_grayscale_vae.py --src pretrained_models/kl-f4 --dst pretrained_models/kl-f4-gray
"""
import os
import sys
import json
import argparse
import torch
import torch.nn as nn


def convert_grayscale(src_path, dst_path):
    from diffusers import AutoencoderKL

    vae = AutoencoderKL.from_pretrained(src_path)
    config = dict(vae.config)
    print(f"=== 原始 VAE ===")
    print(f"  in_channels:  {config['in_channels']}")
    print(f"  out_channels: {config['out_channels']}")
    print(f"  latent_channels: {config['latent_channels']}")
    print(f"  block_out_channels: {config['block_out_channels']}")

    # === 第一刀: encoder.conv_in ===
    old_conv_in = vae.encoder.conv_in
    old_w = old_conv_in.weight.data  # [C_out, 3, K, K]
    old_b = old_conv_in.bias.data   # [C_out]
    C_out, C_in_old, K_h, K_w = old_w.shape
    print(f"\n=== 第一刀: encoder.conv_in ===")
    print(f"  原始: {old_w.shape}  ({C_in_old}ch input)")

    # sum(dim=1, keepdim=True) -> [C_out, 1, K, K]
    new_w = old_w.sum(dim=1, keepdim=True)  # [C_out, 1, K, K]
    print(f"  新: {new_w.shape}  (sum of {C_in_old} channels)")

    new_conv_in = nn.Conv2d(1, C_out, old_conv_in.kernel_size,
                            old_conv_in.stride, old_conv_in.padding,
                            old_conv_in.dilation, old_conv_in.groups,
                            old_conv_in.bias is not None,
                            old_conv_in.padding_mode)
    new_conv_in.weight.data = new_w
    new_conv_in.bias.data = old_b
    vae.encoder.conv_in = new_conv_in

    # === 第二刀: decoder.conv_out ===
    old_conv_out = vae.decoder.conv_out
    old_w2 = old_conv_out.weight.data  # [3, C_in, K, K]
    old_b2 = old_conv_out.bias.data    # [3]
    C_out_old2, C_in2, K_h2, K_w2 = old_w2.shape
    print(f"\n=== 第二刀: decoder.conv_out ===")
    print(f"  原始: {old_w2.shape}  ({C_out_old2}ch output)")

    # mean(dim=0, keepdim=True) -> [1, C_in, K, K]
    new_w2 = old_w2.mean(dim=0, keepdim=True)  # [1, C_in, K, K]
    new_b2 = old_b2.mean(dim=0, keepdim=True)  # [1]
    print(f"  新: {new_w2.shape}  (mean of {C_out_old2} channels)")

    new_conv_out = nn.Conv2d(C_in2, 1, old_conv_out.kernel_size,
                             old_conv_out.stride, old_conv_out.padding,
                             old_conv_out.dilation, old_conv_out.groups,
                             old_conv_out.bias is not None,
                             old_conv_out.padding_mode)
    new_conv_out.weight.data = new_w2
    new_conv_out.bias.data = new_b2
    vae.decoder.conv_out = new_conv_out

    # === 更新 config ===
    # diffusers ConfigMixin 用 frozen_dict, 需要用 __setattr__ 绕过
    vae.register_to_config(in_channels=1, out_channels=1)

    # === 保存 ===
    os.makedirs(dst_path, exist_ok=True)
    vae.save_pretrained(dst_path)
    print(f"\n=== 保存 ===")
    print(f"  {src_path} -> {dst_path}")
    print(f"  in/out: 1ch, latent: {config['latent_channels']}ch")

    # 验证: 重新加载 (ignore_mismatched_sizes 因为 config 保存的 in/out_channels 可能不生效)
    vae2 = AutoencoderKL.from_pretrained(dst_path, ignore_mismatched_sizes=True, low_cpu_mem_usage=False)
    assert vae2.encoder.conv_in.weight.shape[1] == 1, "conv_in not 1ch!"
    assert vae2.decoder.conv_out.weight.shape[0] == 1, "conv_out not 1ch!"
    print(f"  验证: conv_in={vae2.encoder.conv_in.weight.shape}, conv_out={vae2.decoder.conv_out.weight.shape}")
    print(f"  DONE!")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="source 3ch VAE path")
    ap.add_argument("--dst", required=True, help="destination 1ch VAE path")
    args = ap.parse_args()
    convert_grayscale(args.src, args.dst)
