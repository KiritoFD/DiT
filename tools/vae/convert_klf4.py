# -*- coding: utf-8 -*-
"""
convert_klf4.py — 把 kl-f4 (ldm 格式) 转成 diffusers AutoencoderKL.

kl-f4 decoder 用 3 个 resblocks per up stage (layers_per_block=3 在 decoder 侧),
diffusers 的 UpDecoderBlock2D 用 layers_per_block + 1 个 resnet.
所以 layers_per_block=2 → encoder 2 resnets, decoder 3 resnets. 完美匹配.

kl-f4 mid block 有 attention (attn_1), 用 q/k/v (不是 to_q/to_k/to_v).
需要映射: attn_1.q → attentions.0.to_q, attn_1.k → to_k, attn_1.v → to_v,
         attn_1.proj_out → to_out.0

用法:
  python tools/vae/convert_klf4.py --ckpt _sync_work/kl-f4/model.ckpt --out pretrained_models/kl-f4
"""
import os
import sys
import json
import argparse
import torch
import re


def remap_attn_key(k):
    """Map ldm attention keys to diffusers attention keys (on already-mid-block-remapped keys)."""
    # After mid block remap: encoder.mid_block.attentions.0.q.weight -> to_q.weight
    k = k.replace(".attentions.0.q.", ".attentions.0.to_q.")
    k = k.replace(".attentions.0.k.", ".attentions.0.to_k.")
    k = k.replace(".attentions.0.v.", ".attentions.0.to_v.")
    k = k.replace(".attentions.0.proj_out.", ".attentions.0.to_out.0.")
    k = k.replace(".attentions.0.norm.", ".attentions.0.group_norm.")
    return k


def remap_key(k):
    """Convert ldm VAE key to diffusers AutoencoderKL key."""
    # Drop loss.* and other non-VAE keys
    if not (k.startswith("encoder.") or k.startswith("decoder.") or
            k.startswith("quant_conv") or k.startswith("post_quant_conv")):
        return None

    # encoder.down.{i}.block.{j}.xxx -> encoder.down_blocks.{i}.resnets.{j}.xxx
    k = re.sub(r"encoder\.down\.(\d+)\.block\.(\d+)\.", r"encoder.down_blocks.\1.resnets.\2.", k)
    # encoder.down.{i}.downsample.conv -> encoder.down_blocks.{i}.downsamplers.0.conv
    k = re.sub(r"encoder\.down\.(\d+)\.downsample\.conv", r"encoder.down_blocks.\1.downsamplers.0.conv", k)

    # decoder.up.{i} -> up_blocks.{2-i} (reversed order)
    def _dec_remap(m):
        i = int(m.group(1))
        return f"decoder.up_blocks.{2 - i}."
    k = re.sub(r"decoder\.up\.(\d+)\.", _dec_remap, k)
    k = re.sub(r"up_blocks\.(\d+)\.block\.(\d+)\.", r"up_blocks.\1.resnets.\2.", k)
    k = re.sub(r"up_blocks\.(\d+)\.upsample\.conv", r"up_blocks.\1.upsamplers.0.conv", k)

    # mid blocks: encoder.mid.block_1 -> encoder.mid_block.resnets.0
    k = k.replace("encoder.mid.block_1.", "encoder.mid_block.resnets.0.")
    k = k.replace("encoder.mid.block_2.", "encoder.mid_block.resnets.1.")
    k = k.replace("decoder.mid.block_1.", "decoder.mid_block.resnets.0.")
    k = k.replace("decoder.mid.block_2.", "decoder.mid_block.resnets.1.")
    # encoder.mid.attn_1 -> encoder.mid_block.attentions.0
    k = k.replace("encoder.mid.attn_1.", "encoder.mid_block.attentions.0.")
    k = k.replace("decoder.mid.attn_1.", "decoder.mid_block.attentions.0.")

    # Now remap attention sub-keys (q->to_q, k->to_k, etc.)
    k = remap_attn_key(k)

    # nin_shortcut -> conv_shortcut
    k = k.replace(".nin_shortcut.", ".conv_shortcut.")

    # norm_out -> conv_norm_out (diffusers name)
    k = k.replace("encoder.norm_out.", "encoder.conv_norm_out.")
    k = k.replace("decoder.norm_out.", "decoder.conv_norm_out.")

    return k


def convert(ckpt_path, out_dir):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck.get("state_dict", ck)

    # Remap keys + reshape attention weights from 1x1 conv to linear
    new_sd = {}
    skipped = []
    for k, v in sd.items():
        nk = remap_key(k)
        if nk is None:
            skipped.append(k)
            continue
        # Attention weights: ldm stores as Conv2d (out, in, 1, 1), diffusers expects Linear (out, in)
        if any(x in nk for x in [".to_q.weight", ".to_k.weight", ".to_v.weight", ".to_out.0.weight"]):
            if v.ndim == 4:  # (out, in, 1, 1) -> (out, in)
                v = v.squeeze(-1).squeeze(-1)
        new_sd[nk] = v

    print(f"Remapped {len(new_sd)} keys, skipped {len(skipped)} (loss/aux)")

    # Config: layers_per_block=2 means encoder has 2 resnets, decoder has 3
    config = {
        "_class_name": "AutoencoderKL",
        "sample_size": 256,
        "in_channels": 3,
        "out_channels": 3,
        "latent_channels": 3,
        "block_out_channels": [128, 256, 512],
        "layers_per_block": 2,
        "down_block_types": ["DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D"],
        "up_block_types": ["UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"],
        "act_fn": "silu",
        "norm_num_groups": 32,
        "scaling_factor": 0.18215,
    }

    print(f"\n=== kl-f4 VAE config ===")
    print(f"  in/out: 3ch, latent: 3ch, blocks: [128,256,512], layers: 2, f4")
    print(f"  256x256 -> 3x64x64")

    os.makedirs(out_dir, exist_ok=True)

    from diffusers import AutoencoderKL
    vae = AutoencoderKL(**config)
    missing, unexpected = vae.load_state_dict(new_sd, strict=False)
    print(f"\n=== load_state_dict ===")
    print(f"  missing: {len(missing)} keys")
    print(f"  unexpected: {len(unexpected)} keys")
    if missing:
        print(f"  missing (first 10): {missing[:10]}")
    if unexpected:
        print(f"  unexpected (first 10): {unexpected[:10]}")

    # Verify: if no missing/unexpected, it's a perfect match
    if not missing and not unexpected:
        print("  PERFECT MATCH!")
    else:
        print("  WARNING: incomplete match, check above")

    vae.save_pretrained(out_dir)
    print(f"\nSaved to {out_dir}")
    return config


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="_sync_work/kl-f4/model.ckpt")
    ap.add_argument("--out", default="pretrained_models/kl-f4")
    args = ap.parse_args()
    convert(args.ckpt, args.out)
