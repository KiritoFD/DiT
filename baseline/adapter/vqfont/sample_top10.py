# -*- coding: utf-8 -*-
"""VQ-Font k-shot sampling on top10 eval protocol (k=3 refs from refs.json).

Writes {slot}__{char}.png at 256x256 via the repo Generator.infer path,
mirroring evaluator.save_each_imgs but driven directly by our eval protocol.

Run from baseline/VQ-Font:
  python ../adapter/vqfont/sample_top10.py ../adapter/vqfont/cfg_top10.yaml \
      --weight work_dir/checkpoints/top10/iterXXXXXX.pth \
      --eval_tag strict84 --save_dir ../results/vqfont/strict84
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())

from datasets import load_lmdb, read_data_from_lmdb  # noqa: E402
from model import generator_dispatch  # noqa: E402
from sconf import Config  # noqa: E402
from top10_sim import SimMatrix  # noqa: E402
from train_top10 import load_pretrain_vae_model  # noqa: E402

DIT = "/root/Workspace/xy/DiT"
EVAL = os.path.join(DIT, "baseline/data/eval")


def get_codebook_detach(component_embeddings, batch_size):
    N, C = component_embeddings.size()
    out = torch.zeros(N, C).cuda() + component_embeddings
    out = out.unsqueeze(0).repeat(batch_size, 1, 1)
    return out.detach()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config_paths", nargs="+")
    ap.add_argument("--weight", required=True)
    ap.add_argument("--eval_tag", default="strict84", choices=["strict84", "seen20"])
    ap.add_argument("--save_dir", required=True)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--random_init", action="store_true",
                    help="skip gen ckpt loading (inference-path probe only)")
    ap.add_argument("--limit", type=int, default=0, help="only first N items")
    ap.add_argument("--left_argv", nargs="*", default=[])
    args, left_argv = ap.parse_known_args()

    cfg = Config(*args.config_paths, default="cfgs/defaults.yaml")
    cfg.argv_update(left_argv)

    size = cfg.input_size
    tf = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor(),
                             transforms.Normalize([0.5], [0.5])])

    env = load_lmdb(cfg.data_path)
    env_get = lambda env, x, y, transform: transform(  # noqa: E731
        read_data_from_lmdb(env, f"{x}_{y}")["img"])

    g_cls = generator_dispatch()
    gen = g_cls(1, cfg.C, 1, cfg, **cfg.get("g_args", {}))
    gen.cuda().eval()
    component_embeddings = load_pretrain_vae_model(cfg.vae_pth, gen)
    if not args.random_init:
        ckpt = torch.load(args.weight, map_location="cpu")
        sd = ckpt["generator"] if "generator" in ckpt else ckpt
        if any(k.startswith("_orig_mod.") for k in sd):
            sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
        gen.load_state_dict(sd)

    chars_sim_dict = SimMatrix(cfg.sim_npy, cfg.sim_unis)

    with open(os.path.join(EVAL, "refs.json"), encoding="utf-8") as f:
        refs = json.load(f)
    items = sorted((slot, ch) for slot, d in refs.items() for ch in d)
    if args.limit:
        items = items[:args.limit]

    out_dir = args.save_dir
    os.makedirs(out_dir, exist_ok=True)

    font2id = {f: i for i, f in enumerate(sorted(refs.keys()))}
    bs = args.batch_size
    with torch.no_grad():
        for i in range(0, len(items), bs):
            chunk = items[i:i + bs]
            style_ids, style_imgs, style_sidx, trg_ids, trg_sidx = [], [], [], [], []
            content_imgs, trg_unis, ref_unis = [], [], []
            for j, (slot, ch) in enumerate(chunk):
                fid = font2id[slot]
                ref_chars = refs[slot][ch]
                for k, rc in enumerate(ref_chars):
                    style_ids.append(fid)
                    style_sidx.append(i + j)   # unique per item; 3 refs share it
                    img = env_get(env, slot, f"{ord(rc):X}", tf)
                    style_imgs.append(img)
                trg_ids.append(fid)
                trg_sidx.append(0)
                content_imgs.append(env_get(env, cfg.content_font, f"{ord(ch):X}", tf))
                trg_unis.append(f"{ord(ch):X}")
                ref_unis.append([f"{ord(rc):X}" for rc in ref_chars])

            style_imgs = torch.stack(style_imgs).cuda()
            content_imgs = torch.stack(content_imgs).cuda()
            style_ids = torch.tensor(style_ids).cuda()
            trg_ids = torch.tensor(trg_ids).cuda()
            style_sidx = torch.tensor(style_sidx).cuda()
            trg_sidx = torch.tensor(trg_sidx).cuda()
            bs_comp = get_codebook_detach(component_embeddings, len(chunk))

            out, _, _ = gen.infer(style_ids, style_imgs, style_sidx, trg_ids,
                                  content_imgs, bs_comp, trg_unis, ref_unis,
                                  chars_sim_dict, k_shot_tag=True, reduction="mean")

            imgs = out.detach().cpu()
            for (slot, ch), image in zip(chunk, imgs):
                arr = image.squeeze(0).numpy()
                arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
                cv2.imwrite(os.path.join(out_dir, f"{slot}__{ch}.png"), arr)
            print(f"[{i + len(chunk)}/{len(items)}] saved", flush=True)


if __name__ == "__main__":
    main()
