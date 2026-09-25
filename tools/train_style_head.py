#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""只训低秩风格头, 让它在 VAE latent 里预测 20px 实例骨架。

主干全部冻结。头的输入是标准骨架 g 和书家向量, 输出是 (4,32,32) latent,
损失是它和 data/50k/shards_instskel20 的 MSE。两边同一个 VAE, 同一个网格。

不把这个损失加进 flow loss, 也不把预测喂回主干。训完的头以后再接入。

    python tools/train_style_head.py --ckpt <pt> --steps 3000 --device cuda
"""
import argparse
import json
import os
import sys

import torch as th

ROOT = "/root/Workspace/xy/DiT"
sys.path.insert(0, ROOT)
os.chdir(ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="assets/style_head_instskel20.pt")
    a = ap.parse_args()
    dev = a.device

    from src.eval.model_io import load_model_from_ckpt
    # checkpoint 是加 to_latent 之前存的, 整模型 strict 加载会缺这两个键。
    # 头的参数先按旧形状加载, 解码器保持新建时的初始化。
    # 先把 ckpt 里的 lowrank_spatial.* 去掉再加载: 模型按 rank=0 建, 没有这些键;
    # 头的权重在下面单独拷进新建的头。
    _ck = th.load(a.ckpt, map_location="cpu", weights_only=False)
    for _slot in ("ema", "model"):
        if _ck.get(_slot):
            _ck[_slot] = {k: v for k, v in _ck[_slot].items()
                          if "lowrank_spatial." not in k}
    _stripped = a.ckpt + ".stripped.pt"
    th.save(_ck, _stripped)
    del _ck
    model, args_ns = load_model_from_ckpt(
        _stripped, device=dev, use_ema=True, verbose=False,
        model_overrides={"lowrank_spatial_rank": 0})
    head = model.__class__.__dict__  # 占位, 下面重建
    from src.model.dit import LowRankSpatialStyleFiLM
    model.lowrank_spatial = LowRankSpatialStyleFiLM(128, 384, rank=32).to(dev)
    # 位置编码只在建头时注册。覆盖 rank 关掉了它, 这里补回来。
    if model.local_pos is None:
        from src.model.dit import get_2d_sincos_pos_embed
        pe = th.from_numpy(get_2d_sincos_pos_embed(384, 16)).float().unsqueeze(0)
        if "local_pos" in model.__dict__:
            del model.__dict__["local_pos"]
        model.register_buffer("local_pos", pe.to(dev), persistent=False)
    sd = th.load(a.ckpt, map_location="cpu", weights_only=False)
    src = sd.get("ema") or sd["model"]
    src = {k.replace("_orig_mod.", "", 1): v for k, v in src.items()}
    own = model.lowrank_spatial.state_dict()
    taken = {k[len("lowrank_spatial."):]: v for k, v in src.items()
             if k.startswith("lowrank_spatial.") and k[len("lowrank_spatial."):] in own
             and tuple(v.shape) == tuple(own[k[len("lowrank_spatial."):]].shape)}
    own.update(taken)
    model.lowrank_spatial.load_state_dict(own)
    print(f"[head] 从 ckpt 载入 {len(taken)} 个张量, to_latent 重新初始化", flush=True)
    model.train()
    head = model.lowrank_spatial
    if head is None:
        raise SystemExit("checkpoint 没有 lowrank_spatial, 用带 --lowrank-spatial-rank 的配置")

    for p in model.parameters():
        p.requires_grad_(False)
    for p in list(head.parameters()) + list(model.glyph_embedder.parameters()):
        p.requires_grad_(True)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[head] trainable {n_train:,}", flush=True)

    import json as _json
    from src.utils.latent_dataset import MCCDLatentDataset
    csmap = None
    csm = getattr(args_ns, "callig_script_map", "") or ""
    if csm and os.path.exists(csm):
        csmap = _json.load(open(csm, encoding="utf-8"))
    ds = MCCDLatentDataset(
        csv_file=args_ns.data_csv, latent_shards_dir=args_ns.latent_shards_dir,
        img_root="", image_size=256, preload=False, load_image=False,
        skel_latent_shards_dir=getattr(args_ns, "skel_latent_shards_dir", "") or None,
        inst_skel_shards_dir="data/50k/shards_instskel20",
        callig_id_map=None, callig_script_map=csmap)
    print(f"[data] {len(ds)} 行, inst skel 命中 {len(ds._inst_id_to_shard)}", flush=True)

    opt = th.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                         lr=a.lr, weight_decay=0.0)
    loss_ema = None
    for step in range(1, a.steps + 1):
        idx = th.randint(0, len(ds), (a.batch,)).tolist()
        items = [ds[i] for i in idx]
        g = th.stack([it["skel_latent"] for it in items]).to(dev)
        tgt = th.stack([it["inst_skel"] for it in items]).to(dev)
        yc = th.tensor([int(it["y_callig"]) for it in items], device=dev)
        yh = th.zeros_like(yc)
        x = th.zeros(a.batch, 4, 32, 32, device=dev)
        t = th.zeros(a.batch, device=dev)
        model(x, t, y_callig=yc, y_char=yh, g=g)
        pred = head.decode_latent()
        loss = (pred - tgt).pow(2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        v = float(loss)
        loss_ema = v if loss_ema is None else 0.9 * loss_ema + 0.1 * v
        if step % a.log_every == 0 or step == 1:
            with th.no_grad():
                base = (g - tgt).pow(2).mean()
            print(f"step {step:5d}  loss {v:.4f}  ema {loss_ema:.4f}  "
                  f"g_vs_tgt {float(base):.4f}", flush=True)

    th.save({"head": head.state_dict(),
             "glyph_embedder": model.glyph_embedder.state_dict(),
             "steps": a.steps, "loss_ema": loss_ema}, a.out)
    print(f"[saved] {a.out}", flush=True)


if __name__ == "__main__":
    main()
