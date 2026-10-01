"""从零推理: 联合训练的 SkelNet (取 ckpt 里的 gen_ema) 生成骨架 -> 解码校验 -> 出图。

为什么另开一个: 训练期 eval_dump 是在编译(CUDA Graph)+训练状态下采样并直接写盘,
**没有任何"解码后是否有墨"的校验**, 结果落盘的 latent 解码出来是白图却无人发现。
本脚本只做推理:
  · 复用 model_io 拿架构, 复用 joint ckpt 的 gen_ema 权重 (与评测口径一致)
  · 复用训练脚本的 gen_sample (同一套采样, 不重写)
  · 复用 src.eval.in_mem_eval._get_vae 解码
  · **硬校验**: decode 后 ink 必须 > 0, 否则直接报错并打印 decode 的统计
union 模型 2 的推理沿用本脚本的加载 + 解码部分。

用法:
  python tools/infer_joint_skel.py \
      --gen-ckpt assets/results/v33_stage1_xs/.../0030000.pt \
      --joint    assets/results/v34_e2e/joint_005000.pt \
      --csv assets/eval_top10_seen_20.csv --n 8 \
      --out _ot_scratch/infer_joint.png --dump-dir /tmp/infer_dump
"""
import argparse
import json
import os

import numpy as np
import torch as th
from PIL import Image, ImageDraw

from src.eval import model_io
from src.eval.in_mem_eval import _get_vae
from src.utils.latent_dataset import MCCDLatentDataset
from tools.train_joint_stage1_stage2 import gen_sample, strip_orig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-ckpt", required=True, help="原生 stage1 ckpt (取架构/初始权重)")
    ap.add_argument("--joint", default="", help="联合 ckpt; 有则用它覆盖权重")
    ap.add_argument("--weights", default="ema", choices=("ema", "gen"),
                    help="★ 取 ckpt 里哪份权重: ema(评测口径) 或 gen(在线权重)。\n"
                         "  两者若表现不同, 说明 EMA 更新/保存那条路坏了。")
    ap.add_argument("--csv", default="assets/eval_top10_seen_20.csv")
    ap.add_argument("--shards-img", default="data/top10_style23/shards_img")
    ap.add_argument("--shards-std", default="data/top10_style23/shards_std")
    ap.add_argument("--shards-gt", default="data/top10_style23/shards_gtskel_w7")
    ap.add_argument("--callig-map", default="assets/callig_script_id_map_top10.json")
    ap.add_argument("--vae", default="data/pretrained/pretrained_models/sd-vae-ft-ema")
    ap.add_argument("--sf", type=float, default=0.18215)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--beta", type=float, default=1.0,
                    help="条件混合: g = β·pred + (1-β)·std。评测 ckpt 里存的 "
                         "eval_blend_alpha=0.5, 复现它的口径用 0.5")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--cell", type=int, default=256)
    ap.add_argument("--out", default="_ot_scratch/infer_joint.png")
    ap.add_argument("--dump-dir", default="", help="把生成骨架落盘成 shard (供下游判分)")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    dev = a.device

    print(f"[1] 架构/权重: {a.gen_ckpt}", flush=True)
    gen, ga = model_io.load_model_from_ckpt(a.gen_ckpt, device=dev, use_ema=True)
    if a.joint:
        ck = th.load(a.joint, map_location="cpu", weights_only=False)
        _key = "gen_ema" if a.weights == "ema" else "gen"
        sd = strip_orig(ck.get(_key) or ck.get("gen") or {})
        ms, us = gen.load_state_dict(sd, strict=False)
        print(f"[1b] 覆盖权重 <- {a.joint} [{_key}] (step={ck.get('step')}) "
              f"missing={len(ms)} unexpected={len(us)}", flush=True)
    gen.eval()

    print("[2] 数据 (latent 只读 eval 集需要的行)", flush=True)
    csmap = json.load(open(a.callig_map, encoding="utf-8"))
    ds = MCCDLatentDataset(
        csv_file=a.csv, latent_shards_dir=a.shards_img, img_root="",
        image_size=256, is_train=False, preload=True, load_image=False,
        skel_latent_shards_dir=a.shards_std,
        aux_latent_shards_dirs=[a.shards_gt],
        callig_id_map=None, callig_script_map=csmap)
    n = min(a.n, len(ds))
    print(f"    数据集 {len(ds)} 条, 取前 {n} 条", flush=True)

    vae = _get_vae(dev, a.vae).eval()

    def dec(lat):                       # lat: (B,4,32,32) tensor on dev
        with th.no_grad(), th.autocast("cuda", dtype=th.bfloat16):
            d = vae.decode(lat / a.sf).sample
        d = d.float().mean(1)           # (B,256,256), 值域约 [-1,1]
        return d

    ids, conds, gens, tgts = [], [], [], []
    for i in range(n):
        b = ds[i]
        conds.append(b["skel_latent"].float())
        tgts.append(b["aux_latents"].float())
        ids.append(int(b["img_id"]))
    cond = th.stack(conds).to(dev)
    tgt = th.stack(tgts).to(dev)

    with th.no_grad():
        z = gen_sample(gen, cond, th.tensor(
            [int(ds[i]["y_callig"]) for i in range(n)], dtype=th.long, device=dev),
            a.steps, dev)
    if a.beta < 1.0:
        # 复现评测口径: harness 存的 eval_blend_alpha 会把 std 骨架混进 pred
        z = a.beta * z + (1.0 - a.beta) * cond
        print(f"[2b] 条件混合 β={a.beta} (评测口径)")
    gens = z

    # ── ★ 硬校验: 解码后必须有墨 ───────────────────────────────────────
    dcond, dgen, dtgt = dec(cond), dec(gens), dec(tgt)
    ink = {nm: (d < 0) for nm, d in (("cond", dcond), ("gen", dgen), ("tgt", dtgt))}
    print("\n[3] 解码校验 (ink = decode<0 的像素占比; 全 0 = 白图 = 坏)")
    for nm, d in (("cond", dcond), ("gen", dgen), ("tgt", dtgt)):
        print(f"    {nm}: ink={float(ink[nm].float().mean()):.4f}  decode "
              f"min={float(d.min()):+.3f} max={float(d.max()):+.3f} "
              f"mean={float(d.mean()):+.3f}")
    ratio = float(ink["gen"].float().mean() / max(float(ink["tgt"].float().mean()), 1e-9))
    print(f"    gen/tgt 墨量比 = {ratio:.2f}x")
    if float(ink["gen"].float().mean()) <= 0.001:
        raise SystemExit(
            "[FATAL] 生成骨架解码后没有墨 -> 是坏输出, 不要拿它当条件/结论。\n"
            f"  生成 latent 统计: mean={float(gens.mean()):.3f} "
            f"std={float(gens.std()):.3f} |max|={float(gens.abs().max()):.3f}\n"
            f"  decode 统计见上; 对照 cond/tgt 的 ink 是否正常。")

    # ── poster: cond | gen | tgt (原生 256, 不缩放) ────────────────────
    cell = a.cell
    W, H = 4 + cell * n, 3 * (cell + 20) + 4
    cv = Image.new("L", (W, H), 255)
    dr = ImageDraw.Draw(cv)
    for r, (nm, m) in enumerate((("std(g) cond", ink["cond"]),
                                 ("gen (ours)", ink["gen"]),
                                 ("GT(w7 tgt)", ink["tgt"]))):
        y = 2 + r * (cell + 20)
        dr.text((4, y), nm, fill=0)
        for c in range(n):
            bits = m[c].cpu().numpy()
            cv.paste(Image.fromarray(np.where(bits, 0, 255).astype(np.uint8)),
                     (2 + c * cell, y + 18))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    cv.save(a.out)
    print(f"[4] poster -> {a.out}")

    if a.dump_dir:
        os.makedirs(a.dump_dir, exist_ok=True)
        p = os.path.join(a.dump_dir, "shard_00000.npz")
        np.savez_compressed(p, latents=gens.float().cpu().numpy().astype(np.float16),
                            img_ids=np.array(ids, dtype=np.int64))
        print(f"[5] 生成骨架落盘 -> {p} (n={len(ids)})")


if __name__ == "__main__":
    main()
