# -*- coding: utf-8 -*-
"""GPU 批量生成 strict50 样本 (前50, 同协议 cfg=ckpt内 eval_cfg) -> strict50/ 子目录."""
import os
import sys
import time

import torch as th
from PIL import Image

sys.path.insert(0, "/root/Workspace/xy/DiT")
os.chdir("/root/Workspace/xy/DiT")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import src.eval.gpu_ablate_eval as G  # noqa: E402
from src.eval.inference import make_eval_cache, load_eval_vae  # noqa: E402
from src.utils.callig_map import load_callig_id_map  # noqa: E402

CSV = "assets/eval_fame3_strict50.csv"
N = 50


def save_png(t, path):
    x = ((t.clamp(-1, 1) + 1) / 2 * 255).byte().permute(1, 2, 0).cpu().numpy()
    Image.fromarray(x).save(path)


def gen_one(ckpt, dit_batch=4, vae_batch=4):
    t0 = time.time()
    ck = th.load(ckpt, map_location="cpu", weights_only=False)
    na = ck.get("args", {})
    a = vars(na) if not isinstance(na, dict) else na
    model = G.build_model(a, "cuda")
    sd = G.strip(ck.get("ema") or ck.get("model") or ck)
    # 旧 ckpt 带已删除的外挂键 (callig_spatial/callig_basis) -> 剥离 (实测推理影响 ±0.002)
    if any("callig_spatial" in k or "callig_basis" in k for k in sd):
        sd = {k: v for k, v in sd.items()
              if "callig_spatial" not in k and "callig_basis" not in k}
        print("  [strip] callig_spatial keys removed", flush=True)
    miss, unexp = model.load_state_dict(sd, strict=False)
    assert len(unexp) == 0, f"unexpected={unexp[:5]}"
    del ck
    cmap = None
    if a.get("callig_id_map"):
        p = a["callig_id_map"]
        if not os.path.isabs(p) and not os.path.exists(p):
            p = os.path.join(G.BASE, p)
        cmap, _ = load_callig_id_map(p)
    cache = make_eval_cache(CSV, a.get("img_root"), None, 256, N, 8, 4, 0.18215,
                            skel_latent_shards_dir=a.get("skel_latent_shards_dir"),
                            callig_id_map=cmap)
    noise, conds = cache["noise"], cache["conds"]
    g = cache["skels_latent"].float()
    gts = cache["gts"]
    n_eff = gts.shape[0]
    cfg = float(a.get("eval_cfg") or a.get("gpu_eval_cfg") or 0.7)
    steps = int(a.get("eval_steps") or 50)
    shift = float(a.get("shift", 1.0))
    seg = os.path.dirname(os.path.dirname(ckpt))
    step = int(os.path.basename(ckpt).split(".")[0])
    outd = os.path.join(seg, "eval_samples_ctrl", f"step{step:07d}", "strict50")
    os.makedirs(outd, exist_ok=True)
    vae = load_eval_vae("cuda", "data/pretrained/sd-vae-ft-ema")
    lat = G.heun_gpu(model, noise, conds, cfg, dit_batch, skel=g,
                     steps=steps, shift=shift, dev="cuda")
    for i0 in range(0, n_eff, vae_batch):
        i1 = min(i0 + vae_batch, n_eff)
        with th.autocast("cuda", dtype=th.bfloat16):
            dec = vae.decode(lat[i0:i1].to("cuda") / 0.18215).sample
        for j in range(i0, i1):
            save_png(dec[j - i0].float(), os.path.join(outd, f"g{j}.png"))
            save_png(gts[j], os.path.join(outd, f"gt{j}.png"))
        th.cuda.empty_cache()
    del model, vae
    th.cuda.empty_cache()
    print(f"DONE {os.path.basename(seg)}@{step} n={n_eff} cfg={cfg} "
          f"({time.time()-t0:.0f}s) -> {outd}", flush=True)


JOBS = [
    ("v10a_skel_cond_pretrain", 127500),
    ("v10b_skel_only_pretrain", 85000),
    ("v10b_stdskel_pretrain", 2000),
    ("v10b_stdskel_fame3", 57500),
    ("v10b_stdskel_fame3_d01", 20000),
    ("v10b_stdskel_fame3_deep", 30000),
    ("v10b_stdskel_fame3_mid", 1000),
    ("v10brepa_strong", 7500),
    ("v10b_stdskel_fame3_sp2", 27500),
    ("v10b_stdskel_fame3_c41x_scratch", 12500),
    ("v10b_stdskel_fame3_c41x_cos_clean", 175000),
    ("v10b_stdskel_fame3_c41x_cos_e", 390000),
]
R = "/root/Workspace/xy/DiT/assets/results"
for run, step in JOBS:
    tag = f"{step:07d}"
    cks = [p for p in [f"{R}/{run}/{os.path.basename(d)}/checkpoints/{tag}.pt"
                       for d in [""]] if False]
    import glob
    ck = None
    for d in glob.glob(f"{R}/{run}/*/"):
        p = os.path.join(d, "checkpoints", f"{tag}.pt")
        if os.path.exists(p):
            ck = p
            break
    if ck is None:
        print(f"[skip] {run}@{step} ckpt missing", flush=True)
        continue
    seg = os.path.dirname(os.path.dirname(ck))
    if os.path.exists(os.path.join(seg, "eval_samples_ctrl", f"step{tag}", "strict50", "gt49.png")):
        print(f"[done] {run}@{step} exists", flush=True)
        continue
    try:
        gen_one(ck)
    except Exception as e:
        print(f"[fail] {run}@{step}: {e!r}", flush=True)
print("ALL_GPU_DONE", flush=True)
