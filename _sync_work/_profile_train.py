# -*- coding: utf-8 -*-
"""手动计时 profile 纯训练步各阶段（CUPTI 权限受限，不用 torch.profiler）。
测 batch=192 vs batch=8 的 forward/backward/opt/ema 耗时，找 infra free lunch。"""
import os, sys, json, time
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.chdir("/root/Workspace/xy/DiT")
sys.path.insert(0, os.getcwd())
import torch
from src.model import DiT_2Cond_models
from src.loss import create_diffusion_or_flow
cfg = json.load(open("src/train/configs/s28_std_dino_pretrain.json", encoding="utf-8"))
DEV = "cuda"

model = DiT_2Cond_models[cfg["model"]](
    input_size=32, in_channels=4, learn_sigma=False,
    num_calligraphers=cfg["num_calligraphers"], num_characters=cfg["num_characters"],
    condition_fusion=cfg["condition_fusion"],
    callig_embed_dim=cfg["callig_embed_dim"], char_embed_dim=cfg["char_embed_dim"],
    char_proj_mode=cfg.get("char_proj_mode", "full"),
    cond_drop_all_prob=cfg["cond_drop_all_prob"], cond_drop_one_prob=cfg["cond_drop_one_prob"],
    cond_drop_which_glyph_prob=cfg["cond_drop_which_glyph_prob"],
    use_std_dino_char_embedder=cfg.get("use_std_dino_char_embedder", False),
    std_dino_table_path=cfg.get("std_dino_table_path"),
    norm_type=cfg.get("norm_type","rms"), mlp_type=cfg.get("mlp_type","swiglu"),
    qk_norm=bool(cfg.get("qk_norm",1)), rope=bool(cfg.get("rope",1)),
    rope_theta=cfg.get("rope_theta",100.0), attn_impl=cfg.get("attn_impl","sdpa"),
)
model.train().to(DEV)
trainable = [p for p in model.parameters() if p.requires_grad]
opt = torch.optim.AdamW(trainable, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
ema_params = [p.detach().clone() for p in model.parameters()]
diffusion = create_diffusion_or_flow(timestep_respacing="", diffusion_type="flow")
NCH = 7026

def ema_step(decay=0.9999):
    with torch.no_grad():
        for p, mp in zip(model.parameters(), ema_params):
            mp.data.mul_(decay).add_(p.data, alpha=1 - decay)

def run_step(B, do_backward=True, do_opt=True, do_ema=True):
    x = torch.randn(B, 4, 32, 32, device=DEV)
    t = diffusion.sample_t(B, DEV)
    y_callig = torch.randint(0, cfg["num_calligraphers"], (B,), device=DEV)
    script = torch.randint(0, 5, (B,), device=DEV)
    char = torch.randint(0, NCH, (B,), device=DEV)
    y_char = script * NCH + char
    mk = {"y_callig": y_callig, "y_char": y_char}
    with torch.autocast("cuda", dtype=torch.bfloat16):
        ld = diffusion.training_losses(model, x, t, mk)
    loss = ld["loss"].mean()
    if do_backward:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
    if do_opt:
        opt.step(); opt.zero_grad()
    if do_ema:
        ema_step()

def bench(B, n=5):
    for _ in range(2):  # warmup
        run_step(B)
    torch.cuda.synchronize()
    # 分段计时
    def timeit(fn, iters=n):
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        return (time.time() - t0) / iters
    t_forward = timeit(lambda: run_step(B, False, False, False))
    t_backward = timeit(lambda: run_step(B, True, False, False)) - t_forward
    t_opt = timeit(lambda: run_step(B, True, True, False)) - t_forward - t_backward
    t_ema = timeit(lambda: run_step(B, True, True, True)) - t_forward - t_backward - t_opt
    print(f"  batch={B}: fwd={t_forward*1000:.1f}ms  bwd={t_backward*1000:.1f}ms  "
          f"opt={t_opt*1000:.1f}ms  ema={t_ema*1000:.1f}ms  total={sum([t_forward,t_backward,t_opt,t_ema])*1000:.1f}ms "
          f"({1/sum([t_forward,t_backward,t_opt,t_ema]):.2f} steps/s)")
    return t_forward, t_backward, t_opt, t_ema

print("=== batch scaling (samples/s) ===")
for bs in [128, 256, 512, 768]:
    try:
        bm = DiT_2Cond_models[cfg["model"]](
            input_size=32, in_channels=4, learn_sigma=False,
            num_calligraphers=cfg["num_calligraphers"], num_characters=cfg["num_characters"],
            condition_fusion=cfg["condition_fusion"],
            callig_embed_dim=cfg["callig_embed_dim"], char_embed_dim=cfg["char_embed_dim"],
            char_proj_mode=cfg.get("char_proj_mode", "full"),
            cond_drop_all_prob=cfg["cond_drop_all_prob"], cond_drop_one_prob=cfg["cond_drop_one_prob"],
            cond_drop_which_glyph_prob=cfg["cond_drop_which_glyph_prob"],
            use_std_dino_char_embedder=cfg.get("use_std_dino_char_embedder", False),
            std_dino_table_path=cfg.get("std_dino_table_path"),
            norm_type=cfg.get("norm_type","rms"), mlp_type=cfg.get("mlp_type","swiglu"),
            qk_norm=bool(cfg.get("qk_norm",1)), rope=bool(cfg.get("rope",1)),
            rope_theta=cfg.get("rope_theta",100.0), attn_impl="xformers")
        bm.train().to(DEV)
        bo = torch.optim.AdamW([p for p in bm.parameters() if p.requires_grad], lr=1e-4)
        bp = [p.detach().clone() for p in bm.parameters()]
        def bstep():
            x = torch.randn(bs, 4, 32, 32, device=DEV)
            t = diffusion.sample_t(bs, DEV)
            yc = torch.randint(0, cfg["num_calligraphers"], (bs,), device=DEV)
            sc = torch.randint(0, 5, (bs,), device=DEV)
            ch = torch.randint(0, NCH, (bs,), device=DEV)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                ld = diffusion.training_losses(bm, x, t, {"y_callig": yc, "y_char": sc*NCH+ch})
            ld["loss"].mean().backward()
            torch.nn.utils.clip_grad_norm_([p for p in bm.parameters() if p.requires_grad], 1.0)
            bo.step(); bo.zero_grad()
            with torch.no_grad():
                for p, mp in zip(bm.parameters(), bp):
                    mp.data.mul_(0.9999).add_(p.data, alpha=1-0.9999)
        for _ in range(2): bstep()
        torch.cuda.synchronize(); t0=time.time()
        for _ in range(4): bstep()
        torch.cuda.synchronize(); dt=(time.time()-t0)/4
        print(f"  batch={bs}: {dt*1000:.0f}ms/step  {bs/dt/1000:.0f} samples/s  mem={torch.cuda.memory_reserved()/1e9:.2f}G")
        del bm, bo, bp; torch.cuda.empty_cache()
    except Exception as e:
        print(f"  batch={bs}: ERR {e}")
print("=== zero_grad set_to_none vs zero_ (batch=192) ===")
for ztn in [True, False]:
    m = DiT_2Cond_models[cfg["model"]](
        input_size=32, in_channels=4, learn_sigma=False,
        num_calligraphers=cfg["num_calligraphers"], num_characters=cfg["num_characters"],
        condition_fusion=cfg["condition_fusion"],
        callig_embed_dim=cfg["callig_embed_dim"], char_embed_dim=cfg["char_embed_dim"],
        char_proj_mode=cfg.get("char_proj_mode", "full"),
        cond_drop_all_prob=cfg["cond_drop_all_prob"], cond_drop_one_prob=cfg["cond_drop_one_prob"],
        cond_drop_which_glyph_prob=cfg["cond_drop_which_glyph_prob"],
        use_std_dino_char_embedder=cfg.get("use_std_dino_char_embedder", False),
        std_dino_table_path=cfg.get("std_dino_table_path"),
        norm_type=cfg.get("norm_type","rms"), mlp_type=cfg.get("mlp_type","swiglu"),
        qk_norm=bool(cfg.get("qk_norm",1)), rope=bool(cfg.get("rope",1)),
        rope_theta=cfg.get("rope_theta",100.0), attn_impl="xformers")
    m.train().to(DEV)
    o = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-4)
    ep = [p.detach().clone() for p in m.parameters()]
    def zstep():
        x = torch.randn(192, 4, 32, 32, device=DEV)
        t = diffusion.sample_t(192, DEV)
        yc = torch.randint(0, cfg["num_calligraphers"], (192,), device=DEV)
        sc = torch.randint(0, 5, (192,), device=DEV); ch = torch.randint(0, NCH, (192,), device=DEV)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ld = diffusion.training_losses(m, x, t, {"y_callig": yc, "y_char": sc*NCH+ch})
        ld["loss"].mean().backward()
        torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad], 1.0)
        o.step(); o.zero_grad(set_to_none=ztn)
        with torch.no_grad():
            for p, mp in zip(m.parameters(), ep):
                mp.data.mul_(0.9999).add_(p.data, alpha=1-0.9999)
    for _ in range(2): zstep()
    torch.cuda.synchronize(); t0=time.time()
    for _ in range(5): zstep()
    torch.cuda.synchronize(); dt=(time.time()-t0)/5
    print(f"  set_to_none={ztn}: total={dt*1000:.1f}ms/step ({1/dt:.2f} steps/s)")
    del m, o, ep; torch.cuda.empty_cache()
print("=== attn_impl variants (batch=192) ===")
for ai in ["xformers", "eager"]:
    try:
        m = DiT_2Cond_models[cfg["model"]](
            input_size=32, in_channels=4, learn_sigma=False,
            num_calligraphers=cfg["num_calligraphers"], num_characters=cfg["num_characters"],
            condition_fusion=cfg["condition_fusion"],
            callig_embed_dim=cfg["callig_embed_dim"], char_embed_dim=cfg["char_embed_dim"],
            char_proj_mode=cfg.get("char_proj_mode", "full"),
            cond_drop_all_prob=cfg["cond_drop_all_prob"], cond_drop_one_prob=cfg["cond_drop_one_prob"],
            cond_drop_which_glyph_prob=cfg["cond_drop_which_glyph_prob"],
            use_std_dino_char_embedder=cfg.get("use_std_dino_char_embedder", False),
            std_dino_table_path=cfg.get("std_dino_table_path"),
            norm_type=cfg.get("norm_type","rms"), mlp_type=cfg.get("mlp_type","swiglu"),
            qk_norm=bool(cfg.get("qk_norm",1)), rope=bool(cfg.get("rope",1)),
            rope_theta=cfg.get("rope_theta",100.0), attn_impl=ai)
        m.train().to(DEV)
        o = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-4)
        ep = [p.detach().clone() for p in m.parameters()]
        def step():
            x = torch.randn(192, 4, 32, 32, device=DEV)
            t = diffusion.sample_t(192, DEV)
            yc = torch.randint(0, cfg["num_calligraphers"], (192,), device=DEV)
            sc = torch.randint(0, 5, (192,), device=DEV)
            ch = torch.randint(0, NCH, (192,), device=DEV)
            mk = {"y_callig": yc, "y_char": sc*NCH+ch}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                ld = diffusion.training_losses(m, x, t, mk)
            loss = ld["loss"].mean()
            loss.backward(); torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad], 1.0)
            o.step(); o.zero_grad()
            with torch.no_grad():
                for p, mp in zip(m.parameters(), ep):
                    mp.data.mul_(0.9999).add_(p.data, alpha=1-0.9999)
        for _ in range(2): step()
        torch.cuda.synchronize()
        t0=time.time()
        for _ in range(5): step()
        torch.cuda.synchronize()
        dt=(time.time()-t0)/5
        print(f"  attn_impl={ai}: total={dt*1000:.1f}ms/step ({1/dt:.2f} steps/s)  mem={torch.cuda.memory_reserved()/1e9:.2f}G")
        del m, o, ep; torch.cuda.empty_cache()
    except Exception as e:
        print(f"  attn_impl={ai}: ERR {e}")
