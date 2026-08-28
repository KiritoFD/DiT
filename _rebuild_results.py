import json, glob, os, re, csv
os.chdir("/root/Workspace/xy/DiT")

# Determine which series already exist in results.csv (by reading it locally won't work on remote)
# Instead, scan ALL eval_auto_*.json on remote and build rows.
# Column order must match existing header.

HEADER = "series,run,step,model,vae,latent_channels,latent_spatial,vae_scaling_factor,data_csv,img_root,char_embed_dim,callig_embed_dim,num_characters,dino,cond_mode,condition_fusion,sampler,use_skel,skel_root,w_skel,use_canny,w_canny,glyph_init_mix,w_glyph_cond,balance_callig_alpha,balance_char_alpha,lr,weight_decay,warmup_steps,max_steps,batch_size,ema_decay,lr_schedule,min_lr_ratio,resume,train_ctrl_only,main_ckpt,skel_cond_channels,cond_drop_struct_prob,eval_csv,eval_n,eval_steps,eval_cfg,eval_seed,mse,ssim,ssim_old_uniform,skel_iou,lpips,mse_base,ssim_base,mse_ctrl,ssim_ctrl,skel_iou_base,skel_iou_ctrl,lpips_base,lpips_ctrl,delta_ssim,delta_mse,delta_skel_iou,delta_lpips,src"

def load_config(run_dir):
    """Load resolved_config.json from a run directory."""
    p = os.path.join(run_dir, "resolved_config.json")
    if not os.path.exists(p):
        return {}
    return json.load(open(p))

def get_vae_info(cfg):
    vae = cfg.get("vae", "ema")
    if vae == "ema" or vae == "sd-vae-ft-ema":
        return "pretrained_models/sd-vae-ft-ema", 4, 32, 0.18215
    elif "kl-f4" in str(vae):
        return "pretrained_models/kl-f4", 3, 32, 0.102079
    return str(vae), 4, 32, 0.18215

def g(d, k, default=""):
    v = d.get(k, default)
    if v is None: return default
    return v

# Scan all result dirs
all_results = sorted([d for d in os.listdir("5script/results") if os.path.isdir(f"5script/results/{d}")])
rows = []
for series in all_results:
    # Find all run dirs
    run_dirs = sorted(glob.glob(f"5script/results/{series}/*/"))
    for rd in run_dirs:
        run_name = os.path.basename(rd.rstrip("/"))
        ckpt_dir = os.path.join(rd, "checkpoints")
        cfg = load_config(rd)
        # Find all eval_auto_*.json
        evals = sorted(glob.glob(os.path.join(ckpt_dir, "eval_auto_*.json")),
                       key=lambda f: int(re.search(r'eval_auto_(\d+)', f).group(1)) if re.search(r'eval_auto_(\d+)', f) else 0)
        for ef in evals:
            d = json.load(open(ef))
            step = d.get("step", 0)
            vae_path, lat_ch, lat_sp, vae_sf = get_vae_info(cfg)
            # Build row
            row = {
                "series": series,
                "run": run_name,
                "step": step,
                "model": g(cfg, "model"),
                "vae": vae_path,
                "latent_channels": lat_ch,
                "latent_spatial": lat_sp,
                "vae_scaling_factor": vae_sf,
                "data_csv": g(cfg, "data_csv"),
                "img_root": g(cfg, "img_root"),
                "char_embed_dim": g(cfg, "char_embed_dim"),
                "callig_embed_dim": g(cfg, "callig_embed_dim"),
                "num_characters": g(cfg, "num_characters", 35130),
                "dino": g(cfg, "char_dino_embeddings", "none") if cfg.get("char_dino_embeddings") else "none",
                "cond_mode": g(cfg, "cond_mode"),
                "condition_fusion": g(cfg, "condition_fusion"),
                "sampler": g(cfg, "sampler"),
                "use_skel": g(cfg, "use_skel"),
                "skel_root": g(cfg, "skel_root"),
                "w_skel": g(cfg, "w_skel"),
                "use_canny": g(cfg, "use_canny"),
                "w_canny": g(cfg, "w_canny"),
                "glyph_init_mix": g(cfg, "glyph_init_mix"),
                "w_glyph_cond": g(cfg, "w_glyph_cond"),
                "balance_callig_alpha": g(cfg, "balance_callig_alpha"),
                "balance_char_alpha": g(cfg, "balance_char_alpha"),
                "lr": g(cfg, "lr"),
                "weight_decay": g(cfg, "weight_decay"),
                "warmup_steps": g(cfg, "warmup_steps"),
                "max_steps": g(cfg, "max_steps"),
                "batch_size": g(cfg, "global_batch_size"),
                "ema_decay": g(cfg, "ema_decay"),
                "lr_schedule": g(cfg, "lr_schedule"),
                "min_lr_ratio": g(cfg, "min_lr_ratio"),
                "resume": g(cfg, "resume_full", ""),
                "train_ctrl_only": g(cfg, "train_ctrl_only", "False"),
                "main_ckpt": g(cfg, "main_ckpt", ""),
                "skel_cond_channels": g(cfg, "skel_cond_channels", ""),
                "cond_drop_struct_prob": g(cfg, "cond_drop_struct_prob", ""),
                "eval_csv": g(cfg, "eval_csv"),
                "eval_n": g(cfg, "eval_n"),
                "eval_steps": g(cfg, "eval_steps"),
                "eval_cfg": g(cfg, "eval_cfg"),
                "eval_seed": g(cfg, "eval_seed"),
                "mse": d.get("mse", ""),
                "ssim": d.get("ssim", ""),
                "ssim_old_uniform": "",
                "skel_iou": d.get("skel_iou", d.get("skeleton_iou", "")),
                "lpips": d.get("lpips", ""),
                "mse_base": d.get("mse_base", ""),
                "ssim_base": d.get("ssim_base", ""),
                "mse_ctrl": d.get("mse_ctrl", ""),
                "ssim_ctrl": d.get("ssim_ctrl", ""),
                "skel_iou_base": d.get("skel_iou_base", ""),
                "skel_iou_ctrl": d.get("skel_iou_ctrl", ""),
                "lpips_base": d.get("lpips_base", ""),
                "lpips_ctrl": d.get("lpips_ctrl", ""),
                "delta_ssim": d.get("delta_ssim", ""),
                "delta_mse": d.get("delta_mse", ""),
                "delta_skel_iou": d.get("delta_skel_iou", ""),
                "delta_lpips": d.get("delta_lpips", ""),
                "src": "eval_auto",
            }
            rows.append(row)

# Write CSV
cols = HEADER.split(",")
with open("/tmp/results_new.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
    w.writeheader()
    for r in rows:
        w.writerow(r)

print(f"Wrote {len(rows)} rows to /tmp/results_new.csv")
print(f"Series: {sorted(set(r['series'] for r in rows))}")
