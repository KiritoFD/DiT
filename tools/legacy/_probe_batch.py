"""Batch-size probe for DiT-2Cond-S/{2,4,8} on kl-f4 VAE (top6 dataset prep, 106k config).

Replicates the real training step memory profile from train.py:
  - model (fp32) + EMA deepcopy (fp32) on GPU
  - AdamW optimizer states on GPU
  - bf16 autocast forward (diffusion.training_losses) + backward
  - input: (B, 3, 64, 64) latent, y_callig (B,), y_char (B,), t (B,)
Reports peak VRAM per batch size; finds max batch that fits on 24GB (RTX 4090).
"""
import os, sys, gc, json, copy
import torch

DT = 4  # vae_downscale
IMAGE_SIZE = 256
LATENT_SIZE = IMAGE_SIZE // DT
LATENT_CH = 3  # kl-f4
COND_DROP_ALL, COND_DROP_ONE = 0.05, 0.25
CHAR_DIM = 768          # S8 config: dims=128/768
CALLIG_DIM = 128
N_CALLIG, N_CHAR = 1011, 35130
TOTAL_GB = 23.5          # safety headroom on 24GB card
SAFE_GB = 22.8

sys.path.insert(0, "/root/Workspace/xy/DiT")
from models import DiT_2Cond_models  # noqa: E402
from diffusion import create_diffusion  # noqa: E402


def build_model(patch_size):
    name = f"DiT-2Cond-S/{patch_size}"
    m = DiT_2Cond_models[name](
        input_size=LATENT_SIZE,
        num_calligraphers=N_CALLIG,
        num_characters=N_CHAR,
        use_checkpoint=False,
        condition_fusion="factorized_add",
        callig_embed_dim=CALLIG_DIM,
        char_embed_dim=CHAR_DIM,
        cond_drop_all_prob=COND_DROP_ALL,
        cond_drop_one_prob=COND_DROP_ONE,
        skel_head_enabled=False,
        use_glyph_cond=False,
        glyph_scale_init=0.4,
        in_channels=LATENT_CH,
    )
    return m


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def probe_one(patch_size, candidates):
    """For a patch size, run real training steps at each candidate batch, record peak VRAM."""
    print(f"\n{'='*70}\nDiT-2Cond-S/{patch_size}: latent {LATENT_SIZE}², tokens per latent = {LATENT_SIZE**2//patch_size**2}\n{'='*70}")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = build_model(patch_size).cuda().train()
    ema_model = copy.deepcopy(model).eval()
    for p in ema_model.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=2e-4, weight_decay=0.02)
    diffusion = create_diffusion("")  # 1000-step schedule, LEARNED_RANGE like train

    n_params = count_params(model)
    ema_params = count_params(ema_model)
    base_mem = torch.cuda.memory_allocated() / 1024**3
    print(f"trainable={n_params/1e6:.2f}M ema={ema_params/1e6:.2f}M "
          f"base(static, incl optimizer states)={base_mem:.2f} GB")

    results = []
    for bs in candidates:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        x = torch.randn(bs, LATENT_CH, LATENT_SIZE, LATENT_SIZE, device="cuda")
        y_callig = torch.randint(0, N_CALLIG, (bs,), device="cuda")
        y_char = torch.randint(0, N_CHAR, (bs,), device="cuda")
        t = torch.randint(0, 1000, (bs,), device="cuda")
        model_kwargs = dict(y_callig=y_callig, y_char=y_char)
        try:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
                loss = loss_dict["loss"].mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            peak = torch.cuda.max_memory_allocated() / 1024**3
            results.append((bs, peak, None))
            status = "OK" if peak < TOTAL_GB else "over-24G"
            print(f"  batch={bs:4d}: peak VRAM = {peak:6.2f} GB   [{status}]")
            if peak >= TOTAL_GB:
                break  # keep going no point; record but stop
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            results.append((bs, None, "OOM"))
            print(f"  batch={bs:4d}: OOM")
            break
        finally:
            del x, y_callig, y_char, t
    del model, ema_model, opt
    gc.collect()
    torch.cuda.empty_cache()
    return results


def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}  "
          f"total={torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")
    out = {}
    # patch=4 first (as user requested order: 4, then 2, 8)
    order = [4, 2, 8]
    candidates = [16, 32, 48, 64, 96, 128, 160, 192, 224, 256, 320, 384]
    for patch in order:
        res = probe_one(patch, candidates)
        ok = [r for r in res if r[2] is None and r[1] is not None and r[1] < SAFE_GB]
        best = max(ok, key=lambda r: r[0]) if ok else None
        out[str(patch)] = {
            "results": [[b, round(p, 2) if p else None, s] for b, p, s in res],
            "max_batch_under_22.8G": best[0] if best else None,
        }
        verdict = (f"RECOMMEND batch={best[0]} (peak {best[1]:.2f}G)"
                   if best else "NO batch fits under 22.8G!")
        print(f"\n>>> patch={patch}: {verdict}")
    with open("/tmp/probe_batch_result.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved /tmp/probe_batch_result.json")


if __name__ == "__main__":
    main()