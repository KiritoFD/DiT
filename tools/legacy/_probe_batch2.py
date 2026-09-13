"""Refinement probe: patch=2 intermediate batches (16-32) and patch=8 beyond 384."""
import os, sys, gc, json, copy
import torch

DT = 4; IMAGE_SIZE = 256; LATENT_SIZE = IMAGE_SIZE // DT
LATENT_CH = 3
COND_DROP_ALL, COND_DROP_ONE = 0.05, 0.25
CHAR_DIM = 768; CALLIG_DIM = 128
N_CALLIG, N_CHAR = 1011, 35130
SAFE_GB = 22.8

sys.path.insert(0, "/root/Workspace/xy/DiT")
from models import DiT_2Cond_models  # noqa: E402
from diffusion import create_diffusion  # noqa: E402


def build_model(patch_size):
    return DiT_2Cond_models[f"DiT-2Cond-S/{patch_size}"](
        input_size=LATENT_SIZE, num_calligraphers=N_CALLIG, num_characters=N_CHAR,
        use_checkpoint=False, condition_fusion="factorized_add",
        callig_embed_dim=CALLIG_DIM, char_embed_dim=CHAR_DIM,
        cond_drop_all_prob=COND_DROP_ALL, cond_drop_one_prob=COND_DROP_ONE,
        skel_head_enabled=False, use_glyph_cond=False, glyph_scale_init=0.4,
        in_channels=LATENT_CH)


def probe_one(patch_size, candidates):
    print(f"\n{'='*64}\nDiT-2Cond-S/{patch_size}: candidates={candidates}\n{'='*64}")
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    model = build_model(patch_size).cuda().train()
    ema_model = copy.deepcopy(model).eval()
    for p in ema_model.parameters(): p.requires_grad_(False)
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=2e-4, weight_decay=0.02)
    diffusion = create_diffusion("")
    # warm up optimizer states once
    x0 = torch.randn(4, LATENT_CH, LATENT_SIZE, LATENT_SIZE, device="cuda")
    t0 = torch.randint(0, 1000, (4,), device="cuda")
    mk = dict(y_callig=torch.randint(0, N_CALLIG, (4,), device="cuda"),
              y_char=torch.randint(0, N_CHAR, (4,), device="cuda"))
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = diffusion.training_losses(model, x0, t0, mk)["loss"].mean()
    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    del x0, t0, mk; torch.cuda.empty_cache()

    results = []
    for bs in candidates:
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        x = torch.randn(bs, LATENT_CH, LATENT_SIZE, LATENT_SIZE, device="cuda")
        y_callig = torch.randint(0, N_CALLIG, (bs,), device="cuda")
        y_char = torch.randint(0, N_CHAR, (bs,), device="cuda")
        t = torch.randint(0, 1000, (bs,), device="cuda")
        try:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = diffusion.training_losses(model, x, t, dict(y_callig=y_callig, y_char=y_char))["loss"].mean()
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            peak = torch.cuda.max_memory_allocated() / 1024**3
            print(f"  batch={bs:4d}: peak = {peak:6.2f} GB   [{'OK' if peak < SAFE_GB else 'over-safe'}]")
            results.append((bs, peak))
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"  batch={bs:4d}: OOM")
            results.append((bs, None))
        finally:
            del x, y_callig, y_char, t
    del model, ema_model, opt; gc.collect(); torch.cuda.empty_cache()
    return results


def main():
    out = {}
    out["2"] = [[b, round(p, 2) if p else None] for b, p in
                probe_one(2, [20, 24, 28, 32])]
    out["8"] = [[b, round(p, 2) if p else None] for b, p in
                probe_one(8, [448, 512, 640, 768])]
    with open("/tmp/probe_batch_refine.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved /tmp/probe_batch_refine.json")


if __name__ == "__main__":
    main()