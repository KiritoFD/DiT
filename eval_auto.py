"""
In-memory automatic evaluation triggered after each checkpoint save.

The model weights already live on GPU during training; this module evaluates the
*current* weights without reloading anything — the trainer calls
`prepare_eval_cache()` once at startup (pre-encodes N test samples into latents)
and then `eval_in_memory()` after each save.

Procedure (identical to eval_full_3cond.py / the manual remote eval):
  single-step xstart reconstruction at t=T_EVAL, VAE decode, MSE/SSIM vs GT.
"""
import sys, json, os
sys.stdout.reconfigure(encoding='utf-8')
import torch
import torch.nn.functional as F


def _save_eval_visuals(decoded_list, gt_list, conds, vis_out, vis_n=5):
    """把前 vis_n 张 (pred | GT) 横排、多张竖排拼成一张对比图存到 vis_out。

    decoded_list / gt_list: 每张为 (1,3,256,256) 的 [-1,1] tensor (cpu)
    返回 meta dict（含 step 信息供前端展示）。
    """
    from PIL import Image
    k = min(vis_n, len(decoded_list))
    rows = []
    for i in range(k):
        pred = ((decoded_list[i].clamp(-1, 1) + 1) / 2).squeeze(0).permute(1, 2, 0).cpu().float().numpy()
        gt = ((gt_list[i].clamp(-1, 1) + 1) / 2).squeeze(0).permute(1, 2, 0).cpu().float().numpy()
        pred = (pred * 255).clip(0, 255).astype("uint8")
        gt = (gt * 255).clip(0, 255).astype("uint8")
        pw, ph = pred.shape[1], pred.shape[0]
        canvas = Image.new("RGB", (pw * 2, ph), (20, 20, 20))
        canvas.paste(Image.fromarray(pred), (0, 0))
        canvas.paste(Image.fromarray(gt), (pw, 0))
        rows.append(canvas)
    if not rows:
        return None
    W = rows[0].width
    H = sum(r.height for r in rows)
    out = Image.new("RGB", (W, H))
    y = 0
    for r in rows:
        out.paste(r, (0, y))
        y += r.height
    _d = os.path.dirname(vis_out)
    if _d:
        os.makedirs(_d, exist_ok=True)
    out.save(vis_out)
    return vis_out


def _dump_eval_all(decoded_list, gt_list, vis_dir, vis_n=0):
    """把每一张 (pred | GT) 对比图逐张落盘到 vis_dir/pred_%04d.png。
    vis_n<=0 表示全部落盘；否则只落前 vis_n 张。返回文件数。"""
    import os
    from PIL import Image
    os.makedirs(vis_dir, exist_ok=True)
    k = len(decoded_list) if vis_n <= 0 else min(vis_n, len(decoded_list))
    saved = 0
    for i in range(k):
        pred = ((decoded_list[i].clamp(-1, 1) + 1) / 2).squeeze(0).permute(1, 2, 0).cpu().float().numpy()
        gt = ((gt_list[i].clamp(-1, 1) + 1) / 2).squeeze(0).permute(1, 2, 0).cpu().float().numpy()
        pred = (pred * 255).clip(0, 255).astype("uint8")
        gt = (gt * 255).clip(0, 255).astype("uint8")
        pw, ph = pred.shape[1], pred.shape[0]
        canvas = Image.new("RGB", (pw * 2, ph), (20, 20, 20))
        canvas.paste(Image.fromarray(pred), (0, 0))
        canvas.paste(Image.fromarray(gt), (pw, 0))
        canvas.save(os.path.join(vis_dir, f"pred_{i:04d}.png"))
        saved += 1
    return saved


T_EVAL = 150


def _gaussian_window(window_size=11, sigma=1.5, device="cpu"):
    g = torch.arange(window_size, dtype=torch.float32, device=device) - window_size // 2
    g = torch.exp(-(g ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    return (g.reshape(1, 1, window_size, 1) @ g.reshape(1, 1, 1, window_size))


def _ssim(x, y, data_range=1.0, window_size=11, win=None):
    if x.shape[1] == 3:
        return sum(_ssim(x[:, i:i + 1], y[:, i:i + 1], data_range, window_size, win)
                   for i in range(3)) / 3
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    mu_x = F.conv2d(x, win, padding=window_size // 2)
    mu_y = F.conv2d(y, win, padding=window_size // 2)
    mu_x2, mu_y2, mu_xy = mu_x ** 2, mu_y ** 2, mu_x * mu_y
    sx2 = F.conv2d(x * x, win, padding=window_size // 2) - mu_x2
    sy2 = F.conv2d(y * y, win, padding=window_size // 2) - mu_y2
    sxy = F.conv2d(x * y, win, padding=window_size // 2) - mu_xy
    m = ((2 * mu_xy + C1) * (2 * sxy + C2)) / ((mu_x2 + mu_y2 + C1) * (sx2 + sy2 + C2))
    return float(m.mean().item())


def prepare_eval_cache(vae, dataset, device, n=1000, t=T_EVAL, batch_size=16):
    """
    Pre-encode `n` test samples into latents + gather conditions + GT images.
    Returns a dict that `eval_in_memory` reuses across every checkpoint.
    """
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    latents, conds, gts = [], [], []
    with torch.no_grad():
        for b in loader:
            if len(latents) >= n:
                break
            x = b["image"].to(device)          # (B,3,256,256) in [-1,1]
            lat = vae.encode(x).latent_dist.sample().mul_(0.18215)
            latents.append(lat.cpu())
            for _i in range(x.shape[0]):
                conds.append((b["y_callig"][_i].item(), b["y_script"][_i].item(), b["y_char"][_i].item()))
            gts.append(x.cpu())
    latents = torch.cat(latents, dim=0)[:n]
    gts = torch.cat(gts, dim=0)[:n]
    conds = conds[:n]
    # fixed noise per sample, so each checkpoint eval is deterministic
    noise = torch.randn(len(latents), 4, 32, 32)
    return {
        "latents": latents,                    # (n,4,32,32)
        "conds": conds,                          # list of (yc, ys, yh)
        "gts": gts,                            # (n,3,256,256) [-1,1]
        "noise": noise,
        "win": _gaussian_window(11, 1.5, "cpu"),
    }


def eval_in_memory(model, vae, diffusion, device, cache, n=1000, t=T_EVAL,
                   vis_out=None, vis_n=5, batch_size=8):
    """
    Evaluate the *current* model weights (already on GPU) on the cached test set.
    Returns (mse, ssim) floats. Does not modify model training state.

    If `vis_out` is given, the first `vis_n` reconstructed images (pred | GT
    side-by-side) are saved as a single comparison sheet to `vis_out`.
    """
    win = cache["win"].to(device)
    conds = cache["conds"][:n]

    mse_sum, ssim_sum, cnt = 0.0, 0.0, 0
    decoded_list, gt_list = [], []
    t_tensor = torch.full((1,), t, device=device, dtype=torch.long)
    with torch.no_grad():
        for i in range(0, n, batch_size):
            j = min(i + batch_size, n)
            x_lat = cache["latents"][i:j].to(device)
            nz = cache["noise"][i:j].to(device)
            gt = cache["gts"][i:j].to(device)
            yc = torch.tensor([c[0] for c in conds[i:j]], device=device, dtype=torch.long)
            ys = torch.tensor([c[1] for c in conds[i:j]], device=device, dtype=torch.long)
            yh = torch.tensor([c[2] for c in conds[i:j]], device=device, dtype=torch.long)
            mk = dict(y_callig=yc, y_script=ys, y_char=yh)
            # t broadcast to batch size
            tt = t_tensor.expand(j - i)
            ld = diffusion.training_losses(model, x_lat, tt, mk, noise=nz)
            pred = ld["pred_xstart"]                       # (B,4,32,32)
            decoded = vae.decode(pred / 0.18215).sample    # (B,3,256,256) [-1,1]
            mse_sum += F.mse_loss(decoded, gt).item() * (j - i)
            # SSIM must be per-image (batch pooling would mix images); loop is cheap here.
            d01 = (decoded + 1) / 2
            g01 = (gt + 1) / 2
            for _k in range(decoded.shape[0]):
                ssim_sum += _ssim(d01[_k:_k + 1], g01[_k:_k + 1], 1.0, 11, win)
            cnt += (j - i)
            if vis_out:
                # 收集全部预测（用于逐张落盘）
                for _k in range(decoded.shape[0]):
                    decoded_list.append(decoded[_k:_k + 1].detach().cpu())
                    gt_list.append(gt[_k:_k + 1].detach().cpu())
    if vis_out and decoded_list:
        if vis_out.lower().endswith((".png", ".jpg")):
            # 兼容旧的单文件拼图模式
            _save_eval_visuals(decoded_list, gt_list, conds, vis_out, vis_n)
        else:
            # 目录模式：每一张 (pred|GT) 对比图逐张落盘（留档全部 1000 张）
            _dump_eval_all(decoded_list, gt_list, vis_out, vis_n=0)
            # 同时写前 5 张拼图供 dashboard 展示（覆盖式）
            _save_eval_visuals(decoded_list, gt_list, conds,
                               os.path.join(os.path.dirname(vis_out), "eval_latest.png"), 5)
    del win
    del decoded_list, gt_list
    torch.cuda.empty_cache()
    return mse_sum / cnt, ssim_sum / cnt
