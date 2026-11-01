"""Continuous reconstruction objectives; no skeleton counts enter training."""
import torch
import torch.nn.functional as F


def edge_loss(pred, target):
    k = pred.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) / 8
    kernels = torch.stack((k, k.t()))[:, None]
    total = pred.new_zeros(())
    for scale in (1, 2, 4):
        p, t = pred.mean(1, keepdim=True), target.mean(1, keepdim=True)
        if scale > 1:
            p, t = F.avg_pool2d(p, scale), F.avg_pool2d(t, scale)
        total = total + F.l1_loss(F.conv2d(p, kernels, padding=1),
                                  F.conv2d(t, kernels, padding=1)) / 3
    return total


def gradient_norm(loss, parameters):
    grads = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    return torch.sqrt(sum(g.detach().float().square().sum() for g in grads if g is not None))


def reconstruction(pred, target, perceptual, parameters, edge=False, probe=False):
    l1 = F.l1_loss(pred, target)
    lp = perceptual(pred, target).mean()
    base = l1 + lp  # physical VAE recipe: L1=1, LPIPS(VGG)=1
    stats = dict(l1=float(l1.detach()), lpips=float(lp.detach()))
    # V1 cap is measured on ALL decoder parameters, every microbatch; not an
    # output-image gradient proxy or a stale last-layer coefficient.
    if edge or probe:
        nb = gradient_norm(base, parameters)
        stats['base_grad_norm'] = float(nb)
        if probe:
            stats['l1_grad_norm'] = float(gradient_norm(l1, parameters))
            stats['lpips_grad_norm'] = float(gradient_norm(lp, parameters))
    if edge:
        aux = edge_loss(pred, target)
        ne = gradient_norm(aux, parameters)
        weight = (.2 * nb / (ne + 1e-12)).clamp(max=1.).detach()
        stats.update(edge=float(aux.detach()), edge_weight=float(weight),
                     edge_grad_ratio=float(weight * ne / (nb + 1e-12)))
        base = base + weight * aux
    return base, stats
