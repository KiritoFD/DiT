"""Read-only conditioning diagnostics, kept out of the optimization loop logic."""
import torch


def log_writer_glyph(model, logger, step: int) -> None:
    """Inspect existing gradients; never launch a new forward/backward pass."""
    for attribute in ('module', '_orig_mod'):
        model = getattr(model, attribute, model)
    branch = getattr(model, 'writer_glyph', None)
    if branch is None:
        return
    with torch.no_grad():
        p_grad = branch.film.P.weight.grad
        q_grad = branch.film.Q.weight.grad
        logger.info('[writer-glyph-step] step=%d gains=%s P_grad=%.4e Q_grad=%.4e',
                    step, {key: float(gain.detach()) for key, gain in branch.gains.items()},
                    float(p_grad.norm()) if p_grad is not None else 0.,
                    float(q_grad.norm()) if q_grad is not None else 0.)
