# Modified from OpenAI's diffusion repos
#     GLIDE: https://github.com/openai/glide-text2im/blob/main/glide_text2im/gaussian_diffusion.py
#     ADM:   https://github.com/openai/guided-diffusion/blob/main/guided_diffusion
#     IDDPM: https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py

from . import gaussian_diffusion as gd
from .respace import SpacedDiffusion, space_timesteps
from .flow_matching import (
    create_flow_matching, FlowMatching, FLOW_PARAMS, FLOW_PARAM_ALIASES,
)
from .losses import (
    StructDecoder, LatentStructLoss, EdgeGradientLoss, SkeletonLoss, REPALoss,
)


def create_diffusion(
    timestep_respacing,
    noise_schedule="linear", 
    use_kl=False,
    sigma_small=False,
    predict_xstart=False,
    learn_sigma=True,
    rescale_learned_sigmas=False,
    diffusion_steps=1000
):
    betas = gd.get_named_beta_schedule(noise_schedule, diffusion_steps)
    if use_kl:
        loss_type = gd.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = gd.LossType.RESCALED_MSE
    else:
        loss_type = gd.LossType.MSE
    if timestep_respacing is None or timestep_respacing == "":
        timestep_respacing = [diffusion_steps]
    return SpacedDiffusion(
        use_timesteps=space_timesteps(diffusion_steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type
        # rescale_timesteps=rescale_timesteps,
    )


def create_diffusion_or_flow(
    timestep_respacing,
    diffusion_type="ddpm",
    flow_steps=None,
    **kwargs,
):
    """Factory with a DDPM / Flow-Matching switch.

    ``diffusion_type`` in {"ddpm", "flow", "flow_matching"}:
        * "ddpm": standard GaussianDiffusion (epsilon prediction, DDIM sampling).
        * "flow"/"flow_matching": linear-interpolant FlowMatching (velocity
          prediction, Euler ODE sampling).  ``flow_steps`` overrides the Euler
          step count when given; otherwise ``timestep_respacing`` is used.
    """
    diffusion_type = (diffusion_type or "ddpm").lower()
    if diffusion_type in ("flow", "flow_matching", "fm"):
        steps = flow_steps if flow_steps else timestep_respacing
        # create_flow_matching 会自行过滤掉非 flow 参数，因此这里可以整包转发。
        return create_flow_matching(steps, **kwargs)
    return create_diffusion(timestep_respacing, **kwargs)


def flow_kwargs_from(args):
    """从 argparse Namespace（或 dict / 任意对象）里抽取 flow 相关配置。

    目的：让 train.py / eval 各处构造 diffusion 时拿到**同一份** flow 配置。
    过去 eval 侧只传 ``timestep_respacing``，导致训练用 logit-normal + Heun、
    推理却退回默认参数这种静默不一致。

    只返回显式存在且不为 None 的键，缺失项由 FlowMatching 的默认值兜底。
    """
    get = args.get if isinstance(args, dict) else (lambda k, d=None: getattr(args, k, d))
    out = {}
    for k in FLOW_PARAMS:
        if k in ("num_steps", "sigma_min"):
            continue  # 步数由调用方按场景决定（train vs eval），不在这里统一
        v = get(k, None)
        if v is not None:
            out[k] = v
    # 别名：CLI/config 里用 flow_sampler，避免和数据采样器 --sampler 撞 dest。
    for alias, real in FLOW_PARAM_ALIASES.items():
        v = get(alias, None)
        if v is not None:
            out[real] = v
    return out


__all__ = [
    "create_diffusion", "create_diffusion_or_flow", "create_flow_matching",
    "flow_kwargs_from", "FLOW_PARAMS", "FLOW_PARAM_ALIASES",
    "SpacedDiffusion", "space_timesteps", "FlowMatching",
    "StructDecoder", "LatentStructLoss", "EdgeGradientLoss", "SkeletonLoss", "REPALoss",
]

