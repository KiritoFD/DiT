# -*- coding: utf-8 -*-
"""Backward-compat shim: 扩散/loss 代码已迁移至 src.loss. 用法不变:

    from diffusion import create_diffusion_or_flow, FlowMatching, SpacedDiffusion, ...
"""
from src.loss import *  # noqa: F401,F403
# 保持子模块可访问 (from diffusion.gaussian_diffusion import ...)
from src.loss import gaussian_diffusion
from src.loss import flow_matching
from src.loss import respace
from src.loss import timestep_sampler
from src.loss import diffusion_utils
from src.loss import losses  # noqa: F401