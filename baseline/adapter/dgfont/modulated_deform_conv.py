# -*- coding: utf-8 -*-
"""Modulated deformable conv v2 implemented with torchvision.ops.deform_conv2d.

Drop-in replacement for the original CUDA-extension version (functions/
modulated_deform_conv_func.py + src/cuda/*), which cannot compile against
modern PyTorch. Numerically equivalent for the settings DG-Font uses
(groups=1, deformable_groups=1). Exports the same class names the repo
imports (modules/__init__.py: ModulatedDeformConv, _ModulatedDeformConv,
ModulatedDeformConvPack).
"""
import math

import torch
import torch.nn as nn
import torchvision.ops as tvops


class _ModulatedDeformConv(nn.Module):
    """Functional-style wrapper kept for import compatibility."""


class ModulatedDeformConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, deformable_groups=1,
                 im2col_step=64, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = nn.modules.utils._pair(kernel_size)
        self.stride = nn.modules.utils._pair(stride)
        self.padding = nn.modules.utils._pair(padding)
        self.dilation = nn.modules.utils._pair(dilation)
        self.groups = groups
        self.deformable_groups = deformable_groups
        self.im2col_step = im2col_step
        self.use_bias = bias

        self.weight = nn.Parameter(torch.Tensor(
            out_channels, in_channels // groups, *self.kernel_size))
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        self.reset_parameters()
        if not self.use_bias:
            self.bias.requires_grad = False

    def reset_parameters(self):
        n = self.in_channels
        init = nn.init
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input, offset, mask):
        return tvops.deform_conv2d(input, offset, self.weight, self.bias,
                                   stride=self.stride, padding=self.padding,
                                   dilation=self.dilation, mask=mask)


class ModulatedDeformConvPack(ModulatedDeformConv):
    """Offset/mask are predicted from `input_offset`; conv applies to `input_real`.

    Matches the original DG-Font call convention:
        out, offset = ModulatedDeformConvPack(...)(input_offset, input_real)
    where input_offset is the (possibly 2x-channel) guidance tensor and
    input_real the tensor being convolved.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, deformable_groups=1,
                 double=False, im2col_step=64, bias=True, lr_mult=0.1):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding,
                         dilation, groups, deformable_groups, im2col_step, bias)
        om_channels = self.deformable_groups * 3 * self.kernel_size[0] * self.kernel_size[1]
        if not double:
            self.conv_offset_mask = nn.Conv2d(
                self.in_channels, om_channels, kernel_size=self.kernel_size,
                stride=self.stride, padding=self.padding, bias=True)
        else:
            self.conv_offset_mask = nn.Conv2d(
                self.in_channels * 2, om_channels, kernel_size=self.kernel_size,
                stride=self.stride, padding=self.padding, bias=True)
        self.conv_offset_mask.lr_mult = lr_mult
        self.init_offset()

    def init_offset(self):
        self.conv_offset_mask.weight.data.zero_()
        self.conv_offset_mask.bias.data.zero_()

    def forward(self, input_offset, input_real):
        out = self.conv_offset_mask(input_offset)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        out = tvops.deform_conv2d(input_real, offset, self.weight, self.bias,
                                  stride=self.stride, padding=self.padding,
                                  dilation=self.dilation, mask=mask)
        return out, offset
