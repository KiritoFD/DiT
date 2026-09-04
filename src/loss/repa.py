# -*- coding: utf-8 -*-
"""
repa.py — 公共 REPA (Representation Alignment) infra。

统一预训练(train.py) 与后训练(train_controlnet.py / train_repa.py) 的 REPA 用法:
  * 共享一个 DINOv2 teacher (冻结), 每层一个 REPALoss (投影头独立可训练)
  * 多层列表 (layers=[8,11]) 或单层 (layers=[8])
  * warmup 渐进权重 (w_repa * min(1, step/warmup_steps))
  * 一次 teacher 前向, 多层学生特征共享 (REPALoss 已有 list 支持)

用法 (两个阶段一致):
    from src.loss.repa import build_repa_module
    repa = build_repa_module(student_dim=384, layers=(8, 11),
                             teacher_ckpt='pretrained_models/dinov2_vits14_pretrain.safetensors',
                             w_repa=0.3, warmup_steps=2000)
    # 训练循环:
    model.return_intermediate_layers = repa.layers   # 或 forward 传参
    ...
    loss = repa(intermediate_feats_dict_or_list, img, step)
    total = base_loss + loss
    # 优化器:
    trainable += repa.trainable_params()
"""
import os
import torch
import torch.nn as nn

from .losses import REPALoss


class RepaModule(nn.Module):
    """统一 REPA 挂载: 多层对齐 + 共享 teacher + warmup 渐进。"""

    def __init__(self, student_dim, layers=(8,), teacher_ckpt=None,
                 teacher_backbone="dinov2_vits14", w_repa=0.1,
                 warmup_steps=0, device=None):
        super().__init__()
        if isinstance(layers, int):
            layers = (layers,)
        self.layers = tuple(int(l) for l in layers)
        self.w_repa = float(w_repa)
        self.warmup_steps = int(warmup_steps)

        # 共享 teacher: 第一个 REPALoss 加载 teacher (缓存), 其余复用同对象
        # (REPALoss 内部 has 共享 teacher 的 _TeacherWrapper 复用机制)
        self._teacher = None
        self.losses = nn.ModuleList()
        for l in self.layers:
            kw = dict(student_dim=student_dim,
                      teacher_backbone=teacher_backbone,
                      teacher_ckpt=teacher_ckpt)
            if self._teacher is not None:
                kw["teacher"] = self._teacher
            rl = REPALoss(**kw)
            # 保存 teacher 引用供后续层复用 (REPALoss 里 teacher 已 wrapper)
            self._teacher = rl.teacher
            self.losses.append(rl)
            print(f"[repa] layer {l}: REPALoss (student_dim={student_dim}, "
                  f"teacher={rl.is_vits14 and 'vits14' or 'dino'})")

        self._eval_teacher = None  # eval 用
        if device is not None:
            self.to(device)

    def forward(self, intermediate_feats, img, step=0, w_override=None):
        """计算 REPA loss (w 渐进)。

        intermediate_feats: 与 self.layers 对应的特征:
            * dict {layer: feats} (模型新接口 return_intermediate_layers=dict)
            * list/tuple 长度 == len(layers)
            * 单张量 (len(layers)==1)
        img: (B,3,H,W) GT 图 [-1,1]
        返回: w_eff * loss (标量张量), 当 img 为 None 或缺少特征时返回 0
        """
        if img is None:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        # 收集学生特征
        if isinstance(intermediate_feats, dict):
            feats = [intermediate_feats[l] for l in self.layers if l in intermediate_feats]
            if len(feats) != len(self.layers):
                # 缺层: 用存在的 (不报错, 与旧行为兼容)
                feats = [intermediate_feats[l] for l in self.layers
                         if l in intermediate_feats] or None
        elif isinstance(intermediate_feats, (list, tuple)):
            feats = list(intermediate_feats)
        else:
            feats = intermediate_feats if intermediate_feats is not None else None
        if feats is None or (isinstance(feats, list) and not feats):
            return torch.tensor(0.0, device=next(self.parameters()).device)

        w = float(w_override if w_override is not None else self.w_repa)
        if self.warmup_steps > 0:
            w = w * min(1.0, float(step) / float(self.warmup_steps))
        if w <= 0:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        # REPALoss.forward 支持 list (多层取平均共享一次 teacher)
        loss = self.losses[0](feats, img) if len(self.losses) == 1 else \
            sum(l(f, img) for l, f in zip(self.losses, feats)) / len(self.losses)
        return w * loss

    def trainable_params(self):
        """投影头参数 (可训练), 供 optimizer 合并。"""
        out = []
        for l in self.losses:
            out.extend(p for p in l.proj.parameters() if p.requires_grad)
        return out

    def to_ema_copy(self, ema_model):
        """返回一个与主模型结构一致的 eval-only 副本（无梯度）。"""
        import copy
        m = copy.deepcopy(self).eval()
        for p in m.parameters():
            p.requires_grad = False
        return m


def build_repa_module(student_dim, layers=(8,), teacher_ckpt=None,
                      teacher_backbone="dinov2_vits14", w_repa=0.1,
                      warmup_steps=0, device=None):
    """工厂: 从 config 参数构建 RepaModule (nil-safe, w_repa<=0 返回 None)."""
    if float(w_repa) <= 0:
        return None
    return RepaModule(student_dim=student_dim, layers=layers,
                      teacher_ckpt=teacher_ckpt,
                      teacher_backbone=teacher_backbone,
                      w_repa=w_repa, warmup_steps=warmup_steps,
                      device=device)