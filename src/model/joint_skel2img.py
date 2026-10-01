# -*- coding: utf-8 -*-
"""joint_skel2img.py — v30+: 骨架生成器(stage1) + 图像主干(stage2) 的**统一模型包装**.

把两段模型构造成**一个模型**: forward(x, t, y_callig, y_char, g=std_skel) 内部
  1. stage1 条件式采样 (噪声起步 + std 骨架当条件, Euler k 步) -> g_pred
  2. stage2 以 g_pred 为骨架条件输出图像 latent 的速度场
对外接口与 DiT_2Cond 完全一致 (含 forward_with_cfg), in_mem_eval / sample_latents
零改动可用 —— "我们的模型"就是一个模型。

训练侧 (train_joint_stage1_stage2.py) 仍直接用 stage1/stage2 两个裸模块 (梯度控制
更细: 3a 冻结 stage2 / 3b 小 lr); 本包装用于 **eval 与推理**, 或作为未来端到端
训练的基座。与 ckpt 的关系:
  * from_ckpts(): 从 stage1/stage2 两个训练 ckpt 构造 (model_io 严格复刻)
  * save()/load(): 单文件 {"stage1","stage2","args"} —— v35 训练 ckpt 的
    {"gen","gen_ema","bak","bak_ema"} 也可直接 load_v35()
"""
import torch as th
import numpy as np

TIME_SCALE = 1000.0


class JointSkel2Img(th.nn.Module):
    """stage1(骨架生成) + stage2(图像扩散) 的统一可调用模型.

    forward(x, t, y_callig, y_char, g=std_skel):
      x      : (B,4,32,32) 图像 latent 噪声/中间态
      g      : (B,4,32,32) 标准骨架 latent (stage1 的条件)
      返回   : stage2 的速度场 (B,4,32,32)
    内部先做 stage1 的 k 步 Euler 采样得到 g_pred (k=self.gen_steps).
    """

    use_glyph_cond = True          # 让 sample_latents 的 g 路由选中本模型
    use_char_cond = False

    def __init__(self, stage1, stage2, gen_steps=50, scale_gen_grad=False):
        super().__init__()
        self.stage1 = stage1
        self.stage2 = stage2
        self.gen_steps = int(gen_steps)
        self.scale_gen_grad = bool(scale_gen_grad)

    # ---- stage1 采样 (默认 no_grad; 训练侧传 grad=True) ----
    def gen_sample(self, g_std, y, y_char=None, steps=None, grad=False):
        """stage1 条件式采样. y_char 必须传**真实 glyph_id** —— stage1 (v33) 是
        train.py 训的带字条件模型, 传 zeros=错误字条件 (strict 未见组合上会崩)."""
        steps = self.gen_steps if steps is None else steps
        b = g_std.shape[0]
        dev = g_std.device
        z = th.randn(b, g_std.shape[1], g_std.shape[2], g_std.shape[3], device=dev)
        ts = th.linspace(1.0, 0.0, steps + 1, device=dev)
        yc = y_char if y_char is not None else th.zeros(b, dtype=th.long, device=dev)
        ctx = th.enable_grad() if grad else th.no_grad()
        with ctx:
            for k in range(steps):
                t = th.full((b,), float(ts[k]) * TIME_SCALE, device=dev)
                v = self.stage1(z, t, y_callig=y, y_char=yc, g=g_std)
                if isinstance(v, tuple):
                    v = v[0]
                z = z + (ts[k + 1] - ts[k]) * v
        return z

    def forward(self, x, t, y_callig, y_char, g=None, **kw):
        if g is None:
            return self.stage2(x, t, y_callig, y_char, **kw)
        g_pred = self.gen_sample(g, y_callig, y_char=y_char,
                                 grad=self.scale_gen_grad and self.training)
        return self.stage2(x, t, y_callig=y_callig, y_char=y_char, g=g_pred, **kw)

    def forward_with_cfg(self, x, t, y_callig, y_char, cfg_scale=1.0, g=None, **kw):
        """CFG 透传 stage2: g_pred 只采一次 (cfg 两半共享, 保证条件一致)."""
        if g is None or not cfg_scale or cfg_scale <= 0:
            return self.forward(x, t, y_callig, y_char, g=g, **kw)
        g_pred = self.gen_sample(g, y_callig, y_char=y_char)
        return self.stage2.forward_with_cfg(
            x, t, y_callig, y_char, cfg_scale=cfg_scale, g=g_pred, **kw)

    # ---- 单文件 ckpt ----
    def save(self, path, args=None):
        th.save({"stage1": self.stage1.state_dict(),
                 "stage2": self.stage2.state_dict(),
                 "args": args or {"gen_steps": self.gen_steps}}, path)

    def load(self, path):
        d = th.load(path, map_location="cpu", weights_only=False)
        self.stage1.load_state_dict(_strip(d["stage1"]))
        self.stage2.load_state_dict(_strip(d["stage2"]))
        if isinstance(d.get("args"), dict) and "gen_steps" in d["args"]:
            self.gen_steps = int(d["args"]["gen_steps"])

    @classmethod
    def from_ckpts(cls, gen_ckpt, bak_ckpt, gen_steps=50, device="cuda"):
        """从 stage1/stage2 的训练 ckpt 构造 (model_io 严格复刻 train.py 构造)."""
        from src.eval import model_io
        s1, _ = model_io.load_model_from_ckpt(gen_ckpt, device=device, use_ema=True)
        s2, _ = model_io.load_model_from_ckpt(bak_ckpt, device=device, use_ema=True)
        m = cls(s1, s2, gen_steps=gen_steps)
        return m.to(device).eval()

    def load_v35(self, path, device="cuda"):
        """加载 train_joint_stage1_stage2.py 的联训 ckpt {"gen","gen_ema","bak","bak_ema"}."""
        d = th.load(path, map_location="cpu", weights_only=False)
        self.stage1.load_state_dict(_strip(d.get("gen_ema") or d["gen"]))
        self.stage2.load_state_dict(_strip(d.get("bak_ema") or d["bak"]))
        self.to(device).eval()


def _strip(sd):
    return {(k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
            for k, v in sd.items()}
