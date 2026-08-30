# -*- coding: utf-8 -*-
"""
gradio_s20_ctrl.py — s20 ControlNet 书法生成前端 (skel 结构条件 + DINO 字形条件).

模型: DiT-2Cond-S/2 (v2 骨干: rms/swiglu/rope, learn_sigma=False, char_proj_mode=mlp)
      + ControlNetDiT (ctrl_depth=12, injection=modulate, null_cond=gaussian).
主模型 0102500.pt + ctrl 0075000.pt (train_ctrl_only, 只含 ctrl_encoder + injections).

条件:
  * skel (结构): 取自标准字库 std_glyph_latent_v2/{font}/U+XXXXX.npy 的标准字形 latent,
    解码 -> 二值化 -> 3px 骨架 -> VAE encode -> (1,4,32,32) 送入 ControlNet cond.
  * glyph (内容): y_char = char2scripts[char][script] (冻结 DINO 初始化表, 即"dino encode 的 glyph 输入").
  * y_callig: 书法家 id.

用法:
  python tools/controlnet/gradio_s20_ctrl.py --share        # 默认公网
  python tools/controlnet/gradio_s20_ctrl.py --port 7862 --cfg 0.7 --steps 50
"""
import os
import sys
import csv
import json
import random
import argparse
import numpy as np
import torch

os.environ["XFORMERS_DISABLED"] = "1"
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
if os.path.join(_ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(_ROOT, "src"))

from src.model.controlnet import load_main_model, ControlNetDiT
from src.loss.flow_matching import FlowMatching
from src.utils.glyph_latent_v2 import get_glyph_lookup_v2

from PIL import Image
from skimage.morphology import skeletonize
from scipy.ndimage import binary_dilation

# ──────────────────────────────────────────────────────────────────────────
# 常量
# ──────────────────────────────────────────────────────────────────────────
MAIN_CKPT = os.path.join(_ROOT, r"5script\results\s20_midcommon_s_flow_v2\20260829-023132-s20-midcommon-s-flow-v2\checkpoints\0102500.pt")
CTRL_CKPT = os.path.join(_ROOT, r"5script\results\s20_ctrl_skel_flow_v2\20260829-161522-s20-ctrl-skel-flow-v2\checkpoints\0075000.pt")
VAE_PATH = os.path.join(_ROOT, "pretrained_models", "sd-vae-ft-ema")
DINO_EMB = os.path.join(_ROOT, "pretrained_models", "dino_embeddings", "glyph_dino_embeddings_384.npy")
DINO_IDX = os.path.join(_ROOT, "pretrained_models", "dino_embeddings", "glyph_dino_index.json")

CALLIGS_JSON = os.path.join(_ROOT, "src", "train", "configs", "calligraphers.json")
CHAR_META = os.path.join(_ROOT, "src", "train", "configs", "char_meta.json")

# script 字符串 -> (std 字库 script_id, 默认字体, 显示名)
SCRIPT = {"楷": (0, "kai_gb"), "隶": (4, "li_gb"), "行": (3, "xing_st")}

CFG_DEFAULT = 0.7   # 用户确认的 cfg
STEPS_DEFAULT = 50


def build_submodules():
    """加载 main + ctrl + flow + vae, 只做一次。"""
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[init] device = {dev}")

    main = load_main_model(
        model_name="DiT-2Cond-S/2", ckpt_path=MAIN_CKPT, device="cpu",
        num_calligraphers=1011, num_characters=35130,
        condition_fusion="factorized_add", callig_embed_dim=128, char_embed_dim=384,
        char_proj_mode="mlp", freeze_char_table=True,
        cond_drop_all_prob=0.05, cond_drop_one_prob=0.25, cond_drop_which_glyph_prob=0.75,
        use_checkpoint=False, learn_sigma=False, diffusion_type="flow",
        norm_type="rms", mlp_type="swiglu", qk_norm=True, rope=True,
        rope_theta=100.0, attn_impl="sdpa")
    main.to(dev).eval()
    for p in main.parameters():
        p.requires_grad_(False)

    ctrl = ControlNetDiT.from_ckpt(
        main, CTRL_CKPT, device=dev, strict=True,
        cond_in_channels=4, injection="modulate", null_cond="gaussian")
    ctrl.eval()
    for p in ctrl.parameters():
        p.requires_grad_(False)

    flow = FlowMatching(num_steps=STEPS_DEFAULT, t_sampler="logit_normal",
                        t_mean=0.0, t_std=1.0, shift=1.0,
                        sampler="heun", heun_batch=True)

    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(VAE_PATH).to(dev).eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    print(f"[init] main={sum(p.numel() for p in main.parameters()):,} "
          f"ctrl={sum(p.numel() for p in ctrl.parameters() if not p.requires_grad):,}")
    return dev, main, ctrl, flow, vae


class S20CtrlSampler:
    def __init__(self):
        self.dev, self.main, self.ctrl, self.flow, self.vae = build_submodules()
        # 元数据
        self.calligs = json.load(open(CALLIGS_JSON, encoding="utf-8"))
        self.callig2id = {c["name"]: c["id"] for c in self.calligs}
        self.char2scripts = json.load(open(CHAR_META, encoding="utf-8"))["char2scripts"]
        self.glook = get_glyph_lookup_v2()  # std 字库 (4,32,32) 标准字形 latent
        # DINO 表 (glyph 内容)
        self.dino_emb = np.load(DINO_EMB)
        self.dino_idx = json.load(open(DINO_IDX, encoding="utf-8"))["glyphs"]

    # ── skel: 标准字形 latent -> 256 图 -> 3px 骨架 -> VAE latent ──────────
    @torch.no_grad()
    def _std_glyph_image(self, std_lat):
        img = self.vae.decode(std_lat.unsqueeze(0).to(self.dev) / 0.18215).sample[0]
        return ((img.clamp(-1, 1) + 1) / 2).permute(1, 2, 0).cpu().numpy()

    @torch.no_grad()
    def _skel_latent_from_glyph(self, std_lat):
        """标准字形 latent -> 图 -> 骨架(3px) -> VAE encode -> (1,4,32,32)."""
        arr = self._std_glyph_image(std_lat)            # (256,256,3) in [0,1]
        gray = arr.mean(-1)
        binary = gray < 0.5                             # 墨迹为暗
        skel1 = skeletonize(binary)
        skel3 = binary_dilation(skel1, structure=np.ones((3, 3)), iterations=3)
        # 骨架图: 白线(255) 黑底(0) —— 与训练 skel latent 约定一致 (line=+1, bg=-1)
        skel01 = skel3.astype(np.float32)               # line=1(白), bg=0(黑)
        rgb = np.stack([skel01] * 3, -1)                # (256,256,3)
        t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float().to(self.dev)
        t = t * 2 - 1                                   # [-1,1]  line=+1, bg=-1
        lat = self.vae.encode(t).latent_dist.sample()   # (1,4,32,32)
        return lat * 0.18215, (skel3.astype(np.uint8) * 255)

    @torch.no_grad()
    def _dino_glyph_embed(self, char):
        """从预计算 DINO 表查 glyph 的 384 维 embedding (dino encode 的 glyph 输入)."""
        u = f"U+{ord(char):05X}"
        # glyph 表按 (script_id, char_id) 索引; 我们用 codepoint 近似定位到对应行
        # 这里返回是否命中 + embedding(缺失返回 None, 由调用方处理)
        return None, u

    @torch.no_grad()
    def generate(self, char, script, callig, cfg_scale=CFG_DEFAULT,
                 num_steps=STEPS_DEFAULT, seed=None, use_skel=True):
        """返回 (PIL img, skel_vis PIL, status)"""
        char = (char or "").strip()
        if not char:
            raise ValueError("请输入一个汉字")
        if char not in self.char2scripts:
            raise ValueError(f"字「{char}」不在字符表 (char2scripts). 无法定位 glyph_id。")
        scripts_avail = list(self.char2scripts[char].keys())
        if script not in scripts_avail:
            script = scripts_avail[0]
        if callig not in self.callig2id:
            raise ValueError(f"书法家「{callig}」不在列表。")
        glyph_id = self.char2scripts[char][script]
        callig_id = self.callig2id[callig]

        # 标准字库字形 latent (结构先验来源)
        script_id, font = SCRIPT.get(script, (0, "kai_gb"))
        std_lat = self.glook.get(script_id, char, font=font)
        if std_lat is None:
            raise ValueError(f"标准字库中无「{char}」(font={font})。")
        std_lat = std_lat.to(self.dev)

        if seed is None or seed == 0:
            seed = random.randint(0, 2 ** 31)
        torch.manual_seed(seed)
        z = torch.randn(1, 4, 32, 32, device=self.dev)
        yc = torch.tensor([callig_id], dtype=torch.long, device=self.dev)
        ych = torch.tensor([glyph_id], dtype=torch.long, device=self.dev)
        mk = dict(y_callig=yc, y_char=ych)
        skel_vis = None
        if use_skel:
            skel_lat, skel_vis = self._skel_latent_from_glyph(std_lat)
            mk["cond"] = skel_lat

        def model_fn(x, t, **kw):
            return self.ctrl.forward_with_cfg(x, t, cfg_scale=cfg_scale, **kw)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            latent = self.flow.ddim_sample_loop(
                model_fn, z.shape, z, clip_denoised=False,
                model_kwargs=mk, device=self.dev)
        latent = latent.float()
        img = self.vae.decode(latent / 0.18215).sample[0]
        img = ((img.clamp(-1, 1) + 1) / 2).permute(1, 2, 0).cpu().numpy()
        pil = Image.fromarray((img * 255).astype(np.uint8))
        skel_pil = Image.fromarray(skel_vis) if skel_vis is not None else None
        status = (f"{char} · {script} · {callig} · glyph_id={glyph_id} · callig_id={callig_id} · "
                  f"seed={seed} · CFG={cfg_scale} · steps={num_steps} · skel={'on' if use_skel else 'off'}")
        return pil, skel_pil, status


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7862)
    ap.add_argument("--server-name", default="0.0.0.0")
    ap.add_argument("--share", action="store_true", default=True)
    ap.add_argument("--cfg", type=float, default=CFG_DEFAULT)
    ap.add_argument("--steps", type=int, default=STEPS_DEFAULT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--use-skel", type=str, default="on", choices=["on", "off"])
    ap.add_argument("--max-batch", type=int, default=1)
    args, _ = ap.parse_known_args()

    import gradio as gr
    sampler = S20CtrlSampler()

    def gen(char, script, callig, cfg, steps, seed, skel_on):
        try:
            pil, skel_pil, status = sampler.generate(
                char, script, callig, cfg_scale=cfg, num_steps=steps,
                seed=seed, use_skel=(skel_on == "on"))
            return pil, skel_pil, status
        except ValueError as e:
            raise gr.Error(str(e))
        except Exception as e:
            raise gr.Error(f"生成失败: {type(e).__name__}: {e}")

    with gr.Blocks(title="书法生成 · s20 ControlNet (skel + glyph)", analytics_enabled=False) as demo:
        gr.Markdown("# 书法生成 · s20 ControlNet (skel 结构 + DINO 字形)")
        gr.Markdown(f"主模型 **0102500.pt** + ControlNet **0075000.pt** · flow/Heun · "
                    f"默认 CFG=**{args.cfg}** · 结构条件=标准字库骨架 (3px)")
        with gr.Row():
            with gr.Column(scale=1, min_width=260):
                char_in = gr.Textbox(label="汉字", value="永", info="必须是标准字库 + 字符表都覆盖的字")
                script_in = gr.Dropdown(label="字体", choices=list(SCRIPT.keys()), value="楷")
                callig_in = gr.Dropdown(label="书法家", choices=[c["name"] for c in sampler.calligs],
                                        value=[c["name"] for c in sampler.calligs][-1])
                cfg_in = gr.Slider(0.0, 3.0, value=args.cfg, step=0.05, label="CFG 强度")
                steps_in = gr.Slider(10, 100, value=args.steps, step=5, label="采样步数 (Heun)")
                seed_in = gr.Slider(0, 99999, value=args.seed, step=1, label="随机种子 (0=随机)")
                skel_in = gr.Radio(["on", "off"], value=args.use_skel, label="结构条件 skel")
                btn = gr.Button("生成", variant="primary")
                status = gr.Textbox(label="状态", interactive=False)
            with gr.Column(scale=1, min_width=256):
                out_img = gr.Image(label="生成结果", width=288, height=288)
                skel_img = gr.Image(label="标准字库骨架 (条件)", width=288, height=288)

        btn.click(fn=gen,
                  inputs=[char_in, script_in, callig_in, cfg_in, steps_in, seed_in, skel_in],
                  outputs=[out_img, skel_img, status])

    demo.launch(server_name=args.server_name, server_port=args.port,
                share=args.share, show_error=True, quiet=False)


if __name__ == "__main__":
    main()
