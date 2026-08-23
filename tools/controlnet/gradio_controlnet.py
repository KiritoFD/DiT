# -*- coding: utf-8 -*-
"""
gradio_controlnet.py — 书法生成前端 (top6 195k DiT-2Cond-S/2).

功能:
  - 书法家下拉 (top6 的 11 位)
  - 字体下拉 (楷 / 隶 — top6 训练集只有这两种)
  - 字输入 (支持 3154 字, supported_chars.txt)
  - 用 195k diff-only ckpt 的 DiT-2Cond 生成
  - 找条件相近的 GT 从本地 MCCD 加载, 摆在旁边对照
  - 推理结果累积在页面底部 gallery

模型条件:
  - y_callig: 书家 ID (1011 类, factorized_add)
  - y_char:   glyph_id (= char+script 联合编码, 35130 类)
  - 训练时 glyph_id = f(character, script), 同字不同体有不同 glyph_id

用法:
  python tools/controlnet/gradio_controlnet.py \
      --main-ckpt 5script/results/s6_top6_diffonly/20260820-191536-s6-top6-diffonly-resume/checkpoints/0195000.pt
"""
import os
os.environ["XFORMERS_DISABLED"] = "1"
import sys
_s = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src")
if _s not in sys.path:
    sys.path.insert(0, _s)
import csv
import json
import random
import argparse
import numpy as np
import torch
from PIL import Image

from models import DiT_2Cond_models
from diffusion import create_diffusion

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding="utf-8")


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
MANIFEST_PATH = os.path.join(ROOT, "final_manifest.json")
CSV_PATH = os.path.join(ROOT, "5script", "train_top6.csv")
MCCD_CHAR_DIR = os.path.join(ROOT, "MCCD", "MCCD", "MCCD_Character", "character_dataset")
CALLIGS_JSON = os.path.join(HERE, "calligraphers.json")
CHAR_META = os.path.join(HERE, "char_meta.json")


class CalligraphySampler:
    """持有模型 + 映射，提供 generate() 和 find_gt()。模型只加载一次。"""

    def __init__(self, main_ckpt, vae_path, device="cuda",
                 num_calligraphers=1011, num_characters=35130,
                 condition_fusion="factorized_add", callig_embed_dim=128,
                 char_embed_dim=256, cond_drop_all_prob=0.05,
                 cond_drop_one_prob=0.25, model_name="DiT-2Cond-S/2"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # ---- Load data mappings ----
        with open(CSV_PATH, encoding="utf-8") as f:
            self.csv_rows = list(csv.DictReader(f))
        with open(MANIFEST_PATH, encoding="utf-8") as f:
            self.manifest = json.load(f)
        self.id2manifest = {e['img_id']: e for e in self.manifest}

        # char -> {script: glyph_id}  (glyph_id = y_char, 含字体信息)
        with open(CHAR_META, encoding="utf-8") as f:
            cm = json.load(f)
        self.char2scripts = cm['char2scripts']   # char -> {script: glyph_id}
        self.scripts = cm['scripts']             # ['楷', '隶']

        # calligrapher list
        with open(CALLIGS_JSON, encoding="utf-8") as f:
            calligs = json.load(f)
        self.calligraphers = [c['name'] for c in calligs]
        self.callig2id = {c['name']: c['id'] for c in calligs}

        # (char, script) -> list of csv rows (for GT lookup)
        self.cs2rows = {}
        for r in self.csv_rows:
            self.cs2rows.setdefault((r['character'], r['script']), []).append(r)

        self.supported_chars = set(self.char2scripts.keys())

        # ---- Load main model ----
        model = DiT_2Cond_models[model_name](
            num_calligraphers=num_calligraphers, num_characters=num_characters,
            condition_fusion=condition_fusion,
            callig_embed_dim=callig_embed_dim, char_embed_dim=char_embed_dim,
            cond_drop_all_prob=cond_drop_all_prob, cond_drop_one_prob=cond_drop_one_prob,
            use_checkpoint=False, learn_sigma=True)
        ck = torch.load(main_ckpt, map_location="cpu", weights_only=False)
        sd = ck.get("ema") or ck.get("delta")
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"[main] loaded {os.path.basename(main_ckpt)} "
              f"(step={ck.get('train_steps')}, missing={len(missing)}, unexpected={len(unexpected)})")
        model.to(self.device).eval()
        for p in model.parameters():
            p.requires_grad = False
        self.model = model

        # ---- VAE ----
        from diffusers import AutoencoderKL
        self.vae = AutoencoderKL.from_pretrained(vae_path).to(self.device).eval()

    def _get_glyph_id(self, character, script):
        """glyph_id = f(character, script), 训练时作为 y_char."""
        scripts = self.char2scripts.get(character, {})
        if script not in scripts:
            raise ValueError(f"字「{character}」没有字体「{script}」。"
                             f"可用字体: {list(scripts.keys())}")
        return scripts[script]

    def _find_gt_images(self, character, script, calligrapher, max_n=4):
        """从 CSV 找条件相近的 GT 图 (同字+同体+优先同书家), 从本地 MCCD 加载."""
        rows = self.cs2rows.get((character, script), [])
        matched = [r for r in rows if r['calligrapher'] == calligrapher]
        if len(matched) < max_n:
            others = [r for r in rows if r['calligrapher'] != calligrapher]
            random.shuffle(others)
            matched = matched + others[:max_n - len(matched)]
        imgs = []
        for r in matched[:max_n]:
            img_id = int(r['image_path'].split('/')[-1].replace('.png', ''))
            entry = self.id2manifest.get(img_id)
            if not entry:
                continue
            basename = os.path.basename(entry['orig_path'])
            local = os.path.join(MCCD_CHAR_DIR, character, basename)
            if not os.path.exists(local):
                callig_dir = os.path.join(ROOT, "MCCD", "MCCD", "MCCD-Calligrapher",
                                         "calligrapher_dataset", r['calligrapher'])
                local = os.path.join(callig_dir, basename)
            if os.path.exists(local):
                img = Image.open(local).convert("RGB").resize((256, 256))
                imgs.append((img, r['calligrapher'], r['script']))
        return imgs

    @torch.no_grad()
    def generate(self, character, script, calligrapher, cfg_scale=4.0,
                 num_steps=50, seed=None):
        """生成书法图, 返回 (PIL Image, seed)."""
        if character not in self.char2scripts:
            raise ValueError(f"字「{character}」不在 top6 支持的 {len(self.supported_chars)} 字中。")
        if calligrapher not in self.callig2id:
            raise ValueError(f"书法家「{calligrapher}」不在 top6 列表。")

        glyph_id = self._get_glyph_id(character, script)
        callig_id = self.callig2id[calligrapher]

        if seed is None or seed == 0:
            seed = random.randint(0, 2**31)
        torch.manual_seed(seed)
        z = torch.randn(1, 4, 32, 32, device=self.device)
        yc = torch.tensor([callig_id], dtype=torch.long, device=self.device)
        ych = torch.tensor([glyph_id], dtype=torch.long, device=self.device)

        ddim = create_diffusion(str(num_steps))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            latent = ddim.ddim_sample_loop(
                self.model.forward_with_cfg, z.shape, z,
                clip_denoised=False,
                model_kwargs=dict(y_callig=yc, y_char=ych, cfg_scale=cfg_scale),
                progress=False, device=self.device)
        latent = latent.float()
        img = self.vae.decode(latent / 0.18215).sample
        arr = ((img[0].permute(1, 2, 0).cpu().numpy() + 1) / 2 * 255).clip(0, 255)
        return Image.fromarray(arr.astype(np.uint8)), seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--main-ckpt",
                    default="5script/results/s6_top6_diffonly/20260820-191536-s6-top6-diffonly-resume/checkpoints/0195000.pt")
    ap.add_argument("--vae-path", default="pretrained_models/sd-vae-ft-ema")
    ap.add_argument("--port", type=int, default=7861)
    ap.add_argument("--server-name", default="0.0.0.0")
    ap.add_argument("--share", action="store_true", default=True)
    args = ap.parse_args()

    import gradio as gr

    sampler = CalligraphySampler(main_ckpt=args.main_ckpt, vae_path=args.vae_path)

    def gen(character, script, calligrapher, cfg, steps, seed, history):
        try:
            img, used_seed = sampler.generate(character, script, calligrapher,
                                               cfg_scale=cfg, num_steps=steps, seed=seed)
            gt_imgs = sampler._find_gt_images(character, script, calligrapher, max_n=4)

            caption = f"{character} · {script} · {calligrapher} · seed={used_seed}"
            history = history or []
            history = [(img, caption)] + history

            # GT slots with dynamic labels (书家 · 字体)
            gt_outs = []
            for i in range(4):
                if i < len(gt_imgs):
                    g = gt_imgs[i]
                    gt_outs.append(gr.Image(value=g[0], label=f"GT: {g[1]} · {g[2]}"))
                else:
                    gt_outs.append(gr.Image(value=None, label=f"GT {i+1}"))
            status = f"已生成: {character} · {script} · {calligrapher} · seed={used_seed} · CFG={cfg}"
            return img, gt_outs[0], gt_outs[1], gt_outs[2], gt_outs[3], history, status
        except ValueError as e:
            raise gr.Error(str(e))
        except Exception as e:
            raise gr.Error(f"生成失败: {type(e).__name__}: {e}")

    with gr.Blocks(title="书法生成 · DiT-2Cond top6", analytics_enabled=False) as demo:
        gr.Markdown("# 书法生成 · DiT-2Cond-S/2 (195k diff-only)")
        gr.Markdown(f"支持 **{len(sampler.supported_chars)}** 字 · "
                    f"**{len(sampler.calligraphers)}** 位书法家 · "
                    f"**{len(sampler.scripts)}** 种字体 (top6)\n\n"
                    f"支持字表见 `tools/controlnet/supported_chars.txt`")

        with gr.Row():
            with gr.Column(scale=1, min_width=200):
                char_in = gr.Textbox(label="汉字", value="永",
                                     placeholder="输入一个汉字 (须在支持列表中)")
                script_in = gr.Dropdown(label="字体", choices=sampler.scripts,
                                        value=sampler.scripts[0])
                callig_in = gr.Dropdown(label="书法家", choices=sampler.calligraphers,
                                        value=sampler.calligraphers[0])
                cfg_in = gr.Slider(1, 10, value=2.5, step=0.5, label="CFG 强度")
                steps_in = gr.Slider(10, 100, value=50, step=5, label="DDIM 步数")
                seed_in = gr.Slider(0, 99999, value=0, step=1,
                                    label="随机种子 (0=随机)")
                btn = gr.Button("生成", variant="primary")
                status = gr.Textbox(label="状态", interactive=False)

            with gr.Column(scale=1, min_width=256):
                out_img = gr.Image(label="生成结果", width=256, height=256)

        gr.Markdown("### 条件相近 GT")
        with gr.Row():
            gt1 = gr.Image(label="GT 1", width=256, height=256, interactive=False)
            gt2 = gr.Image(label="GT 2", width=256, height=256, interactive=False)
            gt3 = gr.Image(label="GT 3", width=256, height=256, interactive=False)
            gt4 = gr.Image(label="GT 4", width=256, height=256, interactive=False)

        gr.Markdown("### 历史生成 (累积)")
        history_gallery = gr.Gallery(label="历史", show_label=False, columns=6, height=400,
                                     object_fit="contain")

        btn.click(
            fn=gen,
            inputs=[char_in, script_in, callig_in, cfg_in, steps_in, seed_in, history_gallery],
            outputs=[out_img, gt1, gt2, gt3, gt4, history_gallery, status]
        )

    demo.launch(server_name=args.server_name, server_port=args.port,
                share=args.share, show_error=True, quiet=False)


if __name__ == "__main__":
    main()
