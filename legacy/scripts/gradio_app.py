# -*- coding: utf-8 -*-
"""
书法 3-条件生成 Gradio App
输入一个汉字 + 选择书法家/字体 -> DDIM 生成书法图片。
基于 DiT-3Cond-XL/2 + 预训练 body 微调（LoRA r8 最好）的 ckpt。

用法（远程）:
  /opt/conda/bin/python gradio_app.py --share   # 生成公网链接
  /opt/conda/bin/python gradio_app.py --port 7860 --model-name DiT-3Cond-XL/2 \
      --ckpt results/exp_xl_head_r8/20260814-211629-DiT-3Cond-XL-2/checkpoints/0037000.pt \
      --use-lora 1 --lora-r 8 --lora-target all --pretrained pretrained_models/DiT-XL-2-256x256.pt
"""
import os, sys, json, argparse, random
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
import torch
from PIL import Image

os.environ.setdefault("XFORMERS_DISABLED", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from models import DiT_3Cond_models
from lora import inject_lora
from diffusers.models import AutoencoderKL
from diffusion import create_diffusion

T_EVAL = 150  # unused here; sampling uses DDIM over num_sampling_steps


def _s2b(v):
    return str(v).lower() in ("1", "true", "yes", "on")


def load_maps(maps_path=None):
    p = maps_path or os.path.join(HERE, "labels", "final_id_maps.json")
    with open(p, encoding="utf-8") as f:
        m = json.load(f)
    return m["id_to_name"], m["name_to_id"]


class Sampler:
    """持有模型 + VAE + 映射，提供 generate()。模型只加载一次。"""

    def __init__(self, model_name, ckpt_path, pretrained, use_lora,
                 lora_r, lora_target, num_calligraphers=1873,
                 num_scripts=12, num_characters=7765, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        id_to_name, name_to_id = load_maps()
        self.id_to_name, self.name_to_id = id_to_name, name_to_id
        # 下拉选项（按名称排序）
        self.calligraphers = sorted(name_to_id["calligrapher"].keys())
        self.scripts = sorted(name_to_id["script"].keys())
        self.characters = set(name_to_id["character"].keys())

        # ---- model ----
        model = DiT_3Cond_models[model_name](
            input_size=32, num_calligraphers=num_calligraphers,
            num_scripts=num_scripts, num_characters=num_characters,
            use_checkpoint=False)
        if pretrained and pretrained.lower() != "none" and os.path.exists(pretrained):
            pre = torch.load(pretrained, map_location="cpu", weights_only=False)
            if "model" in pre:
                pre = pre["model"]
            pre = {k: v for k, v in pre.items()
                   if not k.startswith(("y_embedder", "y_callig", "y_script", "y_char", "cond_fusion"))}
            model.load_state_dict(pre, strict=False)
            print(f"[Sampler] pretrained body loaded: {pretrained}")
        if use_lora:
            model = inject_lora(model, r=lora_r, lora_alpha=lora_r, target=lora_target)
            print(f"[Sampler] LoRA injected r={lora_r} target={lora_target}")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        delta = ckpt.get("delta", ckpt.get("model", ckpt))
        missing, unexpected = model.load_state_dict(delta, strict=False)
        print(f"[Sampler] ckpt delta loaded: missing={len(missing)} unexpected={len(unexpected)}")
        self.model = model.to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad = False

        self.vae = AutoencoderKL.from_pretrained(
            "pretrained_models/sd-vae-ft-ema").to(self.device).eval()
        self.diffusion = create_diffusion(timestep_respacing="")  # full 1000; DDIM respaces internally

    def generate(self, character, calligrapher, script, cfg_scale=4.0,
                 num_steps=50, seed=None):
        """返回 PIL Image（RGB 书法图）。"""
        if character not in self.name_to_id["character"]:
            raise ValueError(f"字「{character}」不在数据集（7765 字）中。")
        if calligrapher not in self.name_to_id["calligrapher"]:
            raise ValueError(f"书法家「{calligrapher}」不在列表。")
        if script not in self.name_to_id["script"]:
            raise ValueError(f"字体「{script}」不在列表。")

        y_callig = self.name_to_id["calligrapher"][calligrapher]
        y_script = self.name_to_id["script"][script]
        y_char = self.name_to_id["character"][character]

        ck = create_diffusion(str(num_steps))  # DDIM over num_steps
        with torch.no_grad():
            torch.manual_seed(seed if seed is not None else random.randint(0, 2**31))
            z_t = torch.randn(1, 4, 32, 32, device=self.device)
            y_callig_t = torch.tensor([y_callig], device=self.device)
            y_script_t = torch.tensor([y_script], device=self.device)
            y_char_t = torch.tensor([y_char], device=self.device)
            mk = dict(y_callig=y_callig_t, y_script=y_script_t, y_char=y_char_t,
                      cfg_scale=cfg_scale)
            samples = ck.ddim_sample_loop(
                self.model.forward_with_cfg, z_t.shape, z_t,
                clip_denoised=False, model_kwargs=mk, progress=False, device=self.device)
            img = self.vae.decode(samples / 0.18215).sample  # [-1,1]
        arr = ((img[0].permute(1, 2, 0).cpu().numpy() + 1.0) / 2.0 * 255.0).clip(0, 255)
        return Image.fromarray(arr.astype(np.uint8))


def build_arg_parser():
    ap = argparse.ArgumentParser(description="书法 3 条件生成 · gradio")
    ap.add_argument("--model-name", default="DiT-3Cond-XL/2")
    ap.add_argument("--ckpt", default=None,
                    help="单 ckpt 路径（当不提供 --config 或 config 无 ckpts 时使用）")
    ap.add_argument("--use-lora", type=_s2b, default=False)
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--lora-target", default="all")
    ap.add_argument("--pretrained", default="pretrained_models/DiT-XL-2-256x256.pt")
    ap.add_argument("--share", action="store_true", default=True,
                    help="默认开启公网分享（gradio share 隧道）")
    ap.add_argument("--no-share", action="store_true",
                    help="关闭公网分享，仅本地 0.0.0.0:port")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--server-name", default="0.0.0.0")
    ap.add_argument("--config", default="ckpt_config.json",
                    help="含多个 ckpt 的配置；存在时优先于 --ckpt")
    ap.add_argument("--preload", action="store_true", default=False,
                    help="启动时立即加载默认模型（否则首次生成时加载）")
    return ap


def load_ckpt_config(path):
    import json
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    entries = cfg.get("ckpts", [])
    return entries, cfg.get("default", entries[0]["name"] if entries else None)


def main():
    args = build_arg_parser().parse_args()

    # 优先从配置加载多 ckpt（若文件存在）；否则退回单 ckpt（兼容 --ckpt）
    entries = []
    if args.config and os.path.exists(args.config):
        entries, default_name = load_ckpt_config(args.config)
    if not entries:
        entries = [dict(name=os.path.basename(args.ckpt), model=args.model_name,
                        ckpt=args.ckpt, use_lora=args.use_lora, lora_r=args.lora_r,
                        lora_target=args.lora_target, pretrained=args.pretrained)]
        default_name = entries[0]["name"]

    import gradio as gr

    def _build_sampler(entry):
        return Sampler(model_name=entry["model"], ckpt_path=entry["ckpt"],
                       pretrained=entry.get("pretrained"), use_lora=entry.get("use_lora", False),
                       lora_r=entry.get("lora_r", 0), lora_target=entry.get("lora_target", "all"))

    # 当前 sampler（懒加载：选中的配置才加载；切换时释放旧的显存）
    state = {"cur": default_name, "sam": None}

    def get_sam(name):
        if state["sam"] is None or name != state["cur"]:
            entry = next(e for e in entries if e["name"] == name)
            if state["sam"] is not None:
                del state["sam"]
                import torch
                torch.cuda.empty_cache()
            print(f"[gradio] loading model: {name} ...")
            state["sam"] = _build_sampler(entry)
            state["cur"] = name
        return state["sam"]

    # 加载默认模型的映射表用于下拉选项
    first = get_sam(default_name)
    calligraphers, scripts = first.calligraphers, first.scripts
    model_names = [e["name"] for e in entries]

    def gen(model_sel, character, callig, script, cfg, steps, seed):
        try:
            sam = get_sam(model_sel)
            img = sam.generate(character, callig, script, cfg_scale=cfg,
                               num_steps=steps, seed=seed)
            return img
        except ValueError as e:
            raise gr.Error(str(e))
        except Exception as e:
            raise gr.Error(f"生成失败: {type(e).__name__}: {e}")

    demo = gr.Interface(
        fn=gen,
        inputs=[
            gr.Dropdown(label="模型 / checkpoint", choices=model_names,
                        value=default_name),
            gr.Textbox(label="汉字", value="永"),
            gr.Dropdown(label="书法家", choices=calligraphers,
                        value=calligraphers[0]),
            gr.Dropdown(label="字体", choices=scripts, value=scripts[0]),
            gr.Slider(1, 10, value=4, step=0.5, label="CFG 强度"),
            gr.Slider(10, 100, value=50, step=5, label="DDIM 步数"),
            gr.Slider(0, 9999, value=0, step=1, label="随机种子"),
        ],
        outputs=gr.Image(label="生成书法", shape=(256, 256)),
        title="书法 3 条件生成 · DiT-3Cond",
        description="输入汉字（须在 7765 字集中），选模型/书法家/字体，点击 Submit 生成。"
                    "顶部「模型 / checkpoint」可切换已训练的不同容量方案。",
        allow_flagging="never",
        examples=[[default_name, "永", calligraphers[0], "楷", 4.0, 50, 0],
                  [default_name, "福", "王羲之", "行", 4.0, 50, 0]],
    )
    demo.launch(server_name=args.server_name, server_port=args.port,
                share=(not args.no_share) and args.share, show_error=True, quiet=False)


if __name__ == "__main__":
    main()
