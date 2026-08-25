"""
Unified High-Level Inference and Sampling API for DiT Calligraphy Generation.

Supports:
- Automatic architecture & hyperparameter resolution from checkpoint metadata
- Seamless support for kl-f4 (64x64, 3ch) and sd-vae (32x32, 4ch)
- S/4, B/4, S/2, XL/2 backbones
- Standard 1-axis CFG and 2-axis independent CFG (style vs glyph vs interaction)
- Name-based (calligrapher name, character, script) or ID-based queries
- Batch generation and CSV dataset batch evaluation
"""

import os
import sys
import json
import csv
from typing import Union, List, Dict, Tuple, Optional
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from models import DiT_2Cond_models, DiT_3Cond_models
from diffusion import create_diffusion
from lora import inject_lora


# 5script default script mapping
SCRIPT_NAMES_TO_ID = {"楷": 0, "楷书": 0, "篆": 1, "篆书": 1, "草": 2, "草书": 2, "行": 3, "行书": 3, "隶": 4, "隶书": 4}
SCRIPT_ID_TO_NAMES = {0: "楷", 1: "篆", 2: "草", 3: "行", 4: "隶"}
DEFAULT_CHARS_PER_SCRIPT = 7026


class CalligraphySampler:
    """
    Unified Inference & Sampling Engine.
    """

    def __init__(
        self,
        ckpt_path: str,
        device: Optional[str] = None,
        vae_path: Optional[str] = None,
        use_ema: bool = True,
        labels_dir: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.ckpt_path = ckpt_path
        self.use_ema = use_ema

        # 1. Load Checkpoint
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        self.ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.saved_args = self.ckpt.get("args", {})

        # 2. Extract hyperparameters
        self._parse_config(vae_path)

        # 3. Build Model & Load Weights
        self._build_model()

        # 4. Load VAE
        self._load_vae()

        # 5. Load ID mappings
        self._load_label_maps(labels_dir)

    def _get_arg(self, key: str, fallback=None):
        if isinstance(self.saved_args, dict):
            return self.saved_args.get(key, fallback)
        return getattr(self.saved_args, key, fallback)

    def _parse_config(self, vae_path_override: Optional[str] = None):
        self.model_name = self._get_arg("model", "DiT-2Cond-S/4")
        self.cond_mode = self._get_arg("cond_mode", "2cond")
        self.condition_fusion = self._get_arg("condition_fusion", "factorized_add")
        self.image_size = int(self._get_arg("image_size", 256))
        self.num_calligraphers = int(self._get_arg("num_calligraphers", 1011))
        self.num_characters = int(self._get_arg("num_characters", 35130))
        self.num_scripts = int(self._get_arg("num_scripts", 5))

        self.callig_embed_dim = self._get_arg("callig_embed_dim", 128)
        self.char_embed_dim = self._get_arg("char_embed_dim", 768)
        self.script_embed_dim = self._get_arg("script_embed_dim", None)

        self.vae_downscale = int(self._get_arg("vae_downscale", 4 if "/4" in self.model_name else 8))
        self.latent_channels = int(self._get_arg("latent_channels", 3 if self.vae_downscale == 4 else 4))
        default_sf = 0.102079 if self.vae_downscale == 4 else 0.18215
        self.vae_scaling_factor = float(self._get_arg("vae_scaling_factor", default_sf))
        self.use_glyph_cond = bool(self._get_arg("w_glyph_cond", False))

        # Inspect weight dict directly if available to guarantee zero shape mismatch
        w_dict = self.ckpt.get("ema") or self.ckpt.get("delta") or self.ckpt.get("model") or self.ckpt
        self.patch_size = 4 if "/4" in self.model_name else 2
        num_patches = 256
        if isinstance(w_dict, dict):
            # 1. x_embedder: [hidden_size, in_channels, patch_size, patch_size]
            for k in ("x_embedder.proj.weight", "module.x_embedder.proj.weight"):
                if k in w_dict:
                    self.latent_channels = w_dict[k].shape[1]
                    self.patch_size = w_dict[k].shape[2]
                    if self.patch_size == 2 and self.vae_downscale == 4:
                        self.vae_downscale = 8
                        self.vae_scaling_factor = 0.18215
                    break
            # 2. pos_embed: [1, num_patches, hidden_size]
            for k in ("pos_embed", "module.pos_embed"):
                if k in w_dict:
                    num_patches = w_dict[k].shape[1]
                    break
            # 3. y_callig_embedder: [num_calligraphers + 1, callig_embed_dim]
            for k in ("y_callig_embedder.embedding_table.weight", "module.y_callig_embedder.embedding_table.weight"):
                if k in w_dict:
                    self.num_calligraphers = w_dict[k].shape[0] - 1
                    self.callig_embed_dim = w_dict[k].shape[1]
                    break
            # 4. y_char_embedder: [num_characters + 1, char_embed_dim]
            for k in ("y_char_embedder.embedding_table.weight", "module.y_char_embedder.embedding_table.weight"):
                if k in w_dict:
                    self.num_characters = w_dict[k].shape[0] - 1
                    self.char_embed_dim = w_dict[k].shape[1]
                    break

        grid_size = int(num_patches ** 0.5)
        self.latent_spatial = grid_size * self.patch_size

        # Determine default VAE path
        if vae_path_override:
            self.vae_path = vae_path_override
        else:
            saved_vae = self._get_arg("vae_path", None)
            if saved_vae and os.path.exists(os.path.join(ROOT, saved_vae)):
                self.vae_path = os.path.join(ROOT, saved_vae)
            elif saved_vae and os.path.exists(saved_vae):
                self.vae_path = saved_vae
            elif self.vae_downscale == 4:
                cand = os.path.join(ROOT, "pretrained_models", "kl-f4")
                self.vae_path = cand if os.path.exists(cand) else "stabilityai/sd-vae-ft-ema"
            else:
                cand = os.path.join(ROOT, "pretrained_models", "sd-vae-ft-ema")
                self.vae_path = cand if os.path.exists(cand) else "stabilityai/sd-vae-ft-ema"

    def _build_model(self):
        input_size = self.latent_spatial
        if self.cond_mode == "2cond":
            if self.model_name not in DiT_2Cond_models:
                raise ValueError(f"Unknown 2-Cond model: {self.model_name}")
            self.model = DiT_2Cond_models[self.model_name](
                input_size=input_size,
                in_channels=self.latent_channels,
                num_calligraphers=self.num_calligraphers,
                num_characters=self.num_characters,
                use_checkpoint=False,
                condition_fusion=self.condition_fusion,
                callig_embed_dim=self.callig_embed_dim,
                char_embed_dim=self.char_embed_dim,
                cond_drop_all_prob=0.0,
                cond_drop_one_prob=0.0,
                use_glyph_cond=self.use_glyph_cond,
            )
        else:
            if self.model_name not in DiT_3Cond_models:
                raise ValueError(f"Unknown 3-Cond model: {self.model_name}")
            self.model = DiT_3Cond_models[self.model_name](
                input_size=input_size,
                in_channels=self.latent_channels,
                num_calligraphers=self.num_calligraphers,
                num_scripts=self.num_scripts,
                num_characters=self.num_characters,
                use_checkpoint=False,
                condition_fusion=self.condition_fusion,
                callig_embed_dim=self.callig_embed_dim,
                script_embed_dim=self.script_embed_dim,
                char_embed_dim=self.char_embed_dim,
                cond_drop_all_prob=0.0,
                cond_drop_one_prob=0.0,
            )

        # Load weights
        weights = None
        if self.use_ema and "ema" in self.ckpt:
            weights = self.ckpt["ema"]
        elif "delta" in self.ckpt:
            weights = self.ckpt["delta"]
        elif "model" in self.ckpt:
            weights = self.ckpt["model"]
        else:
            weights = self.ckpt

        # Check for LoRA
        if self._get_arg("use_lora", False):
            r = int(self._get_arg("lora_r", 16))
            target = self._get_arg("lora_target", "all")
            self.model = inject_lora(self.model, r=r, lora_alpha=r, target=target)

        missing, unexpected = self.model.load_state_dict(weights, strict=False)
        self.model = self.model.to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    def _load_vae(self):
        from diffusers.models import AutoencoderKL
        if os.path.exists(self.vae_path):
            self.vae = AutoencoderKL.from_pretrained(self.vae_path).to(self.device).eval()
        else:
            self.vae = AutoencoderKL.from_pretrained(
                "stabilityai/sd-vae-ft-ema" if self.vae_downscale == 8 else "stabilityai/sd-vae-ft-mse"
            ).to(self.device).eval()
        for p in self.vae.parameters():
            p.requires_grad_(False)

    def _load_label_maps(self, labels_dir: Optional[str] = None):
        ld = labels_dir or os.path.join(ROOT, "labels")
        self.char_to_id = {}
        self.callig_to_id = {}

        # 1. char_to_id
        c_path = os.path.join(ld, "character_to_id.json")
        if os.path.isfile(c_path):
            with open(c_path, "r", encoding="utf-8") as f:
                self.char_to_id = json.load(f)

        # 2. calligrapher_to_id
        cal_path = os.path.join(ld, "calligrapher_to_id.json")
        if os.path.isfile(cal_path):
            with open(cal_path, "r", encoding="utf-8") as f:
                self.callig_to_id = json.load(f)

        # 3. Clean top30 calligraphers
        top30_path = os.path.join(ROOT, "5script", "top30_calligs_clean.json")
        if os.path.isfile(top30_path):
            with open(top30_path, "r", encoding="utf-8") as f:
                top30_data = json.load(f)
                for sid, clist in top30_data.items():
                    for entry in clist:
                        self.callig_to_id[entry["name"]] = int(entry["id"])

    def resolve_ids(
        self,
        calligrapher: Union[str, int],
        character: Union[str, int],
        script: Union[str, int] = "楷",
    ) -> Tuple[int, int, int]:
        """
        Resolves (calligrapher_id, glyph_id, character_id).
        glyph_id = script_id * 7026 + character_id
        """
        # 1. Script ID
        if isinstance(script, int):
            script_id = script
        else:
            script_id = SCRIPT_NAMES_TO_ID.get(script, 0)

        # 2. Calligrapher ID
        if isinstance(calligrapher, int):
            callig_id = calligrapher
        else:
            callig_id = self.callig_to_id.get(str(calligrapher), 0)

        # 3. Character ID & Glyph ID
        if isinstance(character, int):
            char_id = character
            glyph_id = script_id * DEFAULT_CHARS_PER_SCRIPT + char_id
        else:
            char_id = self.char_to_id.get(str(character), 0)
            glyph_id = script_id * DEFAULT_CHARS_PER_SCRIPT + char_id

        return callig_id, glyph_id, char_id

    @torch.no_grad()
    def sample(
        self,
        calligrapher: Union[str, int] = "颜真卿",
        character: Union[str, int] = "永",
        script: Union[str, int] = "楷",
        num_steps: int = 50,
        cfg_scale: float = 4.0,
        cfg_callig: Optional[float] = None,
        cfg_glyph: Optional[float] = None,
        w_inter: float = 0.0,
        seed: Optional[int] = None,
        batch_size: int = 1,
        return_pil: bool = True,
        g: Optional[torch.Tensor] = None,
    ) -> Union[Image.Image, List[Image.Image], torch.Tensor]:
        """
        Generate calligraphy image(s).

        If cfg_callig and cfg_glyph are provided, 2-Axis CFG is used:
          eps_guided = eps_0 + cfg_glyph*(eps_G - eps_0) + cfg_callig*(eps_A - eps_0) + w_inter*(...)
        Otherwise standard 1-axis CFG (cfg_scale) is used.
        """
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        callig_id, glyph_id, char_id = self.resolve_ids(calligrapher, character, script)

        y_c = torch.tensor([callig_id] * batch_size, device=self.device, dtype=torch.long)
        y_g = torch.tensor([glyph_id] * batch_size, device=self.device, dtype=torch.long)

        # Sampling noise
        z = torch.randn(
            batch_size,
            self.latent_channels,
            self.latent_spatial,
            self.latent_spatial,
            device=self.device,
        )

        diffusion = create_diffusion(str(num_steps))

        use_2axis = (cfg_callig is not None or cfg_glyph is not None)
        if use_2axis:
            c_callig = 2.0 if cfg_callig is None else float(cfg_callig)
            c_glyph = cfg_scale if cfg_glyph is None else float(cfg_glyph)

            def model_fn(x, t):
                return self.model.forward_with_2axis_cfg(
                    x, t, y_c, y_g,
                    cfg_callig=c_callig,
                    cfg_glyph=c_glyph,
                    w_inter=w_inter,
                    g=g,
                )
            samples = diffusion.ddim_sample_loop(
                model_fn, z.shape, z, clip_denoised=False, device=self.device
            )
        else:
            model_kwargs = dict(y_callig=y_c, y_char=y_g, cfg_scale=cfg_scale)
            if g is not None:
                model_kwargs["g"] = g
            samples = diffusion.ddim_sample_loop(
                self.model.forward_with_cfg,
                z.shape,
                z,
                clip_denoised=False,
                model_kwargs=model_kwargs,
                device=self.device,
            )

        # Decode via VAE
        decoded = self.vae.decode(samples / self.vae_scaling_factor).sample  # (B, 3, H, W) in [-1, 1]

        if not return_pil:
            return decoded

        # Convert to PIL
        decoded_01 = ((decoded.clamp(-1.0, 1.0) + 1.0) / 2.0).cpu().permute(0, 2, 3, 1).numpy()
        images = [Image.fromarray((arr * 255).astype(np.uint8)) for arr in decoded_01]

        return images[0] if batch_size == 1 else images

    @torch.no_grad()
    def sample_csv(
        self,
        csv_path: str,
        out_dir: str,
        n: Optional[int] = None,
        num_steps: int = 50,
        cfg_scale: float = 4.0,
        cfg_callig: Optional[float] = None,
        cfg_glyph: Optional[float] = None,
        w_inter: float = 0.0,
        seed: int = 0,
        batch_size: int = 16,
    ) -> List[str]:
        """
        Batch sample from a CSV file (containing calligrapher_id, character_id, script_id).
        """
        os.makedirs(out_dir, exist_ok=True)
        with open(csv_path, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if n is not None:
            rows = rows[:n]

        saved_paths = []
        torch.manual_seed(seed)

        for i in range(0, len(rows), batch_size):
            chunk = rows[i:i + batch_size]
            b_size = len(chunk)

            y_c = torch.tensor(
                [int(r.get("calligrapher_id", 0)) for r in chunk],
                device=self.device,
                dtype=torch.long,
            )
            y_g = torch.tensor(
                [int(r.get("glyph_id", int(r.get("script_id", 0)) * DEFAULT_CHARS_PER_SCRIPT + int(r.get("character_id", 0)))) for r in chunk],
                device=self.device,
                dtype=torch.long,
            )

            z = torch.randn(
                b_size,
                self.latent_channels,
                self.latent_spatial,
                self.latent_spatial,
                device=self.device,
            )

            diffusion = create_diffusion(str(num_steps))
            use_2axis = (cfg_callig is not None or cfg_glyph is not None)

            if use_2axis:
                c_callig = 2.0 if cfg_callig is None else float(cfg_callig)
                c_glyph = cfg_scale if cfg_glyph is None else float(cfg_glyph)

                def model_fn(x, t):
                    return self.model.forward_with_2axis_cfg(
                        x, t, y_c, y_g,
                        cfg_callig=c_callig,
                        cfg_glyph=c_glyph,
                        w_inter=w_inter,
                    )
                samples = diffusion.ddim_sample_loop(
                    model_fn, z.shape, z, clip_denoised=False, device=self.device
                )
            else:
                model_kwargs = dict(y_callig=y_c, y_char=y_g, cfg_scale=cfg_scale)
                samples = diffusion.ddim_sample_loop(
                    self.model.forward_with_cfg,
                    z.shape,
                    z,
                    clip_denoised=False,
                    model_kwargs=model_kwargs,
                    device=self.device,
                )

            decoded = self.vae.decode(samples / self.vae_scaling_factor).sample
            decoded_01 = ((decoded.clamp(-1.0, 1.0) + 1.0) / 2.0).cpu().permute(0, 2, 3, 1).numpy()

            for k, arr in enumerate(decoded_01):
                idx = i + k
                row = chunk[k]
                c_name = row.get("calligrapher", f"c{row.get('calligrapher_id', '')}")
                s_name = SCRIPT_ID_TO_NAMES.get(int(row.get("script_id", 0)), "楷")
                ch_name = row.get("character", f"ch{row.get('character_id', '')}")
                fname = f"{idx:04d}_{c_name}_{s_name}_{ch_name}.png"
                out_path = os.path.join(out_dir, fname)
                Image.fromarray((arr * 255).astype(np.uint8)).save(out_path)
                saved_paths.append(out_path)

        return saved_paths
