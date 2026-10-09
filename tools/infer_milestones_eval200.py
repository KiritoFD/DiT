#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/infer_milestones_eval200.py
各阶段代表性模型在 eval200 基准集上的统一跨阶段推理适配器。

功能特性：
1. 单一入口统一适配所有历史阶段（v10b, v13, v21, v23, v66, v68, v70）；
2. 跨阶段条件自动转换：
   - 骨架驱动模型 (v10b/v13/v21/v23/v70): 自动从 std_path 编码 skel_latent 送入 g 通道；
   - 字表驱动模型 (v66/v68): 自动从 triple_tables 映射 y_char, y_callig, y_script；
   - 跨版本书家字典自动对齐 (41名家 / 45名家 / 87对 / 10名家 remap)；
3. 严格确定性评测：同一个字符在所有阶段采用完全相同的随机噪声初值 (seed = 42 + idx)；
4. 结果分阶段落盘至 exp_milestones/eval200_outputs/{stage_id}/；
5. 可选自动同步至 assets/poster_tree/ 并一键重绘汇报海报。
"""
import os
import sys
import argparse
import json
import csv
import time
from typing import Dict, Any, List

# 保证在工作目录下运行
BASE_DIR = "/root/Workspace/xy/DiT"
if os.path.exists(BASE_DIR):
    os.chdir(BASE_DIR)
sys.path.insert(0, BASE_DIR)

import torch
import torchvision.transforms as T
from PIL import Image
from diffusers.models import AutoencoderKL

from src.eval.model_io import build_model_from_args
from src.loss import create_diffusion_or_flow


class MilestoneInferenceEngine:
    def __init__(self, device_str: str = "cpu", num_threads: int = 16, vae_path: str = None):
        self.device = torch.device(device_str)
        if device_str == "cpu":
            torch.set_num_threads(num_threads)
            print(f"[Engine] 使用 CPU 推理，启用 {num_threads} 线程并行")
        else:
            print(f"[Engine] 使用设备: {self.device}")

        # 载入单例 VAE
        if not vae_path:
            candidates = [
                os.path.join(BASE_DIR, "pretrained_models/sd-vae-ft-ema"),
                os.path.join(BASE_DIR, "data/pretrained/pretrained_models/sd-vae-ft-ema"),
                "stabilityai/sd-vae-ft-ema"
            ]
            for c in candidates:
                if os.path.exists(c):
                    vae_path = c
                    break
            if not vae_path:
                vae_path = "stabilityai/sd-vae-ft-ema"

        print(f"[Engine] 载入统一 VAE: {vae_path}")
        self.vae = AutoencoderKL.from_pretrained(vae_path).to(self.device).eval()
        for p in self.vae.parameters():
            p.requires_grad_(False)
        self.sf = 0.18215

        # 图像预处理
        self.img_tf = T.Compose([
            T.ToTensor(),
            T.Normalize([0.5], [0.5])
        ])

        # 缓存映射表
        self._load_maps()

    def _load_maps(self):
        """载入各阶段需要的映射文件"""
        self.maps = {}
        
        # 1. 41 名家映射 (v10b)
        p41 = os.path.join(BASE_DIR, "assets/callig_id_map.json")
        if os.path.exists(p41):
            with open(p41, "r", encoding="utf-8") as f:
                d = json.load(f)
                self.maps["callig_41"] = d.get("id_map", d)

        # 2. 45 名家映射 (v13, v23)
        p45 = os.path.join(BASE_DIR, "assets/callig_id_map_50k.json")
        if os.path.exists(p45):
            with open(p45, "r", encoding="utf-8") as f:
                d = json.load(f)
                self.maps["callig_45"] = d.get("id_map", d)

        # 3. 87 书家书体对 (v21)
        p87 = os.path.join(BASE_DIR, "assets/callig_script_id_map.json")
        if os.path.exists(p87):
            with open(p87, "r", encoding="utf-8") as f:
                d = json.load(f)
                self.maps["callig_script_87"] = d.get("pair_to_id", d.get("id_map", d))

        # 4. 三表 remap (v66, v68, v70)
        trip_dir = os.path.join(BASE_DIR, "assets/triple_tables_best_minimal")
        if os.path.exists(trip_dir):
            for k, fn in [("callig_remap", "callig_remap.json"),
                          ("char_remap", "char_remap.json"),
                          ("font_remap", "font_remap.json")]:
                fp = os.path.join(trip_dir, fn)
                if os.path.exists(fp):
                    with open(fp, "r", encoding="utf-8") as f:
                        self.maps[k] = {int(k2): int(v2) for k2, v2 in json.load(f).items()}

    def encode_skel(self, img_path: str) -> torch.Tensor:
        """读取标准骨架 PNG 并编码成 VAE latent (1, 4, 32, 32)"""
        full_path = os.path.join(BASE_DIR, img_path) if not os.path.isabs(img_path) else img_path
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"骨架图片不存在: {full_path}")
        im = Image.open(full_path).convert("RGB").resize((256, 256))
        t = self.img_tf(im).unsqueeze(0).to(self.device)
        with torch.no_grad():
            latent = self.vae.encode(t).latent_dist.sample() * self.sf
        return latent

    def decode_latent(self, lat: torch.Tensor) -> Image.Image:
        """将 VAE latent (1, 4, 32, 32) 解码为 PIL Image (256, 256)"""
        with torch.no_grad():
            dec = self.vae.decode(lat.to(self.device) / self.sf).sample
        img_arr = dec[0, :4].permute(1, 2, 0).clamp(-1, 1).add(1).mul(127.5).byte().cpu().numpy()
        # 取前 3 通道保存为 RGB
        if img_arr.shape[-1] >= 3:
            img_arr = img_arr[..., :3]
        return Image.fromarray(img_arr)

    def load_stage_model(self, meta: dict):
        """载入指定阶段的模型并配置其扩散求解器"""
        stage_id = meta["stage_id"]
        ckpt_path = meta["ckpt"]
        resolved_cfg_path = meta.get("resolved_cfg")
        cfg_path = meta.get("config")

        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"[{stage_id}] 权重文件不存在: {ckpt_path}")

        cfg = {}
        if resolved_cfg_path and os.path.exists(resolved_cfg_path):
            with open(resolved_cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        elif cfg_path and os.path.exists(cfg_path):
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)

        ckpt = torch.load(ckpt_path, map_location="cpu")
        ckpt_args = ckpt.get("args")
        if ckpt_args is not None:
            arg_dict = vars(ckpt_args) if hasattr(ckpt_args, "__dict__") else (ckpt_args if isinstance(ckpt_args, dict) else {})
            for k, v in arg_dict.items():
                if v is not None:
                    cfg[k] = v

        # 构建模型
        model = build_model_from_args(cfg, device="cpu").eval()
        sd = ckpt.get("ema") or ckpt.get("model") or ckpt
        clean_sd = {k.replace("_orig_mod.", "").replace("module.", ""): v for k, v in sd.items()}
        model.load_state_dict(clean_sd, strict=False)
        model = model.to(self.device).eval()

        # 构建扩散/流匹配采样器 (统一 20 步)
        diff_type = "ddpm" if "v10" in stage_id else "flow"
        learn_sigma = bool(cfg.get("learn_sigma", False))
        diffusion = create_diffusion_or_flow("20", diffusion_type=diff_type, learn_sigma=learn_sigma)

        return model, diffusion, cfg


def infer_stage_on_eval200(
    engine: MilestoneInferenceEngine,
    meta: dict,
    eval_rows: List[dict],
    out_dir: str,
    selected_indices: List[int] = None
):
    stage_id = meta["stage_id"]
    print(f"\n=======================================================")
    print(f"▶ 正在运行阶段推理: [{stage_id}] {meta['name']}")
    print(f"=======================================================")
    
    stage_out_dir = os.path.join(out_dir, stage_id)
    os.makedirs(stage_out_dir, exist_ok=True)

    # 1. 载入模型与采样器
    t0 = time.time()
    model, diffusion, cfg = engine.load_stage_model(meta)
    print(f"  ✓ 模型与采样器加载完成 (耗时: {time.time() - t0:.2f}s)")

    # 2. 预备各模型的条件解析分支
    cond_mode = meta.get("cond_mode", "")
    use_script_cond = bool(cfg.get("use_script_cond", False))

    indices_to_run = selected_indices if selected_indices is not None else list(range(len(eval_rows)))
    print(f"  本次将推理 {len(indices_to_run)} 个样本...")

    for step_idx, row_i in enumerate(indices_to_run):
        r = eval_rows[row_i]
        char = r["character"]
        callig = r["calligrapher"]
        script = r["script"]
        cid_raw = int(r["calligrapher_id"])
        chid_raw = int(r["character_id"])
        scid_raw = int(r["script_id"]) if r.get("script_id") else 0

        # 固定确定性初始噪声
        g_seed = torch.Generator(device=engine.device).manual_seed(42 + row_i)
        z_init = torch.randn(1, 4, 32, 32, generator=g_seed, device=engine.device)

        # 构造输入条件
        # A. 骨架条件 (若该阶段需要 g)
        need_skel = bool(cfg.get("skel_as_glyph_cond", False) or cfg.get("use_glyph_cond", False))
        skel_lat = None
        if need_skel:
            skel_lat = engine.encode_skel(r["std_path"])

        # B. 书法家条件 y_callig
        if "callig_remap" in engine.maps and ("v66" in stage_id or "v68" in stage_id or "v70" in stage_id):
            y_callig_val = engine.maps["callig_remap"].get(cid_raw, 0)
        elif "callig_script_87" in engine.maps and "v21" in stage_id:
            # v21 使用 (calligrapher, script) pair
            pair_key = f"{callig}_{script}"
            y_callig_val = engine.maps["callig_script_87"].get(pair_key, 0)
        elif "callig_45" in engine.maps and ("v13" in stage_id or "v23" in stage_id):
            y_callig_val = engine.maps["callig_45"].get(str(cid_raw), engine.maps["callig_45"].get(callig, 0))
        elif "callig_41" in engine.maps and "v10" in stage_id:
            y_callig_val = engine.maps["callig_41"].get(str(cid_raw), engine.maps["callig_41"].get(callig, 0))
        else:
            y_callig_val = 0
        y_callig = torch.tensor([int(y_callig_val)], device=engine.device)

        # C. 字符条件 y_char
        if "char_remap" in engine.maps and ("v66" in stage_id or "v68" in stage_id):
            y_char_val = engine.maps["char_remap"].get(chid_raw, 0)
        else:
            y_char_val = 0
        y_char = torch.tensor([int(y_char_val)], device=engine.device)

        # D. 书体条件 y_script
        model_kwargs = {
            "y_callig": y_callig,
            "y_char": y_char,
            "g": skel_lat
        }
        if use_script_cond:
            font_remap = engine.maps.get("font_remap", {})
            y_script = torch.tensor([font_remap.get(scid_raw, 0)], device=engine.device)
            model_kwargs["y_script"] = y_script

        # 3. 执行采样 (20 步)
        t_sample_start = time.time()
        with torch.no_grad():
            sample_lat = diffusion.ddim_sample_loop(
                model, z_init.shape, z_init,
                clip_denoised=False,
                model_kwargs=model_kwargs,
                device=engine.device
            )
        sample_time = time.time() - t_sample_start

        # 4. VAE 解码并保存图像
        out_img = engine.decode_latent(sample_lat)
        
        # 保存标准命名
        fn_std = f"{row_i:02d}.png"
        fn_readable = f"{row_i:02d}_{char}_{callig}_{script}.png"
        out_img.save(os.path.join(stage_out_dir, fn_std))
        out_img.save(os.path.join(stage_out_dir, fn_readable))

        print(f"  [{step_idx+1}/{len(indices_to_run)}] 样本 #{row_i:02d}: '{char}' ({callig} · {script}) -> 生成完毕 ({sample_time:.2f}s)")

    print(f"✓ 阶段 [{stage_id}] 全部样本推理完成，已保存至: {stage_out_dir}")
    return stage_out_dir


def main():
    parser = argparse.ArgumentParser(description="跨阶段代表性模型 eval200 统一对比推理适配器")
    parser.add_argument("--stages", type=str, default="all", help="指定运行阶段，如 '01_v10b,05_v66' 或 'all'")
    parser.add_argument("--eval-csv", type=str, default="exp-std/csv/eval200_fixed.csv", help="评测集 CSV 路径")
    parser.add_argument("--out-dir", type=str, default="exp_milestones/eval200_outputs", help="输出图片根目录")
    parser.add_argument("--indices", type=str, default="0,1,2,3,4,5,6,7,8,9", help="指定样本行索引，例如 '0,1,2,3,4,5,6,7,8,9' 或 'all'")
    parser.add_argument("--device", type=str, default="cpu", help="计算设备 (cpu / cuda:0)")
    parser.add_argument("--threads", type=int, default=16, help="CPU 并行线程数")
    parser.add_argument("--sync-poster-tree", action="store_true", help="是否同时自动同步结果到 assets/poster_tree/")
    args = parser.parse_args()

    # 1. 载入 Manifest
    manifest_path = os.path.join(BASE_DIR, "exp_milestones/milestones_manifest.json")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest 不存在: {manifest_path}，请先运行 setup_milestones_dir.py")
    with open(manifest_path, "r", encoding="utf-8") as f:
        all_milestones = json.load(f)

    # 过滤要执行的阶段
    if args.stages == "all":
        target_stages = all_milestones
    else:
        req = [s.strip() for s in args.stages.split(",")]
        target_stages = [m for m in all_milestones if m["stage_id"] in req or any(req_k in m["stage_id"] for req_k in req)]

    # 2. 读取 eval200 样本行
    eval_csv_path = os.path.join(BASE_DIR, args.eval_csv)
    with open(eval_csv_path, "r", encoding="utf-8") as f:
        eval_rows = list(csv.DictReader(f))
    print(f"载入评测集: {eval_csv_path} (共 {len(eval_rows)} 行)")

    # 确定样本索引
    if args.indices == "all":
        selected_indices = list(range(len(eval_rows)))
    else:
        selected_indices = [int(x.strip()) for x in args.indices.split(",")]

    print(f"目标阶段 ({len(target_stages)} 个): {[m['stage_id'] for m in target_stages]}")
    print(f"测试样本点: {selected_indices}")

    # 3. 初始化推理引擎
    engine = MilestoneInferenceEngine(device_str=args.device, num_threads=args.threads)

    # 4. 逐阶段执行推理
    synced_stages = []
    for m in target_stages:
        stage_dir = infer_stage_on_eval200(
            engine=engine,
            meta=m,
            eval_rows=eval_rows,
            out_dir=args.out_dir,
            selected_indices=selected_indices
        )
        synced_stages.append((m["stage_id"], stage_dir))

    # 5. 若开启 sync-poster-tree，复制到 assets/poster_tree
    if args.sync_poster_tree:
        print("\n=======================================================")
        print("▶ 正在同步结果到本地/汇报 poster_tree ...")
        tree_dir = os.path.join(BASE_DIR, "assets/poster_tree")
        os.makedirs(tree_dir, exist_ok=True)
        import shutil

        # 阶段名映射对照 (匹配 poster_tree 目录格式)
        stage_map_names = {
            "01_v10b": "01_v10b_390k",
            "02_v13": "02_v13_155k",
            "03_v21": "03_v21_skelnet_75k",
            "04_v23": "04_v23_splitnorm_75k",
            "05_v66": "05_v66_condroute_150k",
            "06_v68": "06_v68_sp_aug_200k",
            "07_v70": "07_v70_stdskel_5k"
        }

        for sid, sdir in synced_stages:
            target_tree_sub = stage_map_names.get(sid, sid)
            dst_sub = os.path.join(tree_dir, target_tree_sub)
            os.makedirs(dst_sub, exist_ok=True)
            for idx in selected_indices:
                src_fn = os.path.join(sdir, f"{idx:02d}.png")
                dst_fn = os.path.join(dst_sub, f"{idx:02d}.png")
                if os.path.exists(src_fn):
                    shutil.copy2(src_fn, dst_fn)
            print(f"  • 已同步 [{sid}] -> {dst_sub}")

    print("\n🎉 全部跨阶段统一推理流程执行完毕！")


if __name__ == "__main__":
    main()
