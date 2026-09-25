import json
import torch
import sys
sys.path.insert(0, '.')
from src.model.dit import DiT_2Cond

def main():
    cfg = json.load(open("src/train/configs/v21_skelnet_200k.json", encoding="utf-8"))
    model = DiT_2Cond(
        num_calligraphers=cfg["num_calligraphers"],
        condition_fusion=cfg["condition_fusion"],
        callig_embed_dim=cfg["callig_embed_dim"],
        deform_skel=cfg["deform_skel"],
        deform_width=cfg["deform_width"],
        deform_max_off=cfg["deform_max_off"],
        deform_coarse=cfg["deform_coarse"],
        deform_grid=cfg["deform_grid"],
        residual=cfg["residual"],
        res_cap=cfg["res_cap"],
        stroke_mod=cfg["stroke_mod"],
        stroke_cap=cfg["stroke_cap"],
        gate_radius=cfg["gate_radius"],
        deform_dt_ch=cfg.get("deform_dt_ch", 1),
        deform_ckpt=cfg["deform_ckpt"]
    )
    print("[smoke] Model build & ckpt load SUCCESS!")
    print(f"[smoke] deform_skel params: {sum(p.numel() for p in model.deform_skel.parameters()):,}")

if __name__ == "__main__":
    main()
