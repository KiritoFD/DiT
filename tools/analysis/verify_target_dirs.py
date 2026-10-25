from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

targets = {
    "Base 55k": ROOT / "exp_img/unet/u100_base_55k",
    "DiC 50k": ROOT / "exp_img/unet/u100_dic_b768_50k",
    "Arm 2 50k": ROOT / "assets/results/unet_ablation_50k/arm2_writer_film_dec_b768/20261019-105029-Unet-flow/eval_samples_ctrl/step0050000/eval200fix",
    "Exp A 50k": ROOT / "assets/results/unet_decoupled_114m/armA_writer_decoupled_50k/20261019-191851-Unet-flow/eval_samples_ctrl/step0050000/eval200fix",
    "Exp A 70k": ROOT / "assets/results/unet_decoupled_114m/armA_writer_decoupled_70k/20261020-065018-Unet-flow/eval_samples_ctrl/step0070000/eval200fix",
}

for name, path in targets.items():
    if not path.exists():
        print(f"{name:12s}: NOT FOUND: {path}")
    else:
        cnt = len(list(path.glob("g*.png")))
        print(f"{name:12s}: FOUND {cnt} images at {path.relative_to(ROOT)}")
