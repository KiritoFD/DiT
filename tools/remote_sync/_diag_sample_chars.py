# -*- coding: utf-8 -*-
"""诊断: 用 v3b checkpoint 对 kailishu_eval 前5样本采样, 判断每样本生成什么字。
方法: 对每个样本, 用其 (callig, glyph) + 标准字形g 采样生成;
  再把生成图与 5 个候选标准字形(昌/鼎/商/也/刻)的骨架对比 IoU,
  看生成的是否匹配该样本声明的字, 还是收敛到别的字。
"""
import os, sys, glob
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.environ["XFORMERS_DISABLED"] = "1"
import csv, numpy as np, torch
from PIL import Image, ImageDraw, ImageFont
from diffusers.models import AutoencoderKL
from diffusion import create_diffusion
from models import DiT_2Cond_models
from lora import inject_lora
from download import find_model
import json

ckpt = sorted(glob.glob("5script/results/v3b_xl_glyphcond/*/checkpoints/0015000.pt"))[-1]
print("ckpt:", ckpt)
ck = torch.load(ckpt, map_location="cpu", weights_only=False)
args = ck["args"]
model = DiT_2Cond_models["DiT-2Cond-XL/2"](
    input_size=32, num_calligraphers=args.num_calligraphers, num_characters=args.num_characters,
    use_checkpoint=False, condition_fusion=args.condition_fusion,
    callig_embed_dim=args.callig_embed_dim, char_embed_dim=args.char_embed_dim,
    cond_drop_all_prob=args.cond_drop_all_prob, cond_drop_one_prob=args.cond_drop_one_prob,
    use_glyph_cond=True, glyph_scale_init=args.glyph_scale_init)
pre = find_model("pretrained_models/DiT-XL-2-256x256.pt")
pre2={k:v for k,v in pre.items() if not k.startswith(("y_embedder","cond_fusion","callig_proj","char_proj","y_callig","y_char","skel_head","glyph_embedder"))}
model.load_state_dict(pre2, strict=False)
inject_lora(model, r=args.lora_r, lora_alpha=args.lora_alpha, target=args.lora_target)
w = ck.get("ema") or ck.get("delta")
missing, unexpected = model.load_state_dict(w, strict=False)
print(f"load ema: missing={len(missing)} unexpected={len(unexpected)}")
model = model.to("cuda").eval()

vae = AutoencoderKL.from_pretrained(args.vae_path).to("cuda").eval()
ddim = create_diffusion(str(50))

# 候选字 + 标准字形渲染(本地要字形? 用远程没有字体)。改用远程 GT skeleton 库? 
# 这里用标准字形 latent 反解码成图对比太麻烦。改用: 采样后 decode, 和 GT 骨架对比。
# 先只采样, 输出 decode 后图像 stock 保存, 再用骨架对比。
rows = list(csv.DictReader(open("kailishu_eval.csv", encoding="utf-8")))[:5]
torch.manual_seed(0)
out_imgs = []
for i, r in enumerate(rows):
    glyph = int(r["glyph_id"]); callig = int(r["calligrapher_id"])
    img_id = os.path.basename(r["image_path"])[:-4]
    char = r["character"]
    # g from std_glyph_latent
    book = {"0":"kai","4":"li"}[r["script_id"]]
    g = torch.from_numpy(np.load(f"std_glyph_latent/{book}/U+{ord(char):05X}.npy")).unsqueeze(0).to("cuda")
    yc = torch.tensor([callig], device="cuda")
    yh = torch.tensor([glyph], device="cuda")
    z = torch.randn(1,4,32,32, device="cuda")
    mk = dict(y_callig=yc, y_char=yh, cfg_scale=4.0, g=g)
    # forward_with_cfg 期望 g 匹配2倍batch(内部 duplicat); 传 None 或复制
    # eval_auto 里 g 是整批, forward_with_cfg 内部 duplicat x 不 duplicat g -> 需处理
    # 直接调 model.forward_with_cfg(x, t, yc, yh, cfg, g) 内部 g 不 duplicat, 冲突。
    # 改用 ddim 对 model 直接 forward(无cfg) + 手动组合太复杂。
    # 简化: 用 eval_auto 的路径(它处理好 g duplication)。这里直接用 forward_with_cfg 但 g 传 2份。
    g2 = torch.cat([g, g], dim=0)
    mk = dict(y_callig=torch.cat([yc,yc],0), y_char=torch.cat([yh,yh],0), cfg_scale=4.0, g=g2)
    samples = ddim.ddim_sample_loop(model.forward_with_cfg, z.shape, z, clip_denoised=False,
                                    model_kwargs=mk, progress=False, device="cuda")
    dec = vae.decode(samples/0.18215).sample[0]
    dec_pil = ((dec.clamp(-1,1)+1)/2).permute(1,2,0).detach().cpu().numpy()
    Image.fromarray((dec_pil*255).clip(0,255).astype(np.uint8)).save(f"/tmp/gen_{i}_{char}.png")
    out_imgs.append((char, img_id))
    print(f"[{i}] char={char} id={img_id} -> /tmp/gen_{i}_{char}.png")
print("sample done")
with open("/tmp/sample_report.json","w",encoding="utf-8") as f:
    json.dump([{"idx":i,"char":c,"id":im} for i,(c,im) in enumerate(out_imgs)], f)
