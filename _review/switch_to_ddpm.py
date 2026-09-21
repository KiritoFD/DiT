import io
import os

os.chdir("G:/GitHub/DiT")
p = "ref/moyi/train_moyun_ours.py"
s = io.open(p, encoding="utf-8").read()

# 1) import
s = s.replace("from utils.Sampler.RF import RF  # noqa: E402",
              "from utils.REPA_diffusion import create_diffusion  # noqa: E402")

# 2) 构造
s = s.replace(
    "    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0)\n"
    "    diffusion = RF(without_t=(a.without_t == 1))",
    "    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0)\n"
    "    # ★ 按 ref 生产脚本（_ful.sh -> train_moyun2_diffusion_repa.py）用 DDPM\n"
    "    diffusion = create_diffusion(\n"
    "        timestep_respacing=\"\",\n"
    "        learn_sigma=True,\n"
    "        use_black_white_mse_loss=False,\n"
    "        use_grey_mse_loss=False)")

# 3) 训练步
old_step = """            loss = diffusion.forward(model, x, feature_dict=fdict,
                                     y=y, stroke=stroke)[0]"""
new_step = """            # ★ DDPM：随机采 t，training_losses(model, x, t, feature_dict, model_kwargs)
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],),
                              device=dev)
            fdict["_num_classes"] = a.num_classes
            loss_dict = diffusion.training_losses(
                model, x, t, fdict, {"y": y, "stroke": stroke})
            loss = loss_dict["loss"].mean()"""
if old_step in s:
    s = s.replace(old_step, new_step)
    print("  ✓ 训练步改为 DDPM")
else:
    print("  ⚠ 没找到训练步")

# 4) 模型构造加 learn_sigma=True（Moyun 默认就是 True，显式写出）
s = s.replace(
    "        learn_sigma=False, if_rope=bool(a.if_rope)).to(dev)",
    "        learn_sigma=True, if_rope=bool(a.if_rope)).to(dev)")

io.open(p, "w", encoding="utf-8").write(s)
import ast  # noqa: E402

ast.parse(s)
print("  SYNTAX OK")
