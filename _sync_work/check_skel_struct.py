"""校验实例骨架结构 loss 的接线: 语法 / 参数注册 / 配置落位 / 前向与梯度。"""
import ast, os, subprocess, sys

sys.path.insert(0, "/root/Workspace/xy/DiT")

for p in ["src/train/latent_structure.py", "src/train/train.py"]:
    try:
        ast.parse(open(p, encoding="utf-8").read())
        print("SYNTAX OK  ", p)
    except SyntaxError as e:
        print("SYNTAX FAIL", p, e.lineno, e.msg)

out = subprocess.run(["/opt/conda/envs/cu121/bin/python", "src/train/train.py", "--help"],
                     capture_output=True, text=True).stdout
for k in ["--w-latent-skel", "--latent-skel-probe", "--latent-skel-max-t"]:
    print("ARG", k, "OK" if k in out else "MISSING")

# --- 配置落位 (config 键未注册会被静默丢弃, 已踩 5 次) ---
import importlib.util
spec = importlib.util.spec_from_file_location("T", "src/train/train.py")
T = importlib.util.module_from_spec(spec)
spec.loader.exec_module(T)

captured = {}
T.main = lambda a=None: captured.update(vars(a))
sys.argv = ["train.py", "--config", "src/train/configs/v12_pretrain_S_cat_fame_kxl_tj_px60.json"]
try:
    T.main_from_cli()
except Exception as e:
    print("cli err:", e)
for k in ["w_latent_skel", "latent_skel_probe", "latent_skel_max_t"]:
    print("CFG", k, "=", captured.get(k, "<ABSENT>"))

# --- 模块单元测: 冻结 probe + 门控 + 梯度 ---
import torch
from src.train.latent_structure import LatentSkelProbe, LatentSkelStructureLoss

dev = "cuda" if torch.cuda.is_available() else "cpu"
probe = LatentSkelProbe(in_channels=4, out_channels=4, width=64, depth=3).to(dev)
n_before = sum(p.numel() for p in probe.parameters())
fn = LatentSkelStructureLoss(probe=probe, max_t=0.3)
n_trainable = sum(p.numel() for p in fn.parameters() if p.requires_grad)
print(f"PROBE params={n_before:,}  trainable_after_freeze={n_trainable} (应为 0)")

N = 8
pred = torch.randn(N, 4, 32, 32, device=dev, requires_grad=True)
tgt = torch.randn(N, 4, 32, 32, device=dev)
t = torch.rand(N, device=dev)                       # 全在 (0.3,1] -> 应恒为 0
t = 0.31 + 0.69 * t
print("GATE all-t>max_t  loss =", float(fn(pred, tgt, t)), "(应为 0.0)")

t2 = torch.rand(N, device=dev) * 0.3                # 全 <= 0.3 -> 应为正
l = fn(pred, tgt, t2)
print("GATE all-t<=max_t loss =", float(l), "(应 > 0)")
l.backward()
print("GRAD to pred_xstart =", float(pred.grad.abs().sum()), "(应 > 0: 梯度能回传主干)")
print("GRAD to probe       =", float(sum(p.grad.abs().sum() for p in probe.parameters() if p.grad is not None)),
      "(应为 0: probe 冻结)")

# 通道不匹配应显式报错, 不能静默
try:
    fn(pred, torch.randn(N, 8, 32, 32, device=dev), t2)
    print("CHANNEL-GUARD FAIL: 未报错")
except RuntimeError as e:
    print("CHANNEL-GUARD OK:", str(e)[:70])
