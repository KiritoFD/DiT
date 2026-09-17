# -*- coding: utf-8 -*-
"""verify_v9c_smoke.py — v9c 联训 smoke 的 ckpt 正确性校验.

用法 (远程):
  /opt/conda/envs/cu121/bin/python tools/diag/verify_v9c_smoke.py [smoke_ckpt.pt] [v9a_ckpt.pt]

校验点:
  1. ckpt 结构: model.*/ema_model.*(main) + ctrl.*/ema(ctrl) + train_steps
  2. optimizer 双参数组且 main lr = 0.3 × ctrl lr
  3. char 表 bit-exact (freeze_char_table 豁免生效, DINO 语义锚未漂移)
  4. 其余 main 权重已更新 (联训生效)
  5. 注入权重离开零点 (梯度到达 injections, 条件通路活)
  6. 重载: ctrl/main keys 零缺失; skel 条件前向 vs 无条件前向输出不同
"""
import os, sys, glob, json
import torch

ROOT = "/root/Workspace/xy/DiT"
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.stdout.reconfigure(encoding="utf-8")

SMOKE = sys.argv[1] if len(sys.argv) > 1 else sorted(
    glob.glob("assets/results/_smoke_v9c/*/checkpoints/0000005.pt"))[-1]
V9A = sys.argv[2] if len(sys.argv) > 2 else sorted(
    glob.glob("assets/results/v9a_repa_pretrain/*/checkpoints/eval_auto_*.json"))
V9A = sys.argv[2] if len(sys.argv) > 2 else None
if V9A is None:
    best, best_step, rd = -1, "", None
    for f in glob.glob("assets/results/v9a_repa_pretrain/*/checkpoints/eval_auto_*.json"):
        d = json.load(open(f, encoding="utf-8"))
        if d.get("ssim", -1) > best:
            best, best_step = d["ssim"], os.path.basename(f).replace("eval_auto_", "").replace(".json", "")
            rd = os.path.dirname(f)
    V9A = os.path.join(rd, f"{best_step}.pt")

print(f"smoke ckpt = {SMOKE}\nv9a ckpt   = {V9A}\n")
_ok = True


def check(name, cond, detail=""):
    global _ok
    _ok = _ok and bool(cond)
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


def _strip(sd):
    return {(k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
            for k, v in sd.items()}


ck = torch.load(SMOKE, map_location="cpu", weights_only=False)
va = torch.load(V9A, map_location="cpu", weights_only=False)
# v9a (train.py) 的 ema 键也带 _orig_mod. 前缀, 统一剥离后对比
va_sd = _strip(va.get("ema") or va.get("model") or va)

mkeys = _strip(ck.get("model") or {})
ekeys = _strip(ck.get("ema_model") or {})
ckeys = _strip(ck.get("ctrl") or {})

check("train_steps == 5", ck.get("train_steps") == 5, str(ck.get("train_steps")))
check("model.* (main) 已存", len(mkeys) >= 150, f"{len(mkeys)} keys")
check("ema_model.* (main ema) 已存", len(ekeys) >= 150, f"{len(ekeys)} keys")
check("ctrl.* 已存", len(ckeys) >= 150, f"{len(ckeys)} keys")
check("ema(ctrl) 已存", len(_strip(ck.get("ema") or {})) >= 150,
      f"{len(_strip(ck.get('ema') or {}))} keys")

pg = (ck.get("optimizer") or {}).get("param_groups", [])
check("optimizer 双参数组", len(pg) == 2, f"{len(pg)} groups")
if len(pg) == 2:
    ratio = pg[1]["lr"] / max(pg[0]["lr"], 1e-15)
    check("main lr = 0.3 × ctrl lr", abs(ratio - 0.3) < 1e-5,
          f"ctrl={pg[0]['lr']:.3e} main={pg[1]['lr']:.3e} ratio={ratio:.4f}")

TBL = "y_char_embedder.embedding_table.weight"
tbl_a = va_sd.get(TBL)
tbl_s = mkeys.get(f"main.{TBL}")
tbl_e = ekeys.get(f"main.{TBL}")
check("char 表存在", tbl_a is not None and tbl_s is not None and tbl_e is not None)
if tbl_a is not None and tbl_s is not None:
    check("char 表 bit-exact (联训中未漂移)", torch.equal(tbl_a, tbl_s))
if tbl_a is not None and tbl_e is not None:
    check("char 表 ema bit-exact", torch.equal(tbl_a, tbl_e))

blk_keys = [k for k in mkeys if k.startswith("main.blocks.0.") and k.endswith(".weight")][:2]
if blk_keys:
    dmax = max((mkeys[k] - va_sd[k[len('main.'):]]).abs().max().item() for k in blk_keys)
    check("main block 权重已更新 (联训生效)", dmax > 0, f"max|Δ|={dmax:.3e}")
inj = [v for k, v in ckeys.items() if "injection" in k and k.endswith(".proj.weight")]
check("注入权重离开零点 (梯度到达)", any((v != 0).any().item() for v in inj),
      f"{len(inj)} 个 injection proj 张量")

# ---- 重载 + 条件通路前向 (CPU) ----
from src.model.legacy.controlnet import load_main_model, ControlNetDiT
main = load_main_model(
    "DiT-2Cond-S/2", V9A, device="cpu", num_calligraphers=1013, num_characters=35130,
    condition_fusion="factorized_add", callig_embed_dim=128, char_embed_dim=384,
    char_proj_mode="mlp", freeze_char_table=True)
c = ControlNetDiT(main, cond_in_channels=4, train_ctrl_only=True,
                  injection="modulate", null_cond="gaussian")
r1 = c.load_state_dict(ckeys, strict=False)
n_total = len(c.state_dict())
# 只载 ctrl 半边: unexpected 必须为 0; missing 应恰为模块另一半 (main, 318-159)
check("ctrl keys 重载: unexpected=0", len(r1.unexpected_keys) == 0
      and len(r1.missing_keys) == n_total - len(ckeys),
      f"missing={len(r1.missing_keys)} (期望 {n_total - len(ckeys)}) unexpected={len(r1.unexpected_keys)}")
r2 = c.load_state_dict(mkeys, strict=False)
check("main.* keys 重载: unexpected=0", len(r2.unexpected_keys) == 0
      and len(r2.missing_keys) == n_total - len(mkeys),
      f"missing={len(r2.missing_keys)} (期望 {n_total - len(mkeys)}) unexpected={len(r2.unexpected_keys)}")

c.eval()
x = torch.randn(1, 4, 32, 32)
t = torch.tensor([500.0])
yc = torch.tensor([601])
yh = torch.tensor([2035])
sk = torch.randn(1, 4, 32, 32)
with torch.no_grad():
    o_sk = c(x, t, yc, yh, cond=sk)
    o_no = c(x, t, yc, yh, cond=None)
d = (o_sk - o_no).abs().max().item()
check("skel 条件前向有响应", d > 0, f"max|Δout|={d:.3e}")

print("\n===", "ALL PASS" if _ok else "HAS FAILURES", "===")
sys.exit(0 if _ok else 1)
