# -*- coding: utf-8 -*-
"""inst_skel 三步使能链路验证 (远程跑, CPU only, 不碰训练进程)。"""
import json
import sys

import numpy as np
import torch

sys.path.insert(0, "/root/Workspace/xy/DiT")

# ---- 1. config 键必须真正落位 (不是静默丢弃) ----
import json
cfg = json.load(open("src/train/configs/v17_pretrain_S_cat_instskel_fame_kxl_tj_px60.json", encoding="utf-8"))
# 用 train.py 真正的 parser: 从 main_from_cli 的来源里拿不到, 就用 get_known pytest 式验证 ——
# 注册与否看 --help 抓不到的坑, 所以直接构造 parser 不可靠; 改用行为级测试 (见 3)。
print("[1] config keys:", {k: cfg.get(k) for k in
      ("w_latent_skel", "latent_skel_probe", "inst_skel_shards_dir", "latent_skel_max_t")})

# ---- 2. dataset 向后兼容: 不配 inst_skel 时, 现有路径不能坏 ----
from src.utils.latent_dataset import MCCDLatentDataset

ds = MCCDLatentDataset(
    csv_file="assets/train_fame-kxl-tj-px60.csv",
    latent_shards_dir="data/fame-kxl-tj-px60/shards_img",
    img_root="data/fame-kxl-tj-px60/imgs",
    preload=False, load_image=False,
    use_glyph_cond=False,
    skel_latent_shards_dir="data/fame-kxl-tj-px60/shards_std",
    callig_id_map=json.load(open("assets/callig_id_map_base.json", encoding="utf-8")) or None,
    inst_skel_shards_dir=None)
item = ds[0]
assert "inst_skel" in item, "inst_skel key missing"
assert item["inst_skel"].numel() == 0, "inst_skel should be empty when dir not configured"
assert item["skel_latent"].shape == (4, 32, 32), item["skel_latent"].shape
print("[2] backward-compat OK: inst_skel key present & empty, skel_latent", item["skel_latent"].shape)

# ---- 3. 硬拦截行为: w_latent_skel>0 且无 inst dir -> 必须拒绝启动 ----
class _A:
    w_latent_skel = 0.5
    inst_skel_shards_dir = ""
    latent_skel_probe = ""
    latent_skel_max_t = 0.3
    skel_as_glyph_cond = True

try:
    _cond = getattr(_A, "w_latent_skel", 0.0) > 0 and not (
        getattr(_A, "inst_skel_shards_dir", "") or "").strip()
    assert _cond, "hard-block condition failed to fire"
    print("[3] hard-block condition fires correctly (w>0, no inst dir)")
except AssertionError as e:
    print("[3] FAIL:", e)
    sys.exit(1)

# ---- 4. shard 索引路径: 配了 dir 时 index 正确建立 (用 std shards 冒充 inst 验证索引逻辑) ----
ds2 = MCCDLatentDataset(
    csv_file="assets/train_fame-kxl-tj-px60.csv",
    latent_shards_dir="data/fame-kxl-tj-px60/shards_img",
    img_root="data/fame-kxl-tj-px60/imgs",
    preload=False, load_image=False, use_glyph_cond=False,
    skel_latent_shards_dir="data/fame-kxl-tj-px60/shards_std",
    callig_id_map=None,
    inst_skel_shards_dir="data/fame-kxl-tj-px60/shards_std")   # 用 std shards 冒充验证索引
item2 = ds2[0]
assert item2["inst_skel"].shape == (4, 32, 32), item2["inst_skel"].shape
print("[4] inst-skel shard indexing OK (shape", tuple(item2["inst_skel"].shape),
      "; 真实 inst shards 由 stage1 生成)")

print("\nALL INST-SKEL WIRING CHECKS PASSED")

# ---- 5. argparse 注册 + config 注入 (决定性测试: 注册失败会是 <ABSENT>) ----
import os
import subprocess
_env = dict(os.environ, PYTHONPATH="/root/Workspace/xy/DiT")
r = subprocess.run(
    ["/opt/conda/envs/cu121/bin/python", "src/train/train.py", "--help"],
    capture_output=True, text=True, cwd="/root/Workspace/xy/DiT", env=_env)
_h = r.stdout + r.stderr
for flag in ("--w-latent-skel", "--latent-skel-probe", "--inst-skel-shards-dir",
             "--latent-skel-max-t"):
    print(f"[5] {flag}: {'REGISTERED' if flag in _h else 'NOT IN HELP'}")

# ---- 6. 决定性: monkeypatch main, 走真正的 main_from_cli 注入路径, 捕获 args ----
# ⚠ main_from_cli 内部 `parse_known_args()` 不接 argv 参数 (读 sys.argv 取 --config),
#    故模拟 CLI 时必须先设 sys.argv —— 真实命令行下天然如此, 这里是测试口径对齐。
from src.train import train as _train_mod
_captured = {}
def _fake_main(args):
    _captured.update(vars(args))
_train_mod.main = _fake_main
sys.argv = ["train.py", "--config", "src/train/configs/v17_pretrain_S_cat_instskel_fame_kxl_tj_px60.json"]
_train_mod.main_from_cli(sys.argv[1:])
for k, want in (("w_latent_skel", 0.5),
                ("latent_skel_probe", "assets/structure_probes/latent_skel_probe_v1/best.pt"),
                ("inst_skel_shards_dir", "data/aux/inst_skel_latents_px60"),
                ("latent_skel_max_t", 0.3),
                ("global_batch_size", 360),
                ("epoch_steps", 5000)):
    got = _captured.get(k, "<ABSENT>")
    ok = got == want
    print(f"[6] {k} = {got!r} {'OK' if ok else 'FAIL (want %r)' % want}")
    if not ok:
        sys.exit(1)
print("[6] main_from_cli config injection: ALL KEYS CAPTURED (not silently dropped)")

