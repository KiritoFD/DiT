"""给 16ch 条件补上 glyph_in_channels 开关 (幂等)。

缺口: 条件编码器 glyph_embedder 吃的是 g 骨架 latent, 通道数被硬编码成 4,
      latent_channels 改 16 后 x_embedder 变了、它没变 -> conv 尺寸失配。
做法: 新增 --glyph-latent-channels (默认 4, 老配置行为不变), 三处硬编码改成读它。
"""
import io
import os
import sys

ROOT = "/root/Workspace/xy/DiT"
os.chdir(ROOT)

EDITS = [
    # (文件, 原串, 新串)
    ("src/train/cli.py",
     '    parser.add_argument("--latent-channels", type=int, default=4,',
     '    parser.add_argument("--glyph-latent-channels", type=int, default=4,\n'
     '                        dest="glyph_latent_channels",\n'
     '                        help="条件 g 骨架 latent 的通道数 (随 VAE 变; 默认 4 = sd-vae)")\n'
     '    parser.add_argument("--latent-channels", type=int, default=4,'),
    ("src/train/train.py",
     "            glyph_in_channels=4,",
     "            glyph_in_channels=int(getattr(args, 'glyph_latent_channels', 4) or 4),"),
    ("src/train/configs/model_io.py",
     "        glyph_in_channels=4,",
     '        glyph_in_channels=gi("glyph_latent_channels", 4),'),
    ("src/eval/model_io.py",
     "        glyph_in_channels=4,",
     '        glyph_in_channels=gi("glyph_latent_channels", 4),'),
]

for path, old, new in EDITS:
    if not os.path.exists(path):
        print(f"[skip] {path} 不存在")
        continue
    s = io.open(path, encoding="utf-8").read()
    if new in s:
        print(f"[skip] {path} 已打过补丁")
        continue
    if old not in s:
        print(f"[FAIL] {path} 找不到锚点: {old!r}")
        continue
    s = s.replace(old, new, 1)
    io.open(path, "w", encoding="utf-8").write(s)
    print(f"[ok] {path}")
print("[done] patch_glyph_in_ch")
