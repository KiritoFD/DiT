# -*- coding: utf-8 -*-
"""给中程结构 loss 加 --std-mid-skel-src {g,inst}: 载体从 g(条件) 换成 inst_skel(实例粗骨架)。"""
import io

ROOT = "/root/Workspace/xy/DiT/"

# ── 1) cli.py: 新参数 ────────────────────────────────────────────────────────
P = ROOT + "src/train/cli.py"
src = "\n".join(io.open(P, encoding="utf-8", newline="").read().split("\r\n"))

OLD = ('    parser.add_argument("--std-mid-k-slope", type=float, default=2.0, dest="std_mid_k_slope",\n'
       '                        help="\u65e0\u6821\u51c6 json \u65f6\u9ed8\u8ba4 k(t)=slope\xb7t (latent px), \u4e0a\u9650 2\u3002")')
NEW = OLD + (
    '\n    parser.add_argument("--std-mid-skel-src", type=str, default="g",\n'
    '                        choices=["g", "inst"], dest="std_mid_skel_src",\n'
    '                        help="\u4e2d\u7a0b\u7ed3\u6784\u8f7d\u4f53\u6765\u6e90\u3002\n'
    '                             g    = \u6807\u51c6\u5b57\u5f62\u7ec6\u9aa8\u67b6 (\u5373\u6a21\u578b\u7684 glyph \u6761\u4ef6, \u9ed8\u8ba4, \u4fdd\u6301\u73b0\u72b6);\n'
    '                             inst = \u5b9e\u4f8b\u7c97\u9aa8\u67b6 (GT \u56fe\u6d3e\u751f, \u4e0e\u6761\u4ef6 g \u89e3\u8026)\u3002\n'
    '                             \u5b9e\u6d4b: g \u6b8b\u5dee 0.486, \u800c\u4ece GT \u56fe\u63d0\u9aa8\u67b6\u52a0\u7c97\u5230 20px \u540e\u6b8b\u5dee\u964d\u5230 0.145\n'
    '                             (\u63a2\u9488\u626b\u63cf\u62d0\u70b9 20px; GT \u7b14\u753b\u5bbd\u5ea6\u4e2d\u4f4d\u6570 19.4px \u72ec\u7acb\u5370\u8bc1)\u3002\n'
    '                             \u9009 inst \u65f6\u5fc5\u987b\u914d\u7f6e --inst-skel-shards-dir, \u4e14 k_slope\u5e94\u8bbe 0\n'
    '                             (\u9aa8\u67b6\u5df2\u662f\u76ee\u6807\u5bbd\u5ea6, \u4e0d\u518d\u5728\u7ebf\u81a8\u80c0)\u3002")')
assert OLD in src, "cli OLD not found"
src = src.replace(OLD, NEW)
io.open(P, "w", encoding="utf-8", newline="").write("\r\n".join(src.split("\n")))
print("cli.py patched")

# ── 2) train.py: 调用处按 src 取骨架 ─────────────────────────────────────────
P = ROOT + "src/train/train.py"
src = "\n".join(io.open(P, encoding="utf-8", newline="").read().split("\r\n"))

OLD2 = ("                    _lc = int(getattr(args, 'latent_channels', 4))\n"
        "                    _gt = x_latent[:, :_lc]\n"
        "                    _gsk = model_kwargs.get('g', None)\n"
        "                    loss_std_mid = _mid_struct_loss(\n"
        "                        pred_xstart_latent, _gt, _gsk, t.to(device))")
NEW2 = ("                    _lc = int(getattr(args, 'latent_channels', 4))\n"
        "                    _gt = x_latent[:, :_lc]\n"
        "                    # \u2605 \u4e2d\u7a0b\u8f7d\u4f53\u6765\u6e90\u3002'g' = \u6807\u51c6\u5b57\u5f62\u7ec6\u9aa8\u67b6, \u4f46\u5b83\u540c\u65f6\u5c31\u662f\u6a21\u578b\u7684 glyph \u6761\u4ef6 \u2014\u2014\n"
        "                    #   \u62ff\u6761\u4ef6\u5f53 target = \u7eaf\u91cd\u590d\u6761\u4ef6\u4fe1\u53f7, loss \u4f1a\u6b63\u5e38\u4e0b\u964d\u4f46\u4ec0\u4e48\u4e5f\u6ca1\u5b66\u5230\u3002\n"
        "                    #   'inst' = \u5b9e\u4f8b\u7c97\u9aa8\u67b6 (GT \u56fe -> skeletonize -> 20px -> VAE encode),\n"
        "                    #   \u4e0e\u6761\u4ef6 g \u5b8c\u5168\u89e3\u8026, \u4e14\u5df2\u7ecf\u662f\u76ee\u6807\u5bbd\u5ea6 (\u6545 k_slope \u5e94\u4e3a 0, \u4e0d\u518d\u5728\u7ebf\u81a8\u80c0)\u3002\n"
        "                    if _MID_SKEL_SRC == 'inst':\n"
        "                        _gsk = batch.get('inst_skel', None)\n"
        "                        if _gsk is not None and _gsk.numel() == 0:\n"
        "                            _gsk = None\n"
        "                        if _gsk is None and not _MID_SKEL_WARNED:\n"
        "                            _MID_SKEL_WARNED = True\n"
        "                            logger.warning(\n"
        "                                \"[std_mid] std_mid_skel_src='inst' \u4f46 batch['inst_skel'] \u4e3a\u7a7a \"\n"
        "                                \"(\u9700 --inst-skel-shards-dir \u6307\u5411\u5b9e\u4f8b\u7c97\u9aa8\u67b6 shards) \"\n"
        "                                \"-> \u672c\u9879\u9759\u9ed8\u5931\u6548\u3002\u8fd9\u7c7b\u9759\u9ed8\u5931\u6548\u5df2\u8e29\u8fc7\u591a\u6b21, \u6545\u53ea\u5728\u9996\u6b21\u544a\u8b66\u3002\")\n"
        "                    else:\n"
        "                        _gsk = model_kwargs.get('g', None)\n"
        "                    loss_std_mid = _mid_struct_loss(\n"
        "                        pred_xstart_latent, _gt, _gsk, t.to(device))")
assert OLD2 in src, "train OLD2 not found"
src = src.replace(OLD2, NEW2)

# 3) \u5faa\u73af\u5916\u5b9a\u4e49 _MID_SKEL_SRC / _MID_SKEL_WARNED (\u653e\u5728 MidStructureLoss \u6784\u9020\u5904)
OLD3 = ("        logger.info(f\"[std_mid] MidStructureLoss carrier={_mid_struct_loss.carrier} \"")
NEW3 = ("    # \u4e2d\u7a0b\u8f7d\u4f53\u6765\u6e90\u5f00\u5173 (\u5faa\u73af\u5916\u53ea\u8bfb\u4e00\u6b21, \u907f\u514d\u6bcf\u6b65 getattr)\n"
        "    _MID_SKEL_SRC = str(getattr(args, 'std_mid_skel_src', 'g') or 'g')\n"
        "    _MID_SKEL_WARNED = False\n"
        + OLD3)
assert OLD3 in src, "train OLD3 not found"
src = src.replace(OLD3, NEW3, 1)

io.open(P, "w", encoding="utf-8", newline="").write("\r\n".join(src.split("\n")))
print("train.py patched")
