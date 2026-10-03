# -*- coding: utf-8 -*-
"""清理之前错误插入的 _MID_SKEL_SRC 定义, 再用唯一锚点在正确位置插入。"""
import io

P = "/root/Workspace/xy/DiT/src/train/train.py"
src = "\n".join(io.open(P, encoding="utf-8", newline="").read().split("\r\n"))
lines = src.split("\n")

# 1) 清掉所有历史误插
clean = []
for l in lines:
    if ("_MID_SKEL_SRC = str(" in l or "_MID_SKEL_WARNED = False" in l
            or "中程载体来源开关" in l):
        continue
    clean.append(l)
src = "\n".join(clean)
print("cleaned:", len(lines) - len(clean), "lines")

# 2) 锚点插入 (MidStructureLoss 构造收尾 + 紧随的 if rank == 0)
ANCHOR = ("            latent_channels=int(getattr(args, 'latent_channels', 4)))\n"
          "        if rank == 0:")
assert src.count(ANCHOR) == 1, f"anchor count={src.count(ANCHOR)}"
NEW = ("            latent_channels=int(getattr(args, 'latent_channels', 4)))\n"
       "        # \u2605 \u4e2d\u7a0b\u8f7d\u4f53\u6765\u6e90\u5f00\u5173 (\u5faa\u73af\u5916\u53ea\u8bfb\u4e00\u6b21, \u907f\u514d\u6bcf\u6b65 getattr)\u3002\n"
       "        #   \u5fc5\u987b\u5728 `if rank == 0:` **\u4e4b\u524d** \u5b9a\u4e49 \u2014\u2014 \u5426\u5219 rank!=0 \u65f6\u8fd9\u4e24\u4e2a\u540d\u5b57\u4e0d\u5b58\u5728, \u8c03\u7528\u5904 NameError\u3002\n"
       "        _MID_SKEL_SRC = str(getattr(args, 'std_mid_skel_src', 'g') or 'g')\n"
       "        _MID_SKEL_WARNED = False\n"
       "        if rank == 0:")
src = src.replace(ANCHOR, NEW, 1)

# 3) 调用处 (可能已被 patch1 改过, 先判断)
OLD2 = ("                    _lc = int(getattr(args, 'latent_channels', 4))\n"
        "                    _gt = x_latent[:, :_lc]\n"
        "                    _gsk = model_kwargs.get('g', None)\n"
        "                    loss_std_mid = _mid_struct_loss(\n"
        "                        pred_xstart_latent, _gt, _gsk, t.to(device))")
NEW2 = ("                    _lc = int(getattr(args, 'latent_channels', 4))\n"
        "                    _gt = x_latent[:, :_lc]\n"
        "                    # \u2605 \u4e2d\u7a0b\u8f7d\u4f53\u6765\u6e90\u3002'g' = \u6807\u51c6\u5b57\u5f62\u7ec6\u9aa8\u67b6, \u4f46\u5b83\u540c\u65f6\u5c31\u662f\u6a21\u578b\u7684 glyph \u6761\u4ef6\n"
        "                    #   \u2014\u2014 \u62ff\u6761\u4ef6\u5f53 target = \u7eaf\u91cd\u590d\u6761\u4ef6\u4fe1\u53f7, loss \u4f1a\u6b63\u5e38\u4e0b\u964d\u4f46\u4ec0\u4e48\u4e5f\u6ca1\u5b66\u5230\u3002\n"
        "                    #   'inst' = \u5b9e\u4f8b\u7c97\u9aa8\u67b6 (GT \u56fe -> skeletonize -> 20px -> VAE encode),\n"
        "                    #   \u4e0e\u6761\u4ef6 g \u5b8c\u5168\u89e3\u8026\u3002\u5b9e\u6d4b\u6b8b\u5dee: g=0.486 -> inst20px=0.145\u3002\n"
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
if "if _MID_SKEL_SRC == 'inst':" in src:
    print("call-site already patched")
elif src.count(OLD2) == 1:
    src = src.replace(OLD2, NEW2, 1)
    print("call-site patched")
else:
    raise SystemExit("call-site anchor not found & not already patched")

io.open(P, "w", encoding="utf-8", newline="").write("\r\n".join(src.split("\n")))
print("written")
