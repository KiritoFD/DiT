# -*- coding: utf-8 -*-
"""校准脚本 v3: 修 g 取法 (dataset 里 'g' 是空张量, 真正骨架在 'skel_latent')。"""
import io

P = "/root/Workspace/xy/DiT/tools/calibrate_mid_structure.py"
raw = io.open(P, encoding="utf-8", newline="").read()
src = "\n".join(raw.split("\r\n"))

OLD = ('    g = batch.get("g", batch.get("skel_latent", None))\n'
       '    g = g.to(dev).float()[:, :int(_ns.latent_channels)] if g is not None and g.numel() else None')

NEW = ('    # \u2605 dataset \u540c\u65f6\u8fd4\u56de \'g\' (\u672a\u914d\u7f6e glyph/inst \u65f6\u662f\u7a7a\u5f20\u91cf (B,0)) \u4e0e\n'
       '    #   \'skel_latent\' (B,4,32,32) \u3002\u539f\u5199\u6cd5 batch.get("g", batch.get("skel_latent")) \u4f1a\n'
       '    #   \u4f18\u5148\u62ff\u5230\u7a7a\u7684 \'g\' -> g=None -> dilate \u5206\u652f\u6052 inf, \u88c1\u51b3\u5931\u771f\u3002\n'
       '    g = batch.get("skel_latent", None)\n'
       '    if g is None or g.numel() == 0:\n'
       '        g = batch.get("g", None)\n'
       '    g = g.to(dev).float()[:, :int(_ns.latent_channels)] if g is not None and g.numel() else None\n'
       '    _wants_g = bool(getattr(_ns, "skel_as_glyph_cond", False)) or \\\n'
       '        (float(getattr(_ns, "w_glyph_cond", 0) or 0) > 0) or \\\n'
       '        bool(getattr(_ns, "use_glyph_cond", False))\n'
       '    print(f"[cond] skel_as_glyph_cond={getattr(_ns, \'skel_as_glyph_cond\', None)} "\n'
       '          f"w_glyph_cond={getattr(_ns, \'w_glyph_cond\', None)} wants_g={_wants_g}")')

assert OLD in src, "OLD g-fetch not found"
src = src.replace(OLD, NEW)

OLD2 = ('            mk = dict(y_callig=yc, y_char=yh)\n'
        '            if g is not None:\n'
        '                mk["g"] = g')
NEW2 = ('            mk = dict(y_callig=yc, y_char=yh)\n'
        '            if g is not None and _wants_g:\n'
        '                mk["g"] = g')
assert OLD2 in src, "OLD2 mk not found"
src = src.replace(OLD2, NEW2)

out = "\r\n".join(src.split("\n"))
io.open(P, "w", encoding="utf-8", newline="").write(out)
print("patched OK")
