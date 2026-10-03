# -*- coding: utf-8 -*-
"""校准脚本 v4: 加 --init-random (随机初始化基线, 判定探针判别力) 与 --lp (低通尺度扫描)。"""
import io

P = "/root/Workspace/xy/DiT/tools/calibrate_mid_structure.py"
raw = io.open(P, encoding="utf-8", newline="").read()
src = "\n".join(raw.split("\r\n"))

REPS = [
    # 1) 新参数
    ('    ap.add_argument("--device", default="cuda")',
     '    ap.add_argument("--device", default="cuda")\n'
     '    # \u2605 \u968f\u673a\u521d\u59cb\u5316\u57fa\u7ebf: \u4e0d\u52a0\u8f7d ckpt \u6743\u91cd\u3002\u7528\u6765\u5224\u5b9a\u63a2\u9488\u7684**\u5224\u522b\u529b** \u2014\u2014\n'
     '    #   \u82e5\u968f\u673a\u6a21\u578b\u6b8b\u5dee\u4e5f\u53ea\u6709 ~0.003 (\u4e0e 100k \u540c\u91cf\u7ea7), \u8bf4\u660e\u8fd9\u4e2a\u5ea6\u91cf\u65e0\u6cd5\u533a\u5206\u597d\u574f\u6a21\u578b,\n'
     '    #   \u4e4b\u524d"\u5df2\u7ecf\u5b66\u5f88\u597d"\u7684\u7ed3\u8bba\u4f5c\u5e9f\u3002\n'
     '    ap.add_argument("--init-random", action="store_true")\n'
     '    ap.add_argument("--lp", type=int, default=2)'),
    # 2) \u6743\u91cd\u52a0\u8f7d\u6539\u4e3a\u6761\u4ef6\u5f0f
    ('    model = build_model_from_args(_ns, device=dev)\n'
     '    model.y_callig_embedder.freeze_table()   # \u62c6\u51fa null_embed, \u5bf9\u9f50 ckpt key\n'
     '    model.load_state_dict(sd, strict=True)   # strict=True \u5f53\u62a4\u680f\n'
     '    model.eval()',
     '    model = build_model_from_args(_ns, device=dev)\n'
     '    model.y_callig_embedder.freeze_table()   # \u62c6\u51fa null_embed, \u5bf9\u9f50 ckpt key\n'
     '    if a.init_random:\n'
     '        print("[model] \u2605 --init-random: \u4fdd\u7559\u968f\u673a\u521d\u59cb\u5316\u6743\u91cd (\u63a2\u9488\u5224\u522b\u529b\u57fa\u7ebf)")\n'
     '    else:\n'
     '        model.load_state_dict(sd, strict=True)   # strict=True \u5f53\u62a4\u680f\n'
     '    model.eval()'),
    # 3) resid \u7528\u53ef\u914d\u7f6e\u7684\u4f4e\u901a\u5c3a\u5ea6
    ('        return float((lowpass(chan_norm(zp)) - lowpass(chan_norm(zt))).pow(2).mean())',
     '        return float((lowpass(chan_norm(zp), a.lp) - lowpass(chan_norm(zt), a.lp)).pow(2).mean())'),
]

for old, new in REPS:
    assert old in src, "NOT FOUND: " + old[:70]
    src = src.replace(old, new)

out = "\r\n".join(src.split("\n"))
io.open(P, "w", encoding="utf-8", newline="").write(out)
print("patched OK")
