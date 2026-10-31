"""2026-10-31 [A/B/C] common.py rate() 字段映射修复。

rate() 读 'lr_'+kind → 'lr_style_representation'/'lr_style_injections',
而 Config 字段是 lr_style_repr/lr_style_inj ⇒ style 两组一直落到 legacy 1e-4
(P3A ckpt initial_lr 实证)。加显式映射, 使 spec 的 style LR 真正生效。
"""
from pathlib import Path
import py_compile

P=Path('/root/Workspace/xy/DiT/tools/style_matrix/runtime/common.py')
src=P.read_text(encoding='utf-8')

old="""    def rate(kind,legacy):
        if scratch:return config.lr
        explicit=getattr(config,'lr_'+kind,0.) or 0.
        return explicit or legacy"""
new="""    def rate(kind,legacy):
        if scratch:return config.lr
        # 2026-10-31 [A/B/C]: kind 名与 Config 字段名不一致 (lr_style_repr/lr_style_inj),
        # 旧写法 getattr(config,'lr_'+kind) 永远 miss ⇒ style 组静默落到 legacy 1e-4。
        field={'backbone':'lr_backbone','style_representation':'lr_style_repr',
               'style_injections':'lr_style_inj'}.get(kind,'lr_'+kind)
        explicit=getattr(config,field,0.) or 0.
        return explicit or legacy"""
assert src.count(old)==1,'pattern not found'
P.write_text(src.replace(old,new),encoding='utf-8')
py_compile.compile(str(P),doraise=True)
print('RATE FIXED OK')
