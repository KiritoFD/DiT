# -*- coding: utf-8 -*-
"""FontDiffuser 256px patch (idempotent).

The StyleRSI up-block pairs the style-content feature with a FIXED negative
index (`style_structure_features[-upblock_index-2]`), which is only valid for
the officially trained 96px config. At 256px the content encoder emits a
deeper feature pyramid, so the fixed index grabs a 16x16x512 feature while
the block runs at 64x64 -> GroupNorm channel crash.

Fix: select the style-content feature by MATCHING SPATIAL SIZE with the
resolution feature (the semantics the 96px indices implicitly implement).
Channels stay {128, 64} exactly as at 96px, so the architecture is unchanged.
"""
import os

REPO = "/root/Workspace/xy/DiT/baseline/FontDiffuser"
P = os.path.join(REPO, "src", "modules", "unet_blocks.py")

OLD = "        style_content_feat = style_structure_features[-self.upblock_index-2]"
NEW = """        # top10@256: pair the style-content feature by spatial size
        # (the original fixed negative index is only valid at 96px)
        _res_hw = res_hidden_states_tuple[-1].shape[-1]
        style_content_feat = next(f for f in style_structure_features
                                  if f.shape[-1] == _res_hw and f.shape[-2] == _res_hw)"""


def main():
    src = open(P, encoding="utf-8").read()
    if "pair the style-content feature by spatial size" in src:
        print("already patched")
        return
    assert OLD in src, "expected original line not found"
    open(P, "w", encoding="utf-8").write(src.replace(OLD, NEW))
    print("patched StyleRSI pairing for 256px")


if __name__ == "__main__":
    main()
