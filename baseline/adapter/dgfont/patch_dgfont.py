# -*- coding: utf-8 -*-
"""Idempotent patches to the DG-Font repo for top10 @ 256x256:
  1) att_to_use: hardcoded 400-class list -> range(output_k)
  2) install torchvision-based modulated_deform_conv (no CUDA build)
Run from anywhere:  python ../adapter/dgfont/patch_dgfont.py
"""
import os
import shutil

DIT = "/root/Workspace/xy/DiT"
REPO = os.path.join(DIT, "baseline", "DG-Font")
ADAPTER = os.path.dirname(os.path.abspath(__file__))


def main():
    # 1) att_to_use patch
    p = os.path.join(REPO, "main.py")
    src = open(p, encoding="utf-8").read()
    if "args.att_to_use = list(range(args.output_k))" not in src:
        marker = "    args.att_to_use = [0, 1, 2, 3"
        i = src.index(marker)
        j = src.index("\n", i)
        src = src[:i] + "    args.att_to_use = list(range(args.output_k))" + src[j:]
        open(p, "w", encoding="utf-8").write(src)
        print("patched main.py: att_to_use = range(output_k)")

    # 2) torchvision DCN
    dst = os.path.join(REPO, "modules", "modulated_deform_conv.py")
    bak = dst + ".orig_cuda"
    if not os.path.exists(bak):
        shutil.copyfile(dst, bak)
    shutil.copyfile(os.path.join(ADAPTER, "modulated_deform_conv.py"), dst)
    print("installed torchvision modulated_deform_conv.py (orig kept as .orig_cuda)")

    # 3) validateUN crash when val_num < val_batch: x_ref is repeated to
    #    val_batch while c_src only has val_num rows -> AdaIN param/batch
    #    mismatch. Clamp val_batch to val_num.
    vp = os.path.join(REPO, "validation", "validation.py")
    src = open(vp, encoding="utf-8").read()
    marker = "def validateUN(data_loader, networks, epoch, args, additional=None):"
    if "args.val_batch = min(args.val_batch, args.val_num)" not in src:
        i = src.index(marker)
        j = src.index("\n", i) + 1
        src = (src[:j] +
               "    # top10 adapter: avoid AdaIN param/batch mismatch when val_num < val_batch\n"
               "    args.val_batch = min(args.val_batch, args.val_num)\n" +
               src[j:])
        open(vp, "w", encoding="utf-8").write(src)
        print("patched validation.py: val_batch = min(val_batch, val_num)")

    # 5) optional torch.compile (DG_COMPILE=1). NOTE: checkpoints then carry
    #    _orig_mod.* key prefixes — sample_top10.py strips them on load; keep
    #    the flag consistent between resume runs.
    mp = os.path.join(REPO, "main.py")
    anchor = "        networks['G_EMA'] = Generator(args.img_size, args.sty_dim, use_sn=False)"
    src = open(mp, encoding="utf-8").read()
    if "DG_COMPILE" not in src:
        assert anchor in src
        block = (anchor + "\n"
                 "    if os.environ.get('DG_COMPILE') == '1':  # top10 adapter\n"
                 "        networks = {k: torch.compile(v) for k, v in networks.items()}")
        src = src.replace(anchor, block)
        open(mp, "w", encoding="utf-8").write(src)
        print("patched main.py: DG_COMPILE torch.compile hook")
    # 4) validateUN (FID over output_k x output_k reference pairs) runs every
    #    epoch in the original code — amortizable at 1000 iters/epoch, ruinous
    #    otherwise. Gate it to every 25 epochs / final epoch, with a
    #    DG_SKIP_VAL=1 env override for quick config confirmations.
    mp = os.path.join(REPO, "main.py")
    src = open(mp, encoding="utf-8").read()
    old_call = "        validationFunc(val_loader, networks, epoch, args, {'logger': logger})"
    new_gate = ("        if os.environ.get('DG_SKIP_VAL') != '1' and "
                "((epoch + 1) % 25 == 0 or epoch == args.epochs - 1):  "
                "# every 25 epochs (top10 adapter)\n"
                "            validationFunc(val_loader, networks, epoch, args, {'logger': logger})")
    if "DG_SKIP_VAL" in src:
        print("main.py validation gating already present")
    elif old_call in src:
        src = src.replace(old_call, new_gate)
        open(mp, "w", encoding="utf-8").write(src)
        print("patched main.py: validationFunc gated (every 25 epochs, DG_SKIP_VAL override)")
    else:
        # upgrade an earlier gate without the env override
        import re as _re
        src2 = _re.sub(
            r"        if \(epoch \+ 1\) % 25 == 0 or epoch == args\.epochs - 1:[^\n]*\n"
            r"            validationFunc\(val_loader, networks, epoch, args, \{'logger': logger\}\)",
            new_gate, src)
        assert src2 != src, "expected earlier gate not found"
        open(mp, "w", encoding="utf-8").write(src2)
        print("upgraded main.py validation gate with DG_SKIP_VAL override")


if __name__ == "__main__":
    main()
