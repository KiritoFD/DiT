# -*- coding: utf-8 -*-
"""triple 补丁第二段: train.py 的 y_script 定义 + 三表 SupCon 初始化 (T5 误跳过, 手动补)"""
import shutil
import time
import py_compile
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)


def patch(path, old, new, tag, marker):
    src = open(path, encoding="utf-8").read()
    if marker in src:
        print(f"[{tag}] already patched, skip")
        return
    assert src.count(old) == 1, f"[{tag}] anchor x{src.count(old)}"
    bak = f"{path}.bak_triple2_{int(time.time())}"
    shutil.copy2(path, bak)
    open(path, "w", encoding="utf-8", newline="").write(src.replace(old, new))
    py_compile.compile(path, doraise=True)
    print(f"[{tag}] patched (backup -> {bak}) syntax OK")


T = "src/train/train.py"

# P2a: triple 模式下 y_callig 用书家连续索引, 并定义 y_script
patch(T, """                y_callig = batch['y_callig'].to(device, non_blocking=True)""",
      """                y_callig = batch['y_callig'].to(device, non_blocking=True)
                if getattr(args, 'use_script_cond', False):
                    # v53 三表: y_callig = 书家连续索引(10 类), 另带 y_script
                    y_callig = batch['y_callig_raw'].to(device, non_blocking=True)
                    y_script = batch['y_script'].to(device, non_blocking=True)""",
      "P2a-yscript", marker="v53 三表: y_callig 用书家连续索引")

# P2c: 三表 SupCon 初始化 (T5 因 marker 撞 DINO 注释行被误跳过)
patch(T, """    # ── DINO glyph-embedding init for y_char_embedder ───────────────────────""",
      """    # ── v53: 三表重映射 json + SupCon 表初始化 ──────────────────────────────
    _char_remaps = None
    if getattr(args, 'char_remap_json', ''):
        import json as _json
        _char_remaps = {}
        for _nm, _p in (('char', args.char_remap_json),
                        ('callig', getattr(args, 'callig_remap_json', '')),
                        ('font', getattr(args, 'font_remap_json', ''))):
            if _p:
                _char_remaps[_nm] = {int(k): int(v) for k, v in
                                     _json.load(open(_p, encoding="utf-8")).items()}
        logger.info(f"[triple] remap json 已加载: {list(_char_remaps.keys())}")
    _ttp = getattr(args, 'triple_table_prefix', '') or ''
    if _ttp:
        import json as _json
        import numpy as _np
        for _nm, _emb in (('callig', model.y_callig_embedder),
                          ('font', model.y_script_embedder),
                          ('char', model.y_char_embedder)):
            if _emb is None:
                continue
            _w = _np.load(f"{_ttp}{_nm}_table.npy")
            _cls = _json.load(open(f"{_ttp}{_nm}_index.json",
                                   encoding="utf-8"))["classes"]
            _tab = _emb.embedding_table.weight
            assert _w.shape[0] == len(_cls), \\
                f"{_nm}: 表行 {_w.shape[0]} != 类数 {len(_cls)}"
            assert _w.shape[1] == _tab.shape[1], \\
                f"{_nm}: 维度 {_w.shape[1]} != 表 {_tab.shape[1]}"
            with torch.no_grad():
                for _i, _c in enumerate(_cls):
                    _tab[int(_c)].copy_(torch.from_numpy(_w[_i]).float())
        logger.info(f"[triple-init] 三表 SupCon 初始化完成 (prefix={_ttp})")

    # ── DINO glyph-embedding init for y_char_embedder ───────────────────────""",
      "P2c-init", marker="v53: 三表重映射 json + SupCon 表初始化")

print("[P2 DONE]")
